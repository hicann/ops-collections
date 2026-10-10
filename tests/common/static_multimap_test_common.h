/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#pragma once

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_test_case_info.hpp>
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "extent.h"
#include "pair.h"
#include "static_multimap.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace {
class MultimapProgressListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testCasePartialStarting(Catch::TestCaseInfo const& info, uint64_t partNumber) override
    {
        std::cout << "[ RUN      ] " << info.name << " iteration " << partNumber + 1 << std::endl;
    }

    void testCasePartialEnded(Catch::TestCaseStats const& stats, uint64_t partNumber) override
    {
        std::cout << "[ COMPLETE ] " << stats.testInfo->name << " iteration " << partNumber + 1
                  << ", failed assertions: " << stats.totals.assertions.failed << std::endl;
    }
};
CATCH_REGISTER_LISTENER(MultimapProgressListener)
} // namespace

namespace aclco::test::static_multimap {

struct TestStream {
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard guard;
    aclrtStream stream = guard.stream;
};

template <typename T>
aclco::test::DeviceBuffer<T> UploadBuffer(std::vector<T> const& values, aclrtStream stream)
{
    aclco::test::DeviceBuffer<T> buffer(values.size());
    buffer.CopyFromHostAsync(values.data(), values.size(), stream);
    return buffer;
}

template <typename Key, typename Value>
aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> InsertPairs(aclco::StaticMultimap<Key, Value>& map,
                                                               std::vector<aclco::Pair<Key, Value>> const& pairs,
                                                               aclrtStream stream)
{
    auto buffer = UploadBuffer(pairs, stream);
    map.Insert(buffer.Data(), aclco::Extent<std::size_t>(pairs.size()), stream);
    return buffer;
}

inline std::vector<uint32_t> MakeStencil(std::size_t count, uint32_t mode)
{
    std::vector<uint32_t> stencil(count);
    for (std::size_t i = 0; i < count; ++i) {
        stencil[i] = mode == 0u ? 0u : (mode == 1u ? 1u : static_cast<uint32_t>(i));
    }
    return stencil;
}

template <typename Key>
constexpr Key EmptyKey()
{
    return std::numeric_limits<Key>::lowest();
}

template <typename Value>
constexpr Value EmptyValue()
{
    return std::numeric_limits<Value>::lowest();
}

template <typename Key, typename Value>
auto MakeMultimap(std::size_t capacity, aclrtStream stream)
{
    return aclco::StaticMultimap<Key, Value>(aclco::Extent<std::size_t>(capacity), EmptyKey<Key>(), EmptyValue<Value>(),
                                             stream);
}

template <typename Key, typename Value>
std::vector<aclco::Pair<Key, Value>> MakePairs(std::size_t distinct, std::size_t multiplicity)
{
    std::vector<aclco::Pair<Key, Value>> pairs;
    pairs.reserve(distinct * multiplicity);
    for (std::size_t i = 0; i < distinct; ++i) {
        for (std::size_t repeat = 0; repeat < multiplicity; ++repeat) {
            pairs.emplace_back(static_cast<Key>(i + 1), static_cast<Value>((i + 1) * 1024 + repeat + 1));
        }
    }
    return pairs;
}

template <typename Key>
std::vector<Key> MakeQueries(std::size_t inserted_distinct, std::size_t query_num, double matching_rate)
{
    const auto hit_num = static_cast<std::size_t>(query_num * matching_rate);
    std::vector<Key> queries;
    queries.reserve(query_num);
    for (std::size_t i = 0; i < hit_num; ++i) {
        queries.push_back(static_cast<Key>((i % std::max<std::size_t>(inserted_distinct, 1)) + 1));
    }
    for (std::size_t i = hit_num; i < query_num; ++i) {
        queries.push_back(static_cast<Key>(inserted_distinct + i + 4096));
    }
    // Interleave hits and misses instead of testing only clustered output.
    for (std::size_t i = 1; i < queries.size(); i += 2) {
        std::swap(queries[i], queries[queries.size() - 1 - i / 2]);
    }
    return queries;
}

template <typename Key, typename Value>
std::unordered_multimap<Key, Value> MapOracle(const std::vector<aclco::Pair<Key, Value>>& pairs)
{
    std::unordered_multimap<Key, Value> oracle;
    oracle.reserve(pairs.size());
    for (auto const& p : pairs) {
        oracle.emplace(p.first, p.second);
    }
    return oracle;
}

template <typename Key, typename Value>
struct InsertedMap {
    TestStream context;
    aclco::StaticMultimap<Key, Value> map;
    std::vector<aclco::Pair<Key, Value>> pairs;
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> devicePairs;

    InsertedMap(std::size_t capacity, std::size_t distinct, std::size_t multiplicity)
        : map(MakeMultimap<Key, Value>(capacity, context.stream)),
          pairs(MakePairs<Key, Value>(distinct, multiplicity)),
          devicePairs(InsertPairs(map, pairs, context.stream))
    {}
};

inline std::size_t QueryDistinct(std::size_t capacity, std::size_t multiplicity)
{
    if (multiplicity == 0) {
        throw std::invalid_argument("Multiplicity must be positive");
    }
    return std::max<std::size_t>(1u, capacity / 2u / multiplicity);
}

template <typename Key, typename Value>
struct QueryFixture : InsertedMap<Key, Value> {
    std::unordered_multimap<Key, Value> oracle;
    std::vector<Key> queries;
    aclco::test::DeviceBuffer<Key> deviceQueries;

    QueryFixture(std::size_t capacity, std::size_t multiplicity, double matchingRate)
        : InsertedMap<Key, Value>(capacity, QueryDistinct(capacity, multiplicity), multiplicity),
          oracle(MapOracle(this->pairs)),
          queries(MakeQueries<Key>(QueryDistinct(capacity, multiplicity), std::min<std::size_t>(4096u, capacity),
                                   matchingRate)),
          deviceQueries(UploadBuffer(queries, this->context.stream))
    {}
};

template <typename Key, typename Value>
struct StencilFixture : InsertedMap<Key, Value> {
    std::vector<Key> keys;
    std::vector<uint32_t> stencil;
    aclco::test::DeviceBuffer<Key> deviceKeys;
    aclco::test::DeviceBuffer<uint32_t> deviceStencil;

    StencilFixture(std::size_t capacity, std::size_t count, double matchingRate, uint32_t mode)
        : InsertedMap<Key, Value>(capacity, count, 1u),
          keys(MakeQueries<Key>(count, count, matchingRate)),
          stencil(MakeStencil(count, mode)),
          deviceKeys(UploadBuffer(keys, this->context.stream)),
          deviceStencil(UploadBuffer(stencil, this->context.stream))
    {}
};

template <typename Key, typename Oracle>
uint64_t ExpectedCount(Oracle const& oracle, std::vector<Key> const& queries)
{
    uint64_t total = 0;
    for (auto key : queries) {
        total += static_cast<uint64_t>(oracle.count(key));
    }
    return total;
}

template <typename Key, typename Value>
std::vector<std::pair<Key, Value>> NormalizePairs(std::vector<Key> const& keys, std::vector<Value> const& values,
                                                  std::size_t count)
{
    REQUIRE(keys.size() >= count);
    REQUIRE(values.size() >= count);
    std::vector<std::pair<Key, Value>> out;
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.emplace_back(keys[i], values[i]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

template <typename Key, typename Value>
std::vector<std::pair<Key, Value>> NormalizePairs(std::vector<aclco::Pair<Key, Value>> const& pairs)
{
    std::vector<std::pair<Key, Value>> out;
    out.reserve(pairs.size());
    for (auto const& p : pairs) {
        out.emplace_back(p.first, p.second);
    }
    std::sort(out.begin(), out.end());
    return out;
}

template <typename T>
void RequireEqual(std::vector<T> const& actual, std::vector<T> const& expected)
{
    REQUIRE(actual.size() == expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        REQUIRE(actual[i] == expected[i]);
    }
}

} // namespace aclco::test::static_multimap
