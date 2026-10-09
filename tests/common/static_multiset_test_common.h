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
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "extent.h"
#include "static_multiset.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::static_multiset {

namespace {
class TestProgressListener : public Catch::EventListenerBase {
public:
    using Catch::EventListenerBase::EventListenerBase;

    void testCasePartialStarting(Catch::TestCaseInfo const& info, uint64_t partNumber) override
    {
        std::cout << "[ RUN      ] " << info.name << " (iteration " << partNumber + 1 << ")" << std::endl;
    }

    void sectionStarting(Catch::SectionInfo const& info) override
    {
        std::cout << "[ SECTION  ] " << info.name << std::endl;
    }

    void sectionEnded(Catch::SectionStats const& stats) override
    {
        std::cout << "[ END      ] " << stats.sectionInfo.name << " (" << stats.durationInSeconds << " s)" << std::endl;
    }

    void testCasePartialEnded(Catch::TestCaseStats const& stats, uint64_t partNumber) override
    {
        std::cout << (stats.totals.assertions.failed == 0 ? "[       OK ] " : "[  FAILED  ] ") << stats.testInfo->name
                  << " (iteration " << partNumber + 1 << ")" << std::endl;
    }
};

CATCH_REGISTER_LISTENER(TestProgressListener)
} // namespace

struct TestEnvironment {
    AclGlobalGuard acl;
    AclStreamGuard streamGuard;

    aclrtStream Stream() const { return streamGuard.stream; }
};

template <typename Key>
constexpr Key EmptyKey()
{
    return std::numeric_limits<Key>::lowest();
}

template <typename Key>
auto MakeMultiset(std::size_t capacity, aclrtStream stream)
{
    return aclco::StaticMultiset<Key>(aclco::Extent<std::size_t>(capacity), EmptyKey<Key>(), stream);
}

template <typename Key>
std::vector<Key> MakeRepeatedKeys(std::size_t distinct, std::size_t multiplicity)
{
    std::vector<Key> keys;
    keys.reserve(distinct * multiplicity);
    for (std::size_t i = 0; i < distinct; ++i) {
        for (std::size_t repeat = 0; repeat < multiplicity; ++repeat) {
            keys.push_back(static_cast<Key>(i + 1));
        }
    }
    return keys;
}

template <typename Key>
DeviceBuffer<Key> InsertKeys(aclco::StaticMultiset<Key>& set, const std::vector<Key>& keys, aclrtStream stream)
{
    DeviceBuffer<Key> input(keys.size());
    input.CopyFromHostAsync(keys.data(), keys.size(), stream);
    set.Insert(input.Data(), aclco::Extent<std::size_t>(keys.size()), stream);
    return input;
}

struct CountParameters {
    std::size_t capacity, requestedMultiplicity, multiplicity;
    double matchingRate;
    std::size_t distinct;
};

inline CountParameters GenerateCountParameters()
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto requestedMultiplicity = GENERATE(1u, 2u, 4u, 8u, 1024u);
    auto multiplicity = std::min<std::size_t>(requestedMultiplicity, capacity / 2u);
    auto matchingRate = GENERATE(0.0, 0.1, 0.5, 1.0);
    auto distinct = std::max<std::size_t>(1u, capacity / (2u * multiplicity));
    return {capacity, requestedMultiplicity, multiplicity, matchingRate, distinct};
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

template <typename Key>
std::unordered_multiset<Key> SetOracle(const std::vector<Key>& keys)
{
    return std::unordered_multiset<Key>(keys.begin(), keys.end());
}

struct LookupParameters {
    std::size_t capacity, multiplicity;
    double matchingRate;
    std::size_t distinct;
};

inline LookupParameters GenerateLookupParameters()
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto matchingRate = GENERATE(0.0, 0.1, 0.5, 1.0);
    auto distinct = std::max<std::size_t>(1u, capacity / (2u * multiplicity));
    return {capacity, multiplicity, matchingRate, distinct};
}

struct ConditionalParameters {
    std::size_t capacity;
    double matchingRate;
    uint32_t stencilMode;
    std::size_t count;
};

inline ConditionalParameters GenerateConditionalParameters()
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto matchingRate = GENERATE(0.0, 0.5, 1.0);
    auto stencilMode = GENERATE(0u, 1u, 2u);
    auto count = std::min<std::size_t>(capacity / 2u, 4096u);
    return {capacity, matchingRate, stencilMode, count};
}

template <typename Key>
struct QueryInput {
    aclco::StaticMultiset<Key> set;
    std::vector<Key> keys;
    std::unordered_multiset<Key> oracle;
    DeviceBuffer<Key> input;
    std::vector<Key> queries;

    QueryInput(std::size_t capacity, std::size_t distinct, std::size_t multiplicity, std::size_t queryCount,
               double matchingRate, aclrtStream stream)
        : set(MakeMultiset<Key>(capacity, stream)),
          keys(MakeRepeatedKeys<Key>(distinct, multiplicity)),
          oracle(SetOracle(keys)),
          input(InsertKeys(set, keys, stream)),
          queries(MakeQueries<Key>(distinct, queryCount, matchingRate))
    {}
};

template <typename Key>
struct ConditionalQueryInput {
    aclco::StaticMultiset<Key> set;
    std::vector<Key> keys;
    DeviceBuffer<Key> input;
    std::vector<Key> queries;
    std::vector<uint32_t> stencil;

    ConditionalQueryInput(std::size_t capacity, std::size_t count, double matchingRate, uint32_t stencilMode,
                          aclrtStream stream)
        : set(MakeMultiset<Key>(capacity, stream)),
          keys(MakeRepeatedKeys<Key>(count, 1u)),
          input(InsertKeys(set, keys, stream)),
          queries(MakeQueries<Key>(count, count, matchingRate)),
          stencil(MakeStencil(count, stencilMode))
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

template <typename Key, typename Oracle>
std::vector<uint64_t> ExpectedCountEach(Oracle const& oracle, std::vector<Key> const& queries, bool outer = false)
{
    std::vector<uint64_t> expected;
    expected.reserve(queries.size());
    for (auto key : queries) {
        const auto count = static_cast<uint64_t>(oracle.count(key));
        expected.push_back(outer ? std::max<uint64_t>(count, 1) : count);
    }
    return expected;
}

template <typename Key>
std::vector<Key> NormalizeKeys(std::vector<Key> keys, std::size_t count)
{
    REQUIRE(keys.size() >= count);
    keys.resize(count);
    std::sort(keys.begin(), keys.end());
    return keys;
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

} // namespace aclco::test::static_multiset
