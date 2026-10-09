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

#include "../performance_test_framework.h"

#include <chrono>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

#include "static_multiset.h"

namespace aclco::test::static_multiset_perf {

template <typename Func>
TestResult Measure(Func&& func)
{
    auto const start = std::chrono::high_resolution_clock::now();
    func();
    auto const stop = std::chrono::high_resolution_clock::now();
    auto const cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return TestResult(static_cast<double>(cpu), static_cast<double>(cpu), 0);
}

template <typename T>
constexpr T Empty()
{
    return std::numeric_limits<T>::lowest();
}

template <typename Key>
auto MakeSet(std::size_t capacity, aclrtStream stream)
{
    return aclco::StaticMultiset<Key>(aclco::Extent<std::size_t>(capacity), Empty<Key>(), stream);
}

template <typename Key>
std::vector<Key> MakeKeys(std::size_t count)
{
    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i)
        keys[i] = static_cast<Key>(i + 1);
    return keys;
}

template <typename Key>
std::vector<Key> MakeQueries(std::size_t count, double matchingRate)
{
    auto const hits = static_cast<std::size_t>(static_cast<double>(count) * matchingRate);
    std::vector<Key> queries(count);
    for (std::size_t i = 0; i < hits; ++i)
        queries[i] = static_cast<Key>(i + 1);
    for (std::size_t i = hits; i < count; ++i)
        queries[i] = static_cast<Key>(count + i + 1);
    return queries;
}

template <typename Context>
Context& GetContext()
{
    static Context context;
    return context;
}

template <typename Key>
struct LifecycleContext {
    AclStreamGuard streamGuard;
    std::size_t capacity{};
};

template <typename Key>
struct InputContext {
    AclStreamGuard streamGuard;
    std::size_t numInputs{};
    std::optional<aclco::StaticMultiset<Key>> set;
    DeviceBuffer<Key> keys;
};

template <typename Key>
struct QueryContext : InputContext<Key> {
    DeviceBuffer<Key> queries;
};

template <typename Key>
struct CountContext : QueryContext<Key> {
    DeviceBuffer<std::uint64_t> output;
};

inline std::size_t CapacityFor(std::uint64_t numInputs, double occupancy)
{
    if (!(occupancy > 0.0 && occupancy <= 1.0)) {
        throw std::invalid_argument("Occupancy must be in (0, 1]");
    }
    double capacity = static_cast<double>(numInputs) / occupancy;
    if (capacity >= static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("Capacity exceeds size_t");
    }
    return static_cast<std::size_t>(capacity);
}

template <typename Key>
std::vector<Key> SetupInput(InputContext<Key>& context, std::uint64_t numInputs, double occupancy)
{
    context.numInputs = static_cast<std::size_t>(numInputs);
    context.set.emplace(MakeSet<Key>(CapacityFor(numInputs, occupancy), context.streamGuard.stream));
    auto keys = MakeKeys<Key>(context.numInputs);
    context.keys.Resize(context.numInputs);
    context.keys.CopyFromHostAsync(keys.data(), keys.size(), context.streamGuard.stream);
    return keys;
}

template <typename Key>
std::pair<std::vector<Key>, std::vector<Key>> SetupQueries(QueryContext<Key>& context, std::uint64_t numInputs,
                                                           double occupancy, double matchingRate)
{
    auto keys = SetupInput<Key>(context, numInputs, occupancy);
    context.set->Insert(context.keys.Data(), aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);
    auto queries = MakeQueries<Key>(context.numInputs, matchingRate);
    context.queries.Resize(context.numInputs);
    context.queries.CopyFromHostAsync(queries.data(), queries.size(), context.streamGuard.stream);
    return {std::move(keys), std::move(queries)};
}
} // namespace aclco::test::static_multiset_perf
