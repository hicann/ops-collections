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
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include "static_multimap.h"

namespace aclco::test {
namespace {
constexpr std::uint64_t NUM_INPUTS = 100000000ULL;
constexpr std::uint64_t CAPACITY = 100000000ULL;
constexpr double OCCUPANCY = 0.5;
constexpr std::uint64_t MULTIPLICITY = 1ULL;

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

template <typename Key, typename Value>
auto MakeMap(std::size_t capacity, aclrtStream stream)
{
    return aclco::StaticMultimap<Key, Value>(aclco::Extent<std::size_t>(capacity), Empty<Key>(), Empty<Value>(),
                                             stream);
}

template <typename Key, typename Value>
std::vector<aclco::Pair<Key, Value>> MakePairs(std::size_t count)
{
    std::vector<aclco::Pair<Key, Value>> pairs;
    pairs.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto value = static_cast<Key>(i + 1);
        pairs.emplace_back(value, static_cast<Value>(value));
    }
    return pairs;
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

template <class ContextType>
ContextType& GetContextInstance()
{
    static ContextType context;
    return context;
}

template <typename Key, typename Value>
struct CapacityContext {
    AclStreamGuard streamGuard;
    std::size_t capacity{};
};

template <typename Key, typename Value>
struct InputContext {
    AclStreamGuard streamGuard;
    std::size_t numInputs{};
    std::optional<aclco::StaticMultimap<Key, Value>> map;
    DeviceBuffer<aclco::Pair<Key, Value>> pairs;
};

template <typename Key, typename Value>
struct QueryContext : InputContext<Key, Value> {
    DeviceBuffer<Key> queries;
};

template <typename Key, typename Value, class ContextType>
void SetupPairs(ContextType& context, std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity)
{
    if (multiplicity != 1) {
        throw std::invalid_argument("Multiplicity must be 1");
    }
    if (!(occupancy > 0.0 && occupancy <= 1.0)) {
        throw std::invalid_argument("Occupancy must be in (0, 1]");
    }
    context.numInputs = static_cast<std::size_t>(numInputs);
    context.map.emplace(MakeMap<Key, Value>(static_cast<std::size_t>(static_cast<double>(numInputs) / occupancy),
                                            context.streamGuard.stream));
    auto pairs = MakePairs<Key, Value>(context.numInputs);
    context.pairs.Resize(context.numInputs);
    context.pairs.CopyFromHostAsync(pairs.data(), pairs.size(), context.streamGuard.stream);
    // The copy source is local to this helper; finish the upload before it is destroyed.
    Sync(context.streamGuard.stream);
}

template <typename Key, class ContextType>
void SetupQueries(ContextType& context, double matchingRate)
{
    auto queries = MakeQueries<Key>(context.numInputs, matchingRate);
    context.queries.Resize(context.numInputs);
    context.queries.CopyFromHostAsync(queries.data(), queries.size(), context.streamGuard.stream);
    Sync(context.streamGuard.stream);
}

template <typename Key, typename Value>
void SetupCapacity(std::uint64_t capacity)
{
    GetContextInstance<CapacityContext<Key, Value>>().capacity = static_cast<std::size_t>(capacity);
}

template <typename Key, typename Value, typename Output>
struct OutputContext : QueryContext<Key, Value> {
    DeviceBuffer<Output> output;
};

template <typename Key, typename Value, typename Output>
void SetupOutputQuery(std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity, double matchingRate)
{
    auto& context = GetContextInstance<OutputContext<Key, Value, Output>>();
    SetupPairs<Key, Value>(context, numInputs, occupancy, multiplicity);
    context.map->Insert(context.pairs.Data(), aclco::Extent<std::size_t>(context.numInputs),
                        context.streamGuard.stream);
    SetupQueries<Key>(context, matchingRate);
    context.output.Resize(context.numInputs);
    Sync(context.streamGuard.stream);
}
} // namespace
} // namespace aclco::test
