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
#include "static_multimap_perf_common.h"

namespace aclco::test {
namespace {

template <typename Key, typename Value>
struct Context : QueryContext<Key, Value> {
    DeviceBuffer<Key> keyOutput;
    DeviceBuffer<Value> valueOutput;
};

template <typename Key, typename Value>
Context<Key, Value>& GetContext()
{
    return GetContextInstance<Context<Key, Value>>();
}

template <typename Key, typename Value>
void SetupRetrieve(std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity, double matchingRate)
{
    auto& context = GetContext<Key, Value>();
    SetupPairs<Key, Value>(context, numInputs, occupancy, multiplicity);
    context.map->Insert(context.pairs.Data(), aclco::Extent<std::size_t>(context.numInputs),
                        context.streamGuard.stream);
    SetupQueries<Key>(context, matchingRate);
    context.keyOutput.Resize(context.numInputs);
    context.valueOutput.Resize(context.numInputs);
    Sync(context.streamGuard.stream);
}

template <typename Key, typename Value>
TestResult TestRetrieve()
{
    auto& context = GetContext<Key, Value>();
    return Measure([&] {
        (void)context.map->Retrieve(context.queries.Data(), aclco::Extent<std::size_t>(context.numInputs),
                                    context.keyOutput.Data(), context.valueOutput.Data(),
                                    aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultimapRetrieveI32, (TestRetrieve<std::int32_t, std::int32_t>),
                          (SetupRetrieve<std::int32_t, std::int32_t>), std::uint64_t, double, std::uint64_t, double);
REGISTER_PERFORMANCE_TEST(staticMultimapRetrieveI64, (TestRetrieve<std::int64_t, std::int64_t>),
                          (SetupRetrieve<std::int64_t, std::int64_t>), std::uint64_t, double, std::uint64_t, double);

REGISTER_PERFORMANCE_ARGS(staticMultimapRetrieveI32,
                          "StaticMultimap retrieve (NumInputs, Occupancy, Multiplicity, "
                          "MatchingRate) I32",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
REGISTER_PERFORMANCE_ARGS(staticMultimapRetrieveI64,
                          "StaticMultimap retrieve (NumInputs, Occupancy, Multiplicity, "
                          "MatchingRate) I64",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
} // namespace aclco::test
