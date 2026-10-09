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
#include "static_multiset_perf_common.h"

namespace aclco::test {
namespace {
using namespace static_multiset_perf;

constexpr std::uint64_t NUM_INPUTS = 100000000ULL;
constexpr double OCCUPANCY = 0.5;

template <typename Key>
struct Context : QueryContext<Key> {
    DeviceBuffer<std::uint8_t> output;
};

template <typename Key>
void SetupContains(std::uint64_t numInputs, double occupancy, double matchingRate)
{
    auto& context = GetContext<Context<Key>>();
    auto inputs = SetupQueries<Key>(context, numInputs, occupancy, matchingRate);
    context.output.Resize(context.numInputs);
    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestContains()
{
    auto& context = GetContext<Context<Key>>();
    return Measure([&] {
        context.set->Contains(context.queries.Data(), context.output.Data(),
                              aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetContainsI32, (TestContains<std::int32_t>), (SetupContains<std::int32_t>),
                          std::uint64_t, double, double);
REGISTER_PERFORMANCE_TEST(staticMultisetContainsI64, (TestContains<std::int64_t>), (SetupContains<std::int64_t>),
                          std::uint64_t, double, double);

REGISTER_PERFORMANCE_ARGS(
    staticMultisetContainsI32, "StaticMultiset contains (NumInputs, Occupancy, MatchingRate) I32",
    (std::initializer_list<std::tuple<std::uint64_t, double, double>>{
        {NUM_INPUTS, OCCUPANCY, 0.1}, {NUM_INPUTS, OCCUPANCY, 0.5}, {NUM_INPUTS, OCCUPANCY, 1.0}}),
    std::uint64_t, double, double);
REGISTER_PERFORMANCE_ARGS(
    staticMultisetContainsI64, "StaticMultiset contains (NumInputs, Occupancy, MatchingRate) I64",
    (std::initializer_list<std::tuple<std::uint64_t, double, double>>{
        {NUM_INPUTS, OCCUPANCY, 0.1}, {NUM_INPUTS, OCCUPANCY, 0.5}, {NUM_INPUTS, OCCUPANCY, 1.0}}),
    std::uint64_t, double, double);
} // namespace aclco::test
