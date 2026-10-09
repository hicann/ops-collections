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
constexpr std::uint64_t MULTIPLICITY = 1ULL;

template <typename Key>
using Context = QueryContext<Key>;

template <typename Key>
void SetupCount(std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity, double matchingRate)
{
    if (multiplicity != 1)
        throw std::invalid_argument("Multiplicity must be 1");
    auto& context = GetContext<Context<Key>>();
    auto inputs = SetupQueries<Key>(context, numInputs, occupancy, matchingRate);

    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestCount()
{
    auto& context = GetContext<Context<Key>>();
    return Measure([&] {
        (void)context.set->Count(context.queries.Data(), aclco::Extent<std::size_t>(context.numInputs),
                                 context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetCountI32, (TestCount<std::int32_t>), (SetupCount<std::int32_t>), std::uint64_t,
                          double, std::uint64_t, double);
REGISTER_PERFORMANCE_TEST(staticMultisetCountI64, (TestCount<std::int64_t>), (SetupCount<std::int64_t>), std::uint64_t,
                          double, std::uint64_t, double);

REGISTER_PERFORMANCE_ARGS(staticMultisetCountI32,
                          "StaticMultiset count (NumInputs, Occupancy, Multiplicity, MatchingRate) "
                          "I32",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
REGISTER_PERFORMANCE_ARGS(staticMultisetCountI64,
                          "StaticMultiset count (NumInputs, Occupancy, Multiplicity, MatchingRate) "
                          "I64",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
} // namespace aclco::test
