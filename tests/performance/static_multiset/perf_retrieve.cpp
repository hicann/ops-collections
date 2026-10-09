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
struct Context : QueryContext<Key> {
    DeviceBuffer<Key> keyOutput;
    DeviceBuffer<Key> matchOutput;
};

template <typename Key>
void SetupRetrieve(std::uint64_t numInputs, double occupancy, double matchingRate, std::uint64_t multiplicity)
{
    if (multiplicity != 1)
        throw std::invalid_argument("Multiplicity must be 1");
    auto& context = GetContext<Context<Key>>();
    auto inputs = SetupQueries<Key>(context, numInputs, occupancy, matchingRate);
    context.keyOutput.Resize(context.numInputs);
    context.matchOutput.Resize(context.numInputs);
    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestRetrieve()
{
    auto& context = GetContext<Context<Key>>();
    return Measure([&] {
        (void)context.set->Retrieve(context.queries.Data(), aclco::Extent<std::size_t>(context.numInputs),
                                    context.keyOutput.Data(), context.matchOutput.Data(),
                                    aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetRetrieveI32, (TestRetrieve<std::int32_t>), (SetupRetrieve<std::int32_t>),
                          std::uint64_t, double, double, std::uint64_t);
REGISTER_PERFORMANCE_TEST(staticMultisetRetrieveI64, (TestRetrieve<std::int64_t>), (SetupRetrieve<std::int64_t>),
                          std::uint64_t, double, double, std::uint64_t);

REGISTER_PERFORMANCE_ARGS(staticMultisetRetrieveI32,
                          "StaticMultiset retrieve (NumInputs, Occupancy, MatchingRate, "
                          "Multiplicity) I32",
                          (std::initializer_list<std::tuple<std::uint64_t, double, double, std::uint64_t>>{
                              {NUM_INPUTS, OCCUPANCY, 0.1, MULTIPLICITY},
                              {NUM_INPUTS, OCCUPANCY, 0.5, MULTIPLICITY},
                              {NUM_INPUTS, OCCUPANCY, 1.0, MULTIPLICITY}}),
                          std::uint64_t, double, double, std::uint64_t);
REGISTER_PERFORMANCE_ARGS(staticMultisetRetrieveI64,
                          "StaticMultiset retrieve (NumInputs, Occupancy, MatchingRate, "
                          "Multiplicity) I64",
                          (std::initializer_list<std::tuple<std::uint64_t, double, double, std::uint64_t>>{
                              {NUM_INPUTS, OCCUPANCY, 0.1, MULTIPLICITY},
                              {NUM_INPUTS, OCCUPANCY, 0.5, MULTIPLICITY},
                              {NUM_INPUTS, OCCUPANCY, 1.0, MULTIPLICITY}}),
                          std::uint64_t, double, double, std::uint64_t);
} // namespace aclco::test
