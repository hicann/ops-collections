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
struct Context : InputContext<Key> {
    DeviceBuffer<Key> output;
};

template <typename Key>
void SetupRetrieveAll(std::uint64_t numInputs, double occupancy, std::uint64_t multiplicity)
{
    if (multiplicity != 1)
        throw std::invalid_argument("Multiplicity must be 1");
    auto& context = GetContext<Context<Key>>();
    auto keys = SetupInput<Key>(context, numInputs, occupancy);
    context.set->Insert(context.keys.Data(), aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);
    context.output.Resize(context.numInputs);
    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestRetrieveAll()
{
    auto& context = GetContext<Context<Key>>();
    return Measure([&] {
        (void)context.set->RetrieveAll(context.output.Data(), aclco::Extent<std::size_t>(context.numInputs),
                                       context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetRetrieveAllI32, (TestRetrieveAll<std::int32_t>),
                          (SetupRetrieveAll<std::int32_t>), std::uint64_t, double, std::uint64_t);
REGISTER_PERFORMANCE_TEST(staticMultisetRetrieveAllI64, (TestRetrieveAll<std::int64_t>),
                          (SetupRetrieveAll<std::int64_t>), std::uint64_t, double, std::uint64_t);

REGISTER_PERFORMANCE_ARGS(staticMultisetRetrieveAllI32,
                          "StaticMultiset retrieve_all (NumInputs, Occupancy, Multiplicity) I32",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY}}),
                          std::uint64_t, double, std::uint64_t);
REGISTER_PERFORMANCE_ARGS(staticMultisetRetrieveAllI64,
                          "StaticMultiset retrieve_all (NumInputs, Occupancy, Multiplicity) I64",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY}}),
                          std::uint64_t, double, std::uint64_t);
} // namespace aclco::test
