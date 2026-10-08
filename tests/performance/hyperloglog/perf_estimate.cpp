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
#include "hyperloglog_perf_common.h"

namespace aclco::test {
namespace {
using namespace hll_performance;

template <typename Key>
using Context = InputContext<Key>;

template <typename Key>
void SetupEstimate(std::uint64_t numInputs, std::uint32_t sketchSizeKb)
{
    auto& context = GetContext<Context<Key>>();
    context.numInputs = static_cast<std::size_t>(numInputs);
    context.hll.emplace(MakeHll<Key>(sketchSizeKb, context.streamGuard.stream));

    auto keys = MakeKeys<Key>(context.numInputs);
    context.keys.Resize(context.numInputs);
    context.keys.CopyFromHostAsync(keys.data(), keys.size(), context.streamGuard.stream);
    context.hll->Add(context.keys.Data(), aclco::Extent<std::size_t>(context.numInputs), context.streamGuard.stream);

    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestEstimate()
{
    auto& context = GetContext<Context<Key>>();
    return Measure([&] { (void)context.hll->Estimate(context.streamGuard.stream); });
}
} // namespace

REGISTER_PERFORMANCE_TEST(hyperLogLogEstimateI32, (TestEstimate<std::int32_t>), (SetupEstimate<std::int32_t>),
                          std::uint64_t, std::uint32_t);
REGISTER_PERFORMANCE_TEST(hyperLogLogEstimateI64, (TestEstimate<std::int64_t>), (SetupEstimate<std::int64_t>),
                          std::uint64_t, std::uint32_t);

REGISTER_PERFORMANCE_ARGS(
    hyperLogLogEstimateI32, "HyperLogLog estimate (NumInputs, SketchSizeKB) I32",
    (std::initializer_list<std::tuple<std::uint64_t, std::uint32_t>>{
        {NUM_INPUTS, 8}, {NUM_INPUTS, 16}, {NUM_INPUTS, 32}, {NUM_INPUTS, 64}, {NUM_INPUTS, 128}, {NUM_INPUTS, 256}}),
    std::uint64_t, std::uint32_t);
REGISTER_PERFORMANCE_ARGS(
    hyperLogLogEstimateI64, "HyperLogLog estimate (NumInputs, SketchSizeKB) I64",
    (std::initializer_list<std::tuple<std::uint64_t, std::uint32_t>>{
        {NUM_INPUTS, 8}, {NUM_INPUTS, 16}, {NUM_INPUTS, 32}, {NUM_INPUTS, 64}, {NUM_INPUTS, 128}, {NUM_INPUTS, 256}}),
    std::uint64_t, std::uint32_t);
} // namespace aclco::test
