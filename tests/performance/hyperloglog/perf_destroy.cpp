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
using Context = SketchSizeContext<Key>;

template <typename Key>
void SetupDestroy(std::uint32_t sketchSizeKb)
{
    GetContext<Context<Key>>().sketchSizeKb = sketchSizeKb;
}

template <typename Key>
TestResult TestDestroy()
{
    auto& context = GetContext<Context<Key>>();
    auto hll = std::make_unique<aclco::HyperLogLog<Key>>(
        MakeHll<Key>(context.sketchSizeKb, context.streamGuard.stream));
    return Measure([&] { hll.reset(); });
}
} // namespace

REGISTER_PERFORMANCE_TEST(hyperLogLogDestroyI32, (TestDestroy<std::int32_t>), (SetupDestroy<std::int32_t>),
                          std::uint32_t);
REGISTER_PERFORMANCE_TEST(hyperLogLogDestroyI64, (TestDestroy<std::int64_t>), (SetupDestroy<std::int64_t>),
                          std::uint32_t);

REGISTER_PERFORMANCE_ARGS(hyperLogLogDestroyI32, "HyperLogLog destroy (SketchSizeKB) I32",
                          (std::initializer_list<std::tuple<std::uint32_t>>{{8}, {16}, {32}, {64}, {128}, {256}}),
                          std::uint32_t);
REGISTER_PERFORMANCE_ARGS(hyperLogLogDestroyI64, "HyperLogLog destroy (SketchSizeKB) I64",
                          (std::initializer_list<std::tuple<std::uint32_t>>{{8}, {16}, {32}, {64}, {128}, {256}}),
                          std::uint32_t);
} // namespace aclco::test
