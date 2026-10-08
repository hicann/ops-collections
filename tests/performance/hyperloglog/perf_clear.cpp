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
struct Context {
    AclStreamGuard streamGuard;
    std::optional<aclco::HyperLogLog<Key>> hll;
};

template <typename Key>
void SetupClear(std::uint32_t sketchSizeKb)
{
    auto& context = GetContext<Context<Key>>();
    context.hll.emplace(MakeHll<Key>(sketchSizeKb, context.streamGuard.stream));
}

template <typename Key>
TestResult TestClear()
{
    auto& context = GetContext<Context<Key>>();
    return Measure([&] { context.hll->Clear(context.streamGuard.stream); });
}
} // namespace

REGISTER_PERFORMANCE_TEST(hyperLogLogClearI32, (TestClear<std::int32_t>), (SetupClear<std::int32_t>), std::uint32_t);
REGISTER_PERFORMANCE_TEST(hyperLogLogClearI64, (TestClear<std::int64_t>), (SetupClear<std::int64_t>), std::uint32_t);

REGISTER_PERFORMANCE_ARGS(hyperLogLogClearI32, "HyperLogLog clear (SketchSizeKB) I32",
                          (std::initializer_list<std::tuple<std::uint32_t>>{{8}, {16}, {32}, {64}, {128}, {256}}),
                          std::uint32_t);
REGISTER_PERFORMANCE_ARGS(hyperLogLogClearI64, "HyperLogLog clear (SketchSizeKB) I64",
                          (std::initializer_list<std::tuple<std::uint32_t>>{{8}, {16}, {32}, {64}, {128}, {256}}),
                          std::uint32_t);
} // namespace aclco::test
