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

constexpr std::uint64_t CAPACITY = 100000000ULL;

template <typename Key>
using Context = LifecycleContext<Key>;

template <typename Key>
void SetupDestroy(std::uint64_t capacity)
{
    GetContext<Context<Key>>().capacity = static_cast<std::size_t>(capacity);
}

template <typename Key>
TestResult TestDestroy()
{
    auto& context = GetContext<Context<Key>>();
    auto set = std::make_unique<aclco::StaticMultiset<Key>>(MakeSet<Key>(context.capacity, context.streamGuard.stream));
    return Measure([&] { set.reset(); });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultisetDestroyI32, (TestDestroy<std::int32_t>), (SetupDestroy<std::int32_t>),
                          std::uint64_t);
REGISTER_PERFORMANCE_TEST(staticMultisetDestroyI64, (TestDestroy<std::int64_t>), (SetupDestroy<std::int64_t>),
                          std::uint64_t);

REGISTER_PERFORMANCE_ARGS(staticMultisetDestroyI32, "StaticMultiset destroy (Capacity) I32",
                          (std::initializer_list<std::tuple<std::uint64_t>>{{CAPACITY}}), std::uint64_t);
REGISTER_PERFORMANCE_ARGS(staticMultisetDestroyI64, "StaticMultiset destroy (Capacity) I64",
                          (std::initializer_list<std::tuple<std::uint64_t>>{{CAPACITY}}), std::uint64_t);
} // namespace aclco::test
