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
TestResult TestCreate()
{
    auto& context = GetContextInstance<CapacityContext<Key, Value>>();
    std::optional<aclco::StaticMultimap<Key, Value>> map;
    auto result = Measure([&] { map.emplace(MakeMap<Key, Value>(context.capacity, context.streamGuard.stream)); });
    map.reset();
    return result;
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultimapCreateI32, (TestCreate<std::int32_t, std::int32_t>),
                          (SetupCapacity<std::int32_t, std::int32_t>), std::uint64_t);
REGISTER_PERFORMANCE_TEST(staticMultimapCreateI64, (TestCreate<std::int64_t, std::int64_t>),
                          (SetupCapacity<std::int64_t, std::int64_t>), std::uint64_t);

REGISTER_PERFORMANCE_ARGS(staticMultimapCreateI32, "StaticMultimap create (Capacity) I32",
                          (std::initializer_list<std::tuple<std::uint64_t>>{{CAPACITY}}), std::uint64_t);
REGISTER_PERFORMANCE_ARGS(staticMultimapCreateI64, "StaticMultimap create (Capacity) I64",
                          (std::initializer_list<std::tuple<std::uint64_t>>{{CAPACITY}}), std::uint64_t);
} // namespace aclco::test
