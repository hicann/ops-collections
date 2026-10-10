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
TestResult TestFind()
{
    auto& context = GetContextInstance<OutputContext<Key, Value, Value>>();
    return Measure([&] {
        context.map->Find(context.queries.Data(), context.output.Data(), aclco::Extent<std::size_t>(context.numInputs),
                          context.streamGuard.stream);
    });
}
} // namespace

REGISTER_PERFORMANCE_TEST(staticMultimapFindI32, (TestFind<std::int32_t, std::int32_t>),
                          (SetupOutputQuery<std::int32_t, std::int32_t, std::int32_t>), std::uint64_t, double,
                          std::uint64_t, double);
REGISTER_PERFORMANCE_TEST(staticMultimapFindI64, (TestFind<std::int64_t, std::int64_t>),
                          (SetupOutputQuery<std::int64_t, std::int64_t, std::int64_t>), std::uint64_t, double,
                          std::uint64_t, double);

REGISTER_PERFORMANCE_ARGS(staticMultimapFindI32,
                          "StaticMultimap find (NumInputs, Occupancy, Multiplicity, MatchingRate) "
                          "I32",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
REGISTER_PERFORMANCE_ARGS(staticMultimapFindI64,
                          "StaticMultimap find (NumInputs, Occupancy, Multiplicity, MatchingRate) "
                          "I64",
                          (std::initializer_list<std::tuple<std::uint64_t, double, std::uint64_t, double>>{
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.1},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 0.5},
                              {NUM_INPUTS, OCCUPANCY, MULTIPLICITY, 1.0}}),
                          std::uint64_t, double, std::uint64_t, double);
} // namespace aclco::test
