/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "bloom_filter_perf_common.h"

namespace aclco::test {
namespace {

template <typename Key>
using Filter = bloom_filter_perf::Filter<Key>;

template <typename Key>
using Context = bloom_filter_perf::PairFilterContext<Filter<Key>>;

template <typename Key>
void SetupIntersect(int filterSizeMiB, std::uint64_t numInputs)
{
    auto& context = bloom_filter_perf::PreparePairFilterContext<Key>(filterSizeMiB, numInputs);
    // No key input is allocated for the linear word-wise operation.
    bloom_filter_perf::FillFilter(*context.destination, 0xffu, context.stream);
    bloom_filter_perf::FillFilter(*context.source, 0xa5u, context.stream);
    context.destination->IntersectAsync(*context.source, context.stream);
    Sync(context.stream);
}

template <typename Key>
TestResult TestIntersect()
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    return bloom_filter_perf::MeasureAsync(context.timer, context.stream, [&context] {
        context.destination->IntersectAsync(*context.source, context.stream);
    });
}

} // namespace

REGISTER_PERFORMANCE_TEST(bloomFilterIntersectI32, (TestIntersect<std::uint32_t>), (SetupIntersect<std::uint32_t>), int,
                          std::uint64_t);
REGISTER_PERFORMANCE_TEST(bloomFilterIntersectI64, (TestIntersect<std::uint64_t>), (SetupIntersect<std::uint64_t>), int,
                          std::uint64_t);

REGISTER_PERFORMANCE_ARGS(bloomFilterIntersectI32, "BloomFilter Intersect I32 H1V8 (FilterSizeMiB, NumInput)",
                          (std::initializer_list<std::tuple<int, std::uint64_t>>{
                              {32, bloom_filter_perf::NUM_INPUTS},
                              {256, bloom_filter_perf::NUM_INPUTS},
                              {2048, bloom_filter_perf::NUM_INPUTS}}),
                          int, std::uint64_t);
REGISTER_PERFORMANCE_ARGS(bloomFilterIntersectI64, "BloomFilter Intersect I64 H1V8 (FilterSizeMiB, NumInput)",
                          (std::initializer_list<std::tuple<int, std::uint64_t>>{
                              {32, bloom_filter_perf::NUM_INPUTS},
                              {256, bloom_filter_perf::NUM_INPUTS},
                              {2048, bloom_filter_perf::NUM_INPUTS}}),
                          int, std::uint64_t);

} // namespace aclco::test
