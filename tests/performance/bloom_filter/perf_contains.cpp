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
using Context = bloom_filter_perf::InputContext<Key, bloom_filter_perf::H8V1Policy<Key>>;

template <typename Key>
void SetupContains(int filterSizeMiB, std::uint64_t numInputs)
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    bloom_filter_perf::Release(context);
    if constexpr (std::is_same_v<Key, std::uint64_t>) {
        auto& previous = bloom_filter_perf::LeakedContext<Context<std::uint32_t>>();
        bloom_filter_perf::Release(previous);
    }
    context.numBlocks = bloom_filter_perf::BlocksForMiB(filterSizeMiB);
    context.numInputs = bloom_filter_perf::CheckedInputCount(numInputs);
    std::size_t const numBuildKeys = bloom_filter_perf::ReferenceBuildKeyCount<bloom_filter_perf::H8V1Policy<Key>>(
        bloom_filter_perf::FilterBytesForMiB(filterSizeMiB));
    std::size_t const firstBuildBatch = std::min(numBuildKeys, bloom_filter_perf::KEY_STAGING_ELEMENTS);
    bloom_filter_perf::RequireDeviceMemory(
        bloom_filter_perf::FilterBytesForMiB(filterSizeMiB) +
        bloom_filter_perf::BytesForElements<Key>(context.numInputs) +
        bloom_filter_perf::BytesForElements<std::uint8_t>(context.numInputs) +
        bloom_filter_perf::RoutedAddWorkspaceBytes(context.numBlocks, firstBuildBatch,
                                                   bloom_filter_perf::RoutedAddProducerCores(firstBuildBatch)));
    bloom_filter_perf::EmplaceFilter(context.filter, context.numBlocks, context.stream);
    bloom_filter_perf::BuildSequential(*context.filter, context.keys, numBuildKeys, context.stream);
    bloom_filter_perf::PrepareSequentialKeys(context.keys, context.numInputs, context.stream);
    context.output.Resize(context.numInputs);
    context.output.MemsetZero(context.stream);

    // Match the reference workload: build filterBits/(2*PatternBits) sequential
    // keys, then query the fixed [0, 80M) sequence.
    context.filter->ContainsAsync(context.keys.Data(), aclco::Extent<std::size_t>{context.numInputs},
                                  context.output.Data(), context.stream);
    Sync(context.stream);
}

template <typename Key>
TestResult TestContains()
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    return bloom_filter_perf::MeasureAsync(context.timer, context.stream, [&context] {
        context.filter->ContainsAsync(context.keys.Data(), aclco::Extent<std::size_t>{context.numInputs},
                                      context.output.Data(), context.stream);
    });
}

} // namespace

REGISTER_PERFORMANCE_TEST(bloomFilterContainsI32H8V1, (TestContains<std::uint32_t>), (SetupContains<std::uint32_t>),
                          int, std::uint64_t);
REGISTER_PERFORMANCE_TEST(bloomFilterContainsI64H8V1, (TestContains<std::uint64_t>), (SetupContains<std::uint64_t>),
                          int, std::uint64_t);

REGISTER_PERFORMANCE_ARGS(bloomFilterContainsI32H8V1, "BloomFilter Contains I32 H8V1 (FilterSizeMiB, NumInput)",
                          (std::initializer_list<std::tuple<int, std::uint64_t>>{
                              {32, bloom_filter_perf::NUM_INPUTS},
                              {256, bloom_filter_perf::NUM_INPUTS},
                              {2048, bloom_filter_perf::NUM_INPUTS}}),
                          int, std::uint64_t);
REGISTER_PERFORMANCE_ARGS(bloomFilterContainsI64H8V1, "BloomFilter Contains I64 H8V1 (FilterSizeMiB, NumInput)",
                          (std::initializer_list<std::tuple<int, std::uint64_t>>{
                              {32, bloom_filter_perf::NUM_INPUTS},
                              {256, bloom_filter_perf::NUM_INPUTS},
                              {2048, bloom_filter_perf::NUM_INPUTS}}),
                          int, std::uint64_t);

} // namespace aclco::test
