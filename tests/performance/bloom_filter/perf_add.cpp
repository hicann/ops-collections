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

using AddAllocator = bloom_filter_perf::ObservableDeviceAllocator<std::uint32_t>;

template <typename Key>
using Context = bloom_filter_perf::InputContext<Key, bloom_filter_perf::H8V1Policy<Key>, AddAllocator>;

template <typename Key>
void SetupAdd(int filterSizeMiB, std::uint64_t numInputs)
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    bloom_filter_perf::Release(context);
    if constexpr (std::is_same_v<Key, std::uint64_t>) {
        auto& previous = bloom_filter_perf::LeakedContext<Context<std::uint32_t>>();
        bloom_filter_perf::Release(previous);
    }
    AddAllocator::ResetStatistics();
    context.numBlocks = bloom_filter_perf::BlocksForMiB(filterSizeMiB);
    context.numInputs = bloom_filter_perf::CheckedInputCount(numInputs);
    std::size_t const workspaceBytes = bloom_filter_perf::RoutedAddWorkspaceBytes(
        context.numBlocks, context.numInputs, bloom_filter_perf::RoutedAddProducerCores(context.numInputs));
    bloom_filter_perf::RequireDeviceMemory(bloom_filter_perf::FilterBytesForMiB(filterSizeMiB) +
                                           bloom_filter_perf::BytesForElements<Key>(context.numInputs) +
                                           workspaceBytes);
    bloom_filter_perf::EmplaceFilter(context.filter, context.numBlocks, context.stream);
    bloom_filter_perf::PrepareSequentialKeys(context.keys, context.numInputs, context.stream);

    // One untimed warm-up, followed by a synchronized reset for the first sample.
    context.filter->AddAsync(context.keys.Data(), aclco::Extent<std::size_t>{context.numInputs}, context.stream);
    Sync(context.stream);
    std::size_t const workspaceElements = workspaceBytes / sizeof(std::uint32_t);
    if (workspaceBytes == 0 || AddAllocator::AllocationCount() != 2 ||
        AddAllocator::LastAllocationElements() != workspaceElements) {
        throw std::runtime_error("BloomFilter Add performance warm-up did not allocate the expected "
                                 "routed workspace: allocations=" +
                                 std::to_string(AddAllocator::AllocationCount()) +
                                 ", expectedElements=" + std::to_string(workspaceElements) +
                                 ", actualElements=" + std::to_string(AddAllocator::LastAllocationElements()));
    }
    // The three performance sizes use compile-time block-index shifts. Validate
    // the untimed warm-up against Contains so timing cannot silently proceed
    // with a fast but incorrectly indexed route kernel.
    std::size_t const validationCount = std::min<std::size_t>(context.numInputs, 4096);
    context.output.Resize(validationCount);
    if (validationCount != 0) {
        context.filter->ContainsAsync(context.keys.Data(), aclco::Extent<std::size_t>{validationCount},
                                      context.output.Data(), context.stream);
        auto const validation = context.output.CopyToHost(context.stream);
        if (!std::all_of(validation.begin(), validation.end(), [](std::uint8_t value) { return value == 1; })) {
            throw std::runtime_error("BloomFilter Add performance warm-up failed routed correctness validation");
        }
    }
    context.filter->Clear(context.stream);
}

template <typename Key>
TestResult TestAdd()
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    context.filter->Clear(context.stream);
    return bloom_filter_perf::MeasureAsync(context.timer, context.stream, [&context] {
        context.filter->AddAsync(context.keys.Data(), aclco::Extent<std::size_t>{context.numInputs}, context.stream);
    });
}

} // namespace

REGISTER_PERFORMANCE_TEST(bloomFilterAddI32, (TestAdd<std::uint32_t>), (SetupAdd<std::uint32_t>), int, std::uint64_t);
REGISTER_PERFORMANCE_TEST(bloomFilterAddI64, (TestAdd<std::uint64_t>), (SetupAdd<std::uint64_t>), int, std::uint64_t);

REGISTER_PERFORMANCE_ARGS(bloomFilterAddI32, "BloomFilter Add I32 H8V1 (FilterSizeMiB, NumInput)",
                          (std::initializer_list<std::tuple<int, std::uint64_t>>{
                              {32, bloom_filter_perf::NUM_INPUTS},
                              {256, bloom_filter_perf::NUM_INPUTS},
                              {2048, bloom_filter_perf::NUM_INPUTS}}),
                          int, std::uint64_t);
REGISTER_PERFORMANCE_ARGS(bloomFilterAddI64, "BloomFilter Add I64 H8V1 (FilterSizeMiB, NumInput)",
                          (std::initializer_list<std::tuple<int, std::uint64_t>>{
                              {32, bloom_filter_perf::NUM_INPUTS},
                              {256, bloom_filter_perf::NUM_INPUTS},
                              {2048, bloom_filter_perf::NUM_INPUTS}}),
                          int, std::uint64_t);

} // namespace aclco::test
