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
using Context = bloom_filter_perf::SingleFilterContext<Filter<Key>>;

template <typename Key>
void SetupClear(int filterSizeMiB)
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    bloom_filter_perf::Release(context);
    if constexpr (std::is_same_v<Key, std::uint64_t>) {
        auto& previous = bloom_filter_perf::LeakedContext<Context<std::uint32_t>>();
        bloom_filter_perf::Release(previous);
    }
    context.numBlocks = bloom_filter_perf::BlocksForMiB(filterSizeMiB);
    bloom_filter_perf::RequireDeviceMemory(bloom_filter_perf::FilterBytesForMiB(filterSizeMiB));
    bloom_filter_perf::EmplaceFilter(context.filter, context.numBlocks, context.stream);
    bloom_filter_perf::FillFilter(*context.filter, 0xffu, context.stream);
    context.filter->ClearAsync(context.stream);
    Sync(context.stream);
}

template <typename Key>
TestResult TestClear()
{
    auto& context = bloom_filter_perf::LeakedContext<Context<Key>>();
    return bloom_filter_perf::MeasureAsync(context.timer, context.stream,
                                           [&context] { context.filter->ClearAsync(context.stream); });
}

} // namespace

REGISTER_PERFORMANCE_TEST(bloomFilterClearI32, (TestClear<std::uint32_t>), (SetupClear<std::uint32_t>), int);
REGISTER_PERFORMANCE_TEST(bloomFilterClearI64, (TestClear<std::uint64_t>), (SetupClear<std::uint64_t>), int);

REGISTER_PERFORMANCE_ARGS(bloomFilterClearI32, "BloomFilter Clear I32 H8V1 (FilterSizeMiB)",
                          (std::initializer_list<std::tuple<int>>{{32}, {256}, {2048}}), int);
REGISTER_PERFORMANCE_ARGS(bloomFilterClearI64, "BloomFilter Clear I64 H8V1 (FilterSizeMiB)",
                          (std::initializer_list<std::tuple<int>>{{32}, {256}, {2048}}), int);

} // namespace aclco::test
