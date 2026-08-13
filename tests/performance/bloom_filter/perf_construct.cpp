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
void SetupConstruct(int filterSizeMiB)
{
    auto& context = bloom_filter_perf::LeakedContext<bloom_filter_perf::HostContext>();
    context.numBlocks = bloom_filter_perf::BlocksForMiB(filterSizeMiB);
    bloom_filter_perf::RequireDeviceMemory(bloom_filter_perf::FilterBytesForMiB(filterSizeMiB));
}

template <typename Key>
TestResult TestConstruct()
{
    using Filter = bloom_filter_perf::Filter<Key>;
    auto& context = bloom_filter_perf::LeakedContext<bloom_filter_perf::HostContext>();

    std::optional<Filter> filter;
    auto const start = std::chrono::steady_clock::now();
    filter.emplace(aclco::Extent<std::size_t>{context.numBlocks}, typename Filter::PolicyType{},
                   typename Filter::AllocatorType{}, context.stream);
    auto const stop = std::chrono::steady_clock::now();
    TestResult const result = bloom_filter_perf::HostResult(start, stop);
    filter.reset();
    return result;
}

} // namespace

REGISTER_PERFORMANCE_TEST(bloomFilterConstructI32, (TestConstruct<std::uint32_t>), (SetupConstruct<std::uint32_t>),
                          int);
REGISTER_PERFORMANCE_TEST(bloomFilterConstructI64, (TestConstruct<std::uint64_t>), (SetupConstruct<std::uint64_t>),
                          int);

REGISTER_PERFORMANCE_ARGS(bloomFilterConstructI32, "BloomFilter Construct I32 (FilterSizeMiB)",
                          (std::initializer_list<std::tuple<int>>{{32}, {256}, {2048}}), int);
REGISTER_PERFORMANCE_ARGS(bloomFilterConstructI64, "BloomFilter Construct I64 (FilterSizeMiB)",
                          (std::initializer_list<std::tuple<int>>{{32}, {256}, {2048}}), int);

} // namespace aclco::test
