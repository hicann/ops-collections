/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <acl/acl.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"

namespace {

template <typename Key>
std::vector<Key> MakeLargeKeys(std::size_t count)
{
    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i) {
        keys[i] = static_cast<Key>(static_cast<std::uint64_t>(i) * 11400714819323198485ull + 0x9e3779b9u);
    }
    return keys;
}

template <typename Word>
std::vector<Word> ReadWords(Word const* device, std::size_t count, aclrtStream stream)
{
    std::vector<Word> host(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(host.data(), host.size() * sizeof(Word), device, count * sizeof(Word),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter large D2H");
    aclco::test::Sync(stream);
    return host;
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter remains bit-exact for one million keys", "[bloom_filter][correctness][large]",
                   std::uint32_t, std::uint64_t)
{
    using Key = TestType;
    using Filter = aclco::BloomFilter<Key>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t keyCount = 1ull << 20;
    constexpr std::size_t numBlocks = 65537;
    auto const keys = MakeLargeKeys<Key>(keyCount);
    auto const expected = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);

    aclco::test::DeviceBuffer<Key> deviceKeys(keyCount);
    aclco::test::DeviceBuffer<std::uint8_t> output(keyCount);
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    Filter filter(aclco::Extent<std::size_t>{numBlocks}, typename Filter::PolicyType{},
                  typename Filter::AllocatorType{}, stream);
    filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, stream);

    REQUIRE(ReadWords(filter.Data(), static_cast<std::size_t>(filter.NumWords()), stream) == expected);

    filter.Contains(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, output.Data(), stream);
    auto const observed = output.CopyToHost(stream);
    REQUIRE(std::all_of(observed.begin(), observed.end(), [](std::uint8_t value) { return value == 1; }));
}

TEST_CASE("BloomFilter supports the acceptance filter-size matrix", "[bloom_filter][correctness][capacity]")
{
    using Key = std::uint32_t;
    using Filter = aclco::BloomFilter<Key>;
    using Word = Filter::WordType;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t mib = 1024ull * 1024ull;
    constexpr std::size_t reserveBytes = 64ull * mib;
    constexpr std::size_t keyCount = 4096;
    std::array<int, 3> const filterSizesMiB = {32, 256, 2048};
    auto const keys = MakeLargeKeys<Key>(keyCount);
    aclco::test::DeviceBuffer<Key> deviceKeys(keyCount);
    aclco::test::DeviceBuffer<std::uint8_t> output(keyCount);
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    for (int filterSizeMiB : filterSizesMiB) {
        std::size_t freeBytes = 0;
        std::size_t totalBytes = 0;
        aclco::test::CheckAcl(aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes),
                              "aclrtGetMemInfo BloomFilter capacity");
        std::size_t const filterBytes = static_cast<std::size_t>(filterSizeMiB) * mib;
        CAPTURE(filterSizeMiB, filterBytes, freeBytes, totalBytes);
        if (freeBytes < filterBytes || freeBytes - filterBytes < reserveBytes) {
            SKIP("insufficient HBM for BloomFilter capacity case " + std::to_string(filterSizeMiB) + " MiB");
        }

        std::size_t const numBlocks = filterBytes / (Filter::wordsPerBlock * sizeof(Word));
        Filter filter(aclco::Extent<std::size_t>{numBlocks}, Filter::PolicyType{}, Filter::AllocatorType{}, stream);
        REQUIRE(filter.SizeBytes() == filterBytes);

        filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, stream);
        filter.Contains(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, output.Data(), stream);
        auto observed = output.CopyToHost(stream);
        REQUIRE(std::all_of(observed.begin(), observed.end(), [](std::uint8_t value) { return value == 1; }));

        filter.Clear(stream);
        filter.Contains(deviceKeys.Data(), aclco::Extent<std::size_t>{keyCount}, output.Data(), stream);
        observed = output.CopyToHost(stream);
        REQUIRE(std::all_of(observed.begin(), observed.end(), [](std::uint8_t value) { return value == 0; }));
    }
}
