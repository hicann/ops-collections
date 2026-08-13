/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"

namespace {

constexpr std::uint32_t wordsPerBlock = 8;
constexpr std::uint32_t patternBits = 16;
constexpr std::uint64_t seededPolicySeed = 42;

using CompatibilityDefaultPolicy = aclco::DefaultFilterPolicy<aclco::xxhash_64<std::uint32_t>, std::uint32_t,
                                                              wordsPerBlock>;
using ExpectedDefaultPolicy = aclco::BloomFilterPolicy<std::uint32_t, aclco::xxhash_64<std::uint32_t>, std::uint32_t,
                                                       wordsPerBlock, wordsPerBlock, wordsPerBlock, 1, 1, wordsPerBlock,
                                                       false, false>;
using CompatibilityDefaultPolicy16 = aclco::DefaultFilterPolicy<aclco::xxhash_64<std::uint64_t>, std::uint32_t, 16>;
using ExpectedDefaultPolicy16 = aclco::BloomFilterPolicy<std::uint64_t, aclco::xxhash_64<std::uint64_t>, std::uint32_t,
                                                         16>;
using CompatibilityArrowPolicy = aclco::ArrowFilterPolicy<std::uint32_t>;
using ExpectedArrowPolicy = aclco::BloomFilterPolicy<std::uint32_t, aclco::xxhash_64<std::uint32_t>, std::uint32_t, 8,
                                                     8, 8, 1, 1, 8, false, false>;

static_assert(std::is_same_v<CompatibilityDefaultPolicy, ExpectedDefaultPolicy>);
static_assert(std::is_same_v<CompatibilityDefaultPolicy16, ExpectedDefaultPolicy16>);
static_assert(std::is_same_v<CompatibilityArrowPolicy, ExpectedArrowPolicy>);
static_assert(aclco::detail::UsesMultiplyHighBlockIndex<CompatibilityDefaultPolicy>::value);
static_assert(aclco::detail::UsesMultiplyHighBlockIndex<CompatibilityArrowPolicy>::value);

template <typename Key>
using ScalarPattern16Policy = aclco::BloomFilterPolicy<Key, aclco::xxhash_64<Key>, std::uint32_t, wordsPerBlock,
                                                       patternBits, 1, wordsPerBlock, 1, wordsPerBlock, true, true>;

template <typename Key>
std::vector<Key> MakeKeys(std::size_t count, std::uint64_t offset = 0)
{
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        keys.push_back(static_cast<Key>(offset + i * 17 + 3));
    }
    return keys;
}

template <typename Word>
std::vector<Word> CopyFilterWords(Word const* device, std::size_t count, aclrtStream stream)
{
    std::vector<Word> host(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(host.data(), host.size() * sizeof(Word), device, count * sizeof(Word),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter generalization D2H");
    aclco::test::Sync(stream);
    return host;
}

template <typename Key>
std::vector<Key> FindKeysForBlock(std::size_t numBlocks, std::uint32_t targetBlock, std::size_t count,
                                  std::uint64_t seed = 0)
{
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::uint64_t candidate = 0; keys.size() < count; ++candidate) {
        Key const key = static_cast<Key>(candidate);
        std::uint64_t const hash = aclco::test::bloom_golden_detail::XXHash64(key, seed);
        std::uint32_t const block = aclco::test::bloom_golden_detail::BlockIndex(static_cast<std::uint32_t>(hash >> 32),
                                                                                 numBlocks);
        if (block == targetBlock) {
            keys.push_back(key);
        }
    }
    return keys;
}

template <typename Filter, typename Key>
void AddAndRequireWords(Filter& filter, std::vector<Key> const& keys, std::vector<std::uint32_t> const& expected,
                        aclrtStream stream)
{
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), filter.NumWords(), stream) == expected);
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter seeded PatternBits=16 conditional scalar fallback is bit-exact",
                   "[bloom_filter][generalization][policy]", std::uint32_t, std::uint64_t, float)
{
    using Key = TestType;
    using Policy = ScalarPattern16Policy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy>;

    static_assert(Policy::patternBits == patternBits);
    static_assert(Policy::addHorizontalLayout == 1);
    static_assert(Policy::containsHorizontalLayout == 1);
    static_assert(Policy::conditionalAdd);
    static_assert(Policy::earlyExitContains);

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 257;
    Policy policy(typename Policy::Hasher{seededPolicySeed});
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, policy, typename Filter::AllocatorType{}, stream);
    auto keys = MakeKeys<Key>(2051);
    auto const expectedWords = aclco::test::MakeBloomGolden<Key, wordsPerBlock, patternBits>(numBlocks, keys,
                                                                                             seededPolicySeed);

    AddAndRequireWords(filter, keys, expectedWords, stream);

    auto queries = keys;
    auto misses = MakeKeys<Key>(257, 1000000);
    queries.insert(queries.end(), misses.begin(), misses.end());
    aclco::test::DeviceBuffer<Key> deviceQueries(queries.size());
    aclco::test::DeviceBuffer<std::uint8_t> output(queries.size());
    deviceQueries.CopyFromHostAsync(queries.data(), queries.size(), stream);
    filter.Contains(deviceQueries.Data(), aclco::Extent<std::size_t>{queries.size()}, output.Data(), stream);

    auto const observed = output.CopyToHost(stream);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        CAPTURE(i);
        REQUIRE(((observed[i] != 0) == aclco::test::BloomGoldenContains<Key, wordsPerBlock, patternBits>(
                                           expectedWords, numBlocks, queries[i], seededPolicySeed)));
    }
}

TEST_CASE("BloomFilter is bit-exact for repeated and same-block hotspot keys",
          "[bloom_filter][generalization][distribution]")
{
    using Key = std::uint32_t;
    using Filter = aclco::BloomFilter<Key>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    constexpr std::size_t numBlocks = 257;

    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Filter::PolicyType{}, Filter::AllocatorType{}, stream);

    std::vector<Key> repeated(2048, Key{1234567});
    auto repeatedGolden = aclco::test::MakeBloomGolden<Key>(numBlocks, repeated);
    AddAndRequireWords(filter, repeated, repeatedGolden, stream);

    filter.Clear(stream);
    auto hotspot = FindKeysForBlock<Key>(numBlocks, 73, 512);
    REQUIRE(std::all_of(hotspot.begin(), hotspot.end(), [=](Key key) {
        std::uint64_t const hash = aclco::test::bloom_golden_detail::XXHash64(key);
        return aclco::test::bloom_golden_detail::BlockIndex(static_cast<std::uint32_t>(hash >> 32), numBlocks) == 73;
    }));
    auto hotspotGolden = aclco::test::MakeBloomGolden<Key>(numBlocks, hotspot);
    AddAndRequireWords(filter, hotspot, hotspotGolden, stream);
}

TEST_CASE("BloomFilter query workloads cover explicit membership ratios", "[bloom_filter][generalization][hit_rate]")
{
    using Key = std::uint32_t;
    using Filter = aclco::BloomFilter<Key>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    constexpr std::size_t numBlocks = 1021;

    auto inserted = MakeKeys<Key>(1000);
    auto misses = MakeKeys<Key>(1000, 1000000);
    auto expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, inserted);
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Filter::PolicyType{}, Filter::AllocatorType{}, stream);
    AddAndRequireWords(filter, inserted, expectedWords, stream);

    struct RatioCase {
        std::uint32_t percent;
        std::size_t hits;
    };
    constexpr std::array<RatioCase, 5> cases = {RatioCase{0, 0}, RatioCase{1, 1}, RatioCase{50, 50}, RatioCase{99, 99},
                                                RatioCase{100, 100}};

    for (auto const& ratio : cases) {
        CAPTURE(ratio.percent);
        std::vector<Key> queries;
        queries.reserve(100);
        queries.insert(queries.end(), inserted.begin(), inserted.begin() + ratio.hits);
        queries.insert(queries.end(), misses.begin(), misses.begin() + (100 - ratio.hits));

        aclco::test::DeviceBuffer<Key> deviceQueries(queries.size());
        aclco::test::DeviceBuffer<std::uint8_t> output(queries.size());
        deviceQueries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        filter.Contains(deviceQueries.Data(), aclco::Extent<std::size_t>{queries.size()}, output.Data(), stream);

        auto const observed = output.CopyToHost(stream);
        for (std::size_t i = 0; i < queries.size(); ++i) {
            CAPTURE(i);
            REQUIRE((observed[i] != 0) == aclco::test::BloomGoldenContains(expectedWords, numBlocks, queries[i]));
        }
    }
}
