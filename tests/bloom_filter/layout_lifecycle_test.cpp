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

#include <acl/acl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"
#include "tiling/platform/platform_ascendc.h"

namespace bloom_filter_test_detail {

template <typename Key>
using H8V1Policy = aclco::BloomFilterPolicy<Key, aclco::xxhash_64<Key>, std::uint32_t, 8, 8, 8, 1, 8, 1>;

template <typename Key>
class RotatedBlockH8V1Policy : public H8V1Policy<Key> {
public:
    using Base = H8V1Policy<Key>;
    using Hasher = typename Base::Hasher;

    COLLECTION_HOST_DEVICE constexpr RotatedBlockH8V1Policy(Hasher hash = Hasher{}) noexcept : Base(hash) {}

    template <typename SizeType>
    COLLECTION_HOST_DEVICE constexpr std::uint32_t BlockIndex(std::uint32_t upperHash,
                                                              SizeType numBlocks) const noexcept
    {
        std::uint32_t const block = Base::BlockIndex(upperHash, numBlocks);
        std::uint32_t const blockCount = static_cast<std::uint32_t>(numBlocks);
        return block + 1 == blockCount ? 0U : block + 1;
    }
};

} // namespace bloom_filter_test_detail

namespace {

template <typename Key>
using H8V1Policy = bloom_filter_test_detail::H8V1Policy<Key>;

template <typename Key>
using RotatedBlockH8V1Policy = bloom_filter_test_detail::RotatedBlockH8V1Policy<Key>;

template <typename Key>
std::vector<Key> MakeSequence(std::size_t count, std::uint64_t offset = 0)
{
    std::vector<Key> values;
    values.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        values.push_back(static_cast<Key>(offset + i * 17 + 3));
    }
    return values;
}

template <typename Key>
std::vector<Key> MakeDenseBucketKeys(std::size_t count, std::uint32_t bucket)
{
    H8V1Policy<Key> const policy{};
    std::vector<Key> values;
    values.reserve(count);
    for (std::uint64_t candidate = 1; values.size() < count; ++candidate) {
        Key const key = static_cast<Key>(candidate);
        std::uint64_t const hash = policy.HashKey(key);
        if (static_cast<std::uint32_t>(hash >> 32) >> 19 == bucket) {
            values.push_back(key);
        }
    }
    return values;
}

template <typename Key, typename Policy>
std::vector<typename Policy::WordType> MakePolicyGolden(std::size_t numBlocks, std::vector<Key> const& keys,
                                                        Policy const& policy)
{
    using Word = typename Policy::WordType;
    std::vector<Word> words(numBlocks * Policy::wordsPerBlock, Word{0});
    for (Key const& key : keys) {
        std::uint64_t const hash = policy.HashKey(key);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
        std::uint32_t const block = policy.BlockIndex(static_cast<std::uint32_t>(hash >> 32), numBlocks);
        for (std::uint32_t word = 0; word < Policy::wordsPerBlock; ++word) {
            words[static_cast<std::size_t>(block) * Policy::wordsPerBlock + word] |= policy.WordPatternForLane(
                word, lowerHash);
        }
    }
    return words;
}

template <typename Word>
std::vector<Word> CopyFilterWords(Word const* device, std::size_t count, aclrtStream stream)
{
    std::vector<Word> host(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(host.data(), host.size() * sizeof(Word), device, count * sizeof(Word),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter H8V1 words D2H");
    aclco::test::Sync(stream);
    return host;
}

struct AllocationCounts {
    std::size_t allocations{0};
    std::size_t deallocations{0};
    std::vector<std::size_t> allocationElements;
    void* lastAllocation{nullptr};
};

template <typename T>
class CountingAllocator {
public:
    using ValueType = T;
    using value_type = T;

    CountingAllocator() : counts_{std::make_shared<AllocationCounts>()} {}
    explicit CountingAllocator(std::shared_ptr<AllocationCounts> counts) : counts_{std::move(counts)} {}

    T* Allocate(std::size_t count) const
    {
        void* result = nullptr;
        if (aclrtMalloc(&result, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        try {
            counts_->allocationElements.push_back(count);
        } catch (...) {
            (void)aclrtFree(result);
            throw;
        }
        counts_->lastAllocation = result;
        ++counts_->allocations;
        return static_cast<T*>(result);
    }

    void Deallocate(T* pointer)
    {
        if (pointer != nullptr) {
            (void)aclrtFree(pointer);
            ++counts_->deallocations;
        }
    }

    std::shared_ptr<AllocationCounts> Counts() const { return counts_; }

private:
    std::shared_ptr<AllocationCounts> counts_;
};

template <typename T>
class FailingAllocator {
public:
    using ValueType = T;
    using value_type = T;

    T* Allocate(std::size_t) const noexcept { return nullptr; }

    void Deallocate(T*) noexcept {}
};

struct LimitedAllocationCounts {
    std::size_t attempts{0};
    std::size_t deallocations{0};
};

template <typename T>
class FilterOnlyAllocator {
public:
    using ValueType = T;
    using value_type = T;

    FilterOnlyAllocator() : counts_{std::make_shared<LimitedAllocationCounts>()} {}

    explicit FilterOnlyAllocator(std::shared_ptr<LimitedAllocationCounts> counts) : counts_{std::move(counts)} {}

    T* Allocate(std::size_t count) const
    {
        ++counts_->attempts;
        if (counts_->attempts != 1) {
            return nullptr;
        }
        void* result = nullptr;
        if (aclrtMalloc(&result, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        return static_cast<T*>(result);
    }

    void Deallocate(T* pointer)
    {
        if (pointer != nullptr) {
            (void)aclrtFree(pointer);
            ++counts_->deallocations;
        }
    }

private:
    std::shared_ptr<LimitedAllocationCounts> counts_;
};

class FourByteOffsetAllocator {
public:
    using ValueType = std::uint32_t;
    using value_type = ValueType;

    ValueType* Allocate(std::size_t count) const
    {
        void* allocation = nullptr;
        std::size_t const bytes = count * sizeof(ValueType) + sizeof(ValueType);
        if (aclrtMalloc(&allocation, bytes, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        auto* bytesPointer = static_cast<std::uint8_t*>(allocation);
        return reinterpret_cast<ValueType*>(bytesPointer + sizeof(ValueType));
    }

    void Deallocate(ValueType* pointer)
    {
        if (pointer != nullptr) {
            auto* bytesPointer = reinterpret_cast<std::uint8_t*>(pointer);
            (void)aclrtFree(bytesPointer - sizeof(ValueType));
        }
    }
};

class EventGuard {
public:
    EventGuard()
    {
        aclco::test::CheckAcl(aclrtCreateEventWithFlag(&event_, ACL_EVENT_SYNC),
                              "aclrtCreateEventWithFlag BloomFilter test");
    }

    ~EventGuard()
    {
        if (event_ != nullptr) {
            (void)aclrtDestroyEvent(event_);
        }
    }

    EventGuard(EventGuard const&) = delete;
    EventGuard& operator=(EventGuard const&) = delete;

    aclrtEvent event_{nullptr};
};

template <typename Key, typename Filter>
void VerifyGroupBoundaryOperations(Filter& filter, std::size_t numBlocks, std::size_t count, aclrtStream stream)
{
    CAPTURE(count);
    filter.Clear(stream);
    auto keys = MakeSequence<Key>(count);
    auto queries = keys;
    auto misses = MakeSequence<Key>(9, 1000000);
    queries.insert(queries.end(), misses.begin(), misses.end());

    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    aclco::test::DeviceBuffer<Key> deviceQueries(queries.size());
    aclco::test::DeviceBuffer<std::uint8_t> output(queries.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    deviceQueries.CopyFromHostAsync(queries.data(), queries.size(), stream);
    aclco::test::CheckAcl(aclrtMemsetAsync(output.Data(), output.Bytes(), 0xa5, output.Bytes(), stream),
                          "aclrtMemsetAsync BloomFilter H8V1 output sentinel");

    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);
    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    filter.ContainsAsync(deviceQueries.Data(), aclco::Extent<std::size_t>{queries.size()}, output.Data(), stream);

    auto const observed = output.CopyToHost(stream);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        CAPTURE(i);
        REQUIRE(observed[i] <= 1);
        REQUIRE((observed[i] != 0) == aclco::test::BloomGoldenContains(expectedWords, numBlocks, queries[i]));
    }
    REQUIRE(std::all_of(observed.begin(), observed.begin() + count, [](std::uint8_t value) { return value != 0; }));
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            expectedWords);

    // Exercise stream ordering without an intermediate synchronization:
    // AddAsync must complete before ClearAsync, and H8V1 ContainsAsync must
    // observe the cleared words.
    aclco::test::CheckAcl(aclrtMemsetAsync(output.Data(), output.Bytes(), 0xa5, output.Bytes(), stream),
                          "aclrtMemsetAsync BloomFilter H8V1 cleared-output sentinel");
    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    filter.ClearAsync(stream);
    filter.ContainsAsync(deviceQueries.Data(), aclco::Extent<std::size_t>{queries.size()}, output.Data(), stream);
    auto const cleared = output.CopyToHost(stream);
    REQUIRE(std::all_of(cleared.begin(), cleared.end(), [](std::uint8_t value) { return value == 0; }));
    auto const clearedWords = CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()),
                                              stream);
    REQUIRE(std::all_of(clearedWords.begin(), clearedWords.end(), [](std::uint32_t value) { return value == 0; }));
}

template <typename Key>
struct RepeatedHotBlockKeys {
    std::vector<Key> hotKeys;
    std::vector<Key> keys;
};

template <typename Key, typename Policy>
RepeatedHotBlockKeys<Key> MakeRepeatedHotBlockKeys(Policy const& policy, std::size_t numBlocks, std::size_t keyCount,
                                                   std::uint32_t targetBlock)
{
    std::vector<Key> hotKeys;
    for (Key candidate = 0; hotKeys.size() < 4; ++candidate) {
        std::uint64_t const hash = policy.HashKey(candidate);
        if (policy.BlockIndex(static_cast<std::uint32_t>(hash >> 32), numBlocks) == targetBlock) {
            hotKeys.push_back(candidate);
        }
    }
    std::vector<Key> keys;
    keys.reserve(keyCount);
    for (std::size_t i = 0; i < keyCount; ++i) {
        keys.push_back(hotKeys[i % hotKeys.size()]);
    }
    return {std::move(hotKeys), std::move(keys)};
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter H8V1 operations are bit-exact across group boundaries",
                   "[bloom_filter][correctness][layout][tail][stream]", std::uint32_t, std::uint64_t, float)
{
    using Key = TestType;
    using Policy = H8V1Policy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 131;
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, typename Filter::AllocatorType{}, stream);
    std::size_t aivCoreNum = static_cast<std::size_t>(
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    if (aivCoreNum == 0) {
        aivCoreNum = 1;
    }
    // Exercise multiple routed Add rounds plus a partial tail.
    std::size_t const multiRoundCount = aivCoreNum * aclco::BLOOM_FILTER_THREAD_NUM * 4 + 17;
    std::vector<std::size_t> const counts = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1023, 1024, 1025, 16384, multiRoundCount};
    for (std::size_t count : counts) {
        VerifyGroupBoundaryOperations<Key>(filter, numBlocks, count, stream);
    }
}

TEST_CASE("BloomFilter H8V1 routed Add is bit-exact when block slots overflow",
          "[bloom_filter][correctness][layout][routed_add][overflow]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Allocator = CountingAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    static_assert(aclco::detail::UsesMultiplyHighBlockIndex<Policy>::value,
                  "BloomFilterPolicy must enable the multiply-high shortcut");

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 32768;
    constexpr std::size_t keyCount = 16384;
    constexpr std::uint64_t hashSeed = 42;
    Policy policy(typename Policy::Hasher{hashSeed});
    constexpr std::uint32_t targetBlock = 7;
    auto const repeatedKeys = MakeRepeatedHotBlockKeys<Key>(policy, numBlocks, keyCount, targetBlock);
    auto const& hotKeys = repeatedKeys.hotKeys;
    auto const& keys = repeatedKeys.keys;
    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys, hashSeed);
    for (Key hotKey : hotKeys) {
        std::vector<Key> const oneKey{hotKey};
        REQUIRE(expectedWords != aclco::test::MakeBloomGolden<Key>(numBlocks, oneKey, hashSeed));
    }

    auto allocationCounts = std::make_shared<AllocationCounts>();
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, policy, Allocator{allocationCounts}, stream);
    REQUIRE(allocationCounts->allocations == 1);
    REQUIRE(reinterpret_cast<std::uintptr_t>(std::as_const(filter).Data()) %
                (Policy::wordsPerBlock * sizeof(typename Policy::WordType)) ==
            0);
    REQUIRE(std::any_of(expectedWords.begin(), expectedWords.end(),
                        [](std::uint32_t word) { return word != std::numeric_limits<std::uint32_t>::max(); }));
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    for (std::uint32_t round = 0; round < 3; ++round) {
        CAPTURE(round);
        filter.Clear(stream);
        filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
        REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
                expectedWords);
        REQUIRE(allocationCounts->allocations == 2);
    }
}

TEST_CASE("BloomFilter H8V1 routed Add resets counters between batches",
          "[bloom_filter][correctness][layout][routed_add][lifecycle]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 257;
    auto const firstKeys = MakeSequence<Key>(16384);
    auto const secondKeys = MakeSequence<Key>(16384, 1U << 24);
    auto allKeys = firstKeys;
    allKeys.insert(allKeys.end(), secondKeys.begin(), secondKeys.end());

    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, typename Filter::AllocatorType{}, stream);
    REQUIRE(reinterpret_cast<std::uintptr_t>(std::as_const(filter).Data()) %
                (Policy::wordsPerBlock * sizeof(typename Policy::WordType)) ==
            0);
    aclco::test::DeviceBuffer<Key> firstDevice(firstKeys.size());
    aclco::test::DeviceBuffer<Key> secondDevice(secondKeys.size());
    firstDevice.CopyFromHostAsync(firstKeys.data(), firstKeys.size(), stream);
    secondDevice.CopyFromHostAsync(secondKeys.data(), secondKeys.size(), stream);

    filter.AddAsync(firstDevice.Data(), aclco::Extent<std::size_t>{firstKeys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            aclco::test::MakeBloomGolden<Key>(numBlocks, firstKeys));

    filter.AddAsync(secondDevice.Data(), aclco::Extent<std::size_t>{secondKeys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            aclco::test::MakeBloomGolden<Key>(numBlocks, allKeys));

    // Queue both routed phases back-to-back on the same stream. The second
    // Route must observe the first Apply's counter reset without a host-side
    // synchronization between the two AddAsync calls.
    filter.Clear(stream);
    filter.AddAsync(firstDevice.Data(), aclco::Extent<std::size_t>{firstKeys.size()}, stream);
    filter.AddAsync(secondDevice.Data(), aclco::Extent<std::size_t>{secondKeys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            aclco::test::MakeBloomGolden<Key>(numBlocks, allKeys));
}

TEMPLATE_TEST_CASE("BloomFilter H8V1 routed Add covers sparse, dense, and one-block layouts",
                   "[bloom_filter][correctness][layout][routed_add][index]", std::int32_t, std::uint32_t, std::int64_t,
                   std::uint64_t)
{
    using Key = TestType;
    using Policy = H8V1Policy<Key>;
    using Allocator = CountingAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    auto verify = [stream](std::size_t numBlocks, std::size_t keyCount, std::size_t expectedWorkspaceWords = 0,
                           bool forceLastBlock = false, bool verifyQueuedReuse = false) {
        auto keys = MakeSequence<Key>(keyCount);
        if (forceLastBlock) {
            Policy const policy{};
            for (std::uint64_t candidate = 1;; ++candidate) {
                Key const key = static_cast<Key>(candidate);
                std::uint64_t const hash = policy.HashKey(key);
                if (policy.BlockIndex(static_cast<std::uint32_t>(hash >> 32), numBlocks) == numBlocks - 1) {
                    keys.front() = key;
                    break;
                }
            }
        }
        auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);
        auto allocationCounts = std::make_shared<AllocationCounts>();
        Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, Allocator{allocationCounts}, stream);
        REQUIRE(allocationCounts->allocations == 1);
        REQUIRE((allocationCounts->allocationElements == std::vector<std::size_t>{numBlocks * Policy::wordsPerBlock}));
        aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
        deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
        filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
        REQUIRE(allocationCounts->allocations == 2);
        if (expectedWorkspaceWords != 0) {
            REQUIRE((allocationCounts->allocationElements ==
                     std::vector<std::size_t>{numBlocks * Policy::wordsPerBlock, expectedWorkspaceWords}));
        }
        auto requireWorkspaceCountersCleared = [&] {
            if (expectedWorkspaceWords == 0) {
                return;
            }
            REQUIRE(allocationCounts->lastAllocation != nullptr);
            REQUIRE(expectedWorkspaceWords > numBlocks);
            auto const workspace = CopyFilterWords(static_cast<std::uint32_t const*>(allocationCounts->lastAllocation),
                                                   expectedWorkspaceWords, stream);
            bool allCountersCleared = true;
            for (std::size_t block = 0; block < numBlocks; ++block) {
                if (workspace[block] != 0) {
                    allCountersCleared = false;
                    break;
                }
            }
            REQUIRE(allCountersCleared);
        };
        REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
                expectedWords);
        requireWorkspaceCountersCleared();
        if (verifyQueuedReuse) {
            filter.Clear(stream);
            filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
            filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
            REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()),
                                    stream) == expectedWords);
            requireWorkspaceCountersCleared();
        }
    };

    SECTION("95-slot block-major records retain routed hashes exactly") { verify(256, 16384, 24576, false, true); }

    SECTION("20-slot block-major records retain routed hashes exactly") { verify(1664, 16384, 34944, false, true); }

    SECTION("four-slot block-major records retain routed hashes exactly") { verify(8192, 16384, 40960, false, true); }

    SECTION("five-slot records support a non-aligned block count") { verify(12289, 16384, 73734); }

    SECTION("20-slot records address the final block of a partial group") { verify(1665, 16384, 34965, true); }

    SECTION("one block uses the power-of-two shift sentinel") { verify(1, 16384, 96); }

    SECTION("multiple route rounds update a non-saturated filter")
    {
        std::size_t aivCoreNum = static_cast<std::size_t>(
            platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
        if (aivCoreNum == 0) {
            aivCoreNum = 1;
        }
        std::size_t const keyCount = std::max<std::size_t>(16384, aivCoreNum * aclco::BLOOM_FILTER_THREAD_NUM * 4 + 17);
        std::size_t const targetBlocks = (keyCount + 31) / 32;
        std::size_t numBlocks = 1;
        while (numBlocks < targetBlocks) {
            numBlocks <<= 1;
        }
        verify(numBlocks, keyCount);
    }
}

TEMPLATE_TEST_CASE("BloomFilter producer-private Add is bit-exact and reusable",
                   "[bloom_filter][correctness][layout][routed_add][private_route]", std::uint32_t, std::uint64_t)
{
    using Key = TestType;
    using Policy = H8V1Policy<Key>;
    using Allocator = CountingAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 1U << 20;
    constexpr std::size_t keyCount = 1U << 20;
    constexpr std::size_t collisionCount = 512;
    auto keys = MakeSequence<Key>(keyCount);
    auto const collidingKeys = MakeDenseBucketKeys<Key>(collisionCount, 0);
    std::copy(collidingKeys.begin(), collidingKeys.end(), keys.begin());
    auto const secondKeys = MakeSequence<Key>(keyCount, 1U << 28);
    auto const existingKeys = MakeSequence<Key>(17, 1U << 27);

    auto firstGoldenKeys = existingKeys;
    firstGoldenKeys.insert(firstGoldenKeys.end(), keys.begin(), keys.end());
    auto const firstExpected = aclco::test::MakeBloomGolden<Key>(numBlocks, firstGoldenKeys);
    auto queuedKeys = keys;
    queuedKeys.insert(queuedKeys.end(), secondKeys.begin(), secondKeys.end());
    auto const queuedExpected = aclco::test::MakeBloomGolden<Key>(numBlocks, queuedKeys);

    std::size_t producerCores = static_cast<std::size_t>(
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    if (producerCores == 0) {
        producerCores = 1;
    }
    REQUIRE(producerCores <= aclco::detail::bloom_filter::maximumPrivateRouteProducers);
    std::size_t const counterWords = aclco::detail::bloom_filter::privateRoutePackedCounterWords * producerCores;
    std::size_t const workspaceWords = counterWords +
                                       2ULL * aclco::detail::bloom_filter::densePrivateRouteBucketCount *
                                           producerCores *
                                           aclco::detail::bloom_filter::densePrivateRouteRecordsPerSegment;

    auto allocationCounts = std::make_shared<AllocationCounts>();
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, Allocator{allocationCounts}, stream);
    auto* exposed = filter.Data();
    REQUIRE(exposed != nullptr);
    auto const existingWords = aclco::test::MakeBloomGolden<Key>(numBlocks, existingKeys);
    aclco::test::CheckAcl(
        aclrtMemcpyAsync(exposed, filter.SizeBytes(), existingWords.data(),
                         existingWords.size() * sizeof(std::uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream),
        "aclrtMemcpyAsync BloomFilter private existing contents H2D");

    aclco::test::DeviceBuffer<Key> firstDevice(keys.size());
    aclco::test::DeviceBuffer<Key> secondDevice(secondKeys.size());
    firstDevice.CopyFromHostAsync(keys.data(), keys.size(), stream);
    secondDevice.CopyFromHostAsync(secondKeys.data(), secondKeys.size(), stream);
    filter.AddAsync(firstDevice.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    REQUIRE(allocationCounts->allocations == 2);
    REQUIRE((allocationCounts->allocationElements ==
             std::vector<std::size_t>{numBlocks * Policy::wordsPerBlock, workspaceWords}));
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            firstExpected);

    auto const counters = CopyFilterWords(static_cast<std::uint32_t const*>(allocationCounts->lastAllocation),
                                          counterWords, stream);
    std::uint64_t routedKeys = 0;
    for (std::uint32_t count : counters) {
        routedKeys += count;
    }
    REQUIRE(routedKeys == keyCount);

    filter.Clear(stream);
    filter.AddAsync(firstDevice.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    filter.AddAsync(secondDevice.Data(), aclco::Extent<std::size_t>{secondKeys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            queuedExpected);
    REQUIRE(allocationCounts->allocations == 2);
}

TEST_CASE("BloomFilter H8V1 routed Add honors a custom block-index policy",
          "[bloom_filter][correctness][layout][routed_add][policy]")
{
    using Key = std::uint32_t;
    using Policy = RotatedBlockH8V1Policy<Key>;
    using Allocator = CountingAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    static_assert(!aclco::detail::UsesMultiplyHighBlockIndex<Policy>::value,
                  "custom block-index policies must not use the multiply-high shortcut");

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 1024;
    auto const keys = MakeSequence<Key>(16384);
    Policy const policy{};
    auto const expectedWords = MakePolicyGolden(numBlocks, keys, policy);
    auto allocationCounts = std::make_shared<AllocationCounts>();
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, policy, Allocator{allocationCounts}, stream);
    REQUIRE(allocationCounts->allocations == 1);
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    REQUIRE(allocationCounts->allocations == 2);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            expectedWords);
}

TEST_CASE("BloomFilter H8V1 routed Add remains exact after mutable Data exposure",
          "[bloom_filter][correctness][layout][routed_add][data]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 67;
    auto const keys = MakeSequence<Key>(16384);
    auto const existingKeys = MakeSequence<Key>(16, 1U << 25);
    auto allKeys = existingKeys;
    allKeys.insert(allKeys.end(), keys.begin(), keys.end());
    auto const existingWords = aclco::test::MakeBloomGolden<Key>(numBlocks, existingKeys);
    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, allKeys);
    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, typename Filter::AllocatorType{}, stream);

    auto* exposed = filter.Data();
    REQUIRE(exposed != nullptr);
    aclco::test::CheckAcl(
        aclrtMemcpyAsync(exposed, filter.SizeBytes(), existingWords.data(),
                         existingWords.size() * sizeof(std::uint32_t), ACL_MEMCPY_HOST_TO_DEVICE, stream),
        "aclrtMemcpyAsync BloomFilter exposed contents H2D");

    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            expectedWords);
}

TEST_CASE("BloomFilter H8V1 Add falls back for four-byte-aligned storage",
          "[bloom_filter][correctness][layout][packed_atomic][allocator]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, FourByteOffsetAllocator>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 31;
    auto const keys = MakeSequence<Key>(257);
    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);

    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, FourByteOffsetAllocator{}, stream);
    REQUIRE(reinterpret_cast<std::uintptr_t>(std::as_const(filter).Data()) % alignof(std::uint64_t) != 0);

    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            expectedWords);
}

TEST_CASE("BloomFilter H8V1 Contains routes every lane result mask exactly",
          "[bloom_filter][correctness][layout][lane_mask]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 131;
    constexpr std::uint32_t tileWidth = 8;
    constexpr std::uint32_t maskCount = 1U << tileWidth;
    auto const hits = MakeSequence<Key>(tileWidth);
    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, hits);

    std::vector<Key> misses;
    misses.reserve(tileWidth);
    for (Key candidate = 1000003; misses.size() < tileWidth; candidate += 17) {
        if (!aclco::test::BloomGoldenContains(expectedWords, numBlocks, candidate)) {
            misses.push_back(candidate);
        }
    }

    std::vector<Key> queries;
    queries.reserve(static_cast<std::size_t>(maskCount) * tileWidth);
    for (std::uint32_t mask = 0; mask < maskCount; ++mask) {
        for (std::uint32_t lane = 0; lane < tileWidth; ++lane) {
            queries.push_back(((mask >> lane) & 1U) != 0 ? hits[lane] : misses[lane]);
        }
    }

    Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, typename Filter::AllocatorType{}, stream);
    aclco::test::DeviceBuffer<Key> deviceHits(hits.size());
    aclco::test::DeviceBuffer<Key> deviceQueries(queries.size());
    aclco::test::DeviceBuffer<std::uint8_t> output(queries.size());
    deviceHits.CopyFromHostAsync(hits.data(), hits.size(), stream);
    deviceQueries.CopyFromHostAsync(queries.data(), queries.size(), stream);
    aclco::test::CheckAcl(aclrtMemsetAsync(output.Data(), output.Bytes(), 0xa5, output.Bytes(), stream),
                          "aclrtMemsetAsync BloomFilter H8V1 lane-mask sentinel");
    filter.AddAsync(deviceHits.Data(), aclco::Extent<std::size_t>{hits.size()}, stream);
    filter.ContainsAsync(deviceQueries.Data(), aclco::Extent<std::size_t>{queries.size()}, output.Data(), stream);

    auto const observed = output.CopyToHost(stream);
    for (std::uint32_t mask = 0; mask < maskCount; ++mask) {
        for (std::uint32_t lane = 0; lane < tileWidth; ++lane) {
            CAPTURE(mask, lane);
            std::size_t const index = static_cast<std::size_t>(mask) * tileWidth + lane;
            REQUIRE(observed[index] == static_cast<std::uint8_t>((mask >> lane) & 1U));
        }
    }
}

TEST_CASE("BloomFilter move operations transfer ownership exactly once", "[bloom_filter][lifecycle]")
{
    using Key = std::uint32_t;
    using Policy = aclco::BloomFilterPolicy<Key>;
    using Allocator = CountingAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    auto counts = std::make_shared<AllocationCounts>();
    Allocator allocator(counts);

    {
        Filter first(aclco::Extent<std::size_t>{17}, Policy{}, allocator, stream);
        auto* firstData = first.Data();
        Filter second(std::move(first));
        REQUIRE(first.Data() == nullptr);
        REQUIRE(second.Data() == firstData);
        REQUIRE_THROWS_AS(first.AddAsync(firstData, aclco::Extent<std::size_t>{1}, stream), std::logic_error);
        REQUIRE_THROWS_AS(first.ContainsAsync(firstData, aclco::Extent<std::size_t>{1}, firstData, stream),
                          std::logic_error);

        Filter third(aclco::Extent<std::size_t>{3}, Policy{}, allocator, stream);
        third = std::move(second);
        REQUIRE(second.Data() == nullptr);
        REQUIRE(third.Data() == firstData);
        REQUIRE_THROWS_AS(second.MergeAsync(third, stream), std::logic_error);
        REQUIRE_THROWS_AS(second.IntersectAsync(third, stream), std::logic_error);
        REQUIRE(counts->allocations == 2);
        REQUIRE(counts->deallocations == 1);
    }

    REQUIRE(counts->allocations == 2);
    REQUIRE(counts->deallocations == 2);
}

TEST_CASE("BloomFilter routed Add workspace is lazy and moves with the filter",
          "[bloom_filter][lifecycle][allocator][routed_add]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Allocator = CountingAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    auto counts = std::make_shared<AllocationCounts>();
    Allocator allocator(counts);
    constexpr std::size_t numBlocks = 1024;
    auto const keys = MakeSequence<Key>(16384);
    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    {
        Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, allocator, stream);
        REQUIRE(counts->allocations == 1);

        filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size() - 1}, stream);
        REQUIRE(counts->allocations == 1);
        filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
        REQUIRE(counts->allocations == 2);
        filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
        REQUIRE(counts->allocations == 2);

        Filter moved(std::move(filter));
        REQUIRE(counts->allocations == 2);
        Filter assigned(aclco::Extent<std::size_t>{3}, Policy{}, allocator, stream);
        REQUIRE(counts->allocations == 3);
        assigned = std::move(moved);
        REQUIRE(counts->deallocations == 1);
        REQUIRE(CopyFilterWords(std::as_const(assigned).Data(), static_cast<std::size_t>(assigned.NumWords()),
                                stream) == expectedWords);
    }

    REQUIRE(counts->deallocations == 3);
}

TEST_CASE("BloomFilter routed Add falls back when workspace allocation fails",
          "[bloom_filter][correctness][allocator][routed_add][fallback]")
{
    using Key = std::uint32_t;
    using Policy = H8V1Policy<Key>;
    using Allocator = FilterOnlyAllocator<typename Policy::WordType>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    auto counts = std::make_shared<LimitedAllocationCounts>();
    constexpr std::size_t numBlocks = 1024;
    auto const keys = MakeSequence<Key>(16384);
    auto const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    {
        Filter filter(aclco::Extent<std::size_t>{numBlocks}, Policy{}, Allocator{counts}, stream);
        filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
        REQUIRE(counts->attempts == 2);
        filter.Add(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
        REQUIRE(counts->attempts == 2);
        REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
                expectedWords);
    }

    REQUIRE(counts->deallocations == 1);
}

TEST_CASE("BloomFilter reports allocator failure", "[bloom_filter][validation]")
{
    using Key = std::uint32_t;
    using Policy = aclco::BloomFilterPolicy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy,
                                      FailingAllocator<typename Policy::WordType>>;

    REQUIRE_THROWS_AS((Filter{aclco::Extent<std::size_t>{1}, Policy{}, Filter::AllocatorType{}}), std::bad_alloc);
}

TEST_CASE("BloomFilter honors an explicit cross-stream event dependency", "[bloom_filter][stream]")
{
    using Key = std::uint32_t;
    using Filter = aclco::BloomFilter<Key>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard buildStreamGuard;
    aclco::test::AclStreamGuard queryStreamGuard;
    EventGuard ready;
    aclrtStream buildStream = buildStreamGuard.stream;
    aclrtStream queryStream = queryStreamGuard.stream;

    auto keys = MakeSequence<Key>(257);
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    aclco::test::DeviceBuffer<std::uint8_t> output(keys.size());
    Filter filter(aclco::Extent<std::size_t>{67}, Filter::PolicyType{}, Filter::AllocatorType{}, buildStream);

    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), buildStream);
    filter.AddAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, buildStream);
    aclco::test::CheckAcl(aclrtRecordEvent(ready.event_, buildStream), "aclrtRecordEvent BloomFilter build ready");
    aclco::test::CheckAcl(aclrtStreamWaitEvent(queryStream, ready.event_), "aclrtStreamWaitEvent BloomFilter query");
    filter.ContainsAsync(deviceKeys.Data(), aclco::Extent<std::size_t>{keys.size()}, output.Data(), queryStream);

    auto const observed = output.CopyToHost(queryStream);
    REQUIRE(std::all_of(observed.begin(), observed.end(), [](std::uint8_t value) { return value == 1; }));

    aclco::test::CheckAcl(aclrtResetEvent(ready.event_, queryStream), "aclrtResetEvent BloomFilter test");
    aclco::test::Sync(queryStream);
}
