/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "bloom_filter.h"
#include "../performance_test_framework.h"

namespace aclco::test::bloom_filter_perf {

inline constexpr std::size_t MIB_BYTES = 1024ull * 1024ull;
inline constexpr std::uint64_t NUM_INPUTS = 80000000ull;
inline constexpr std::size_t WORDS_PER_BLOCK = 8;
inline constexpr std::size_t BYTES_PER_BLOCK = WORDS_PER_BLOCK * sizeof(std::uint32_t);
inline constexpr std::size_t DEVICE_MEMORY_RESERVE = 256ull * MIB_BYTES;
inline constexpr std::size_t KEY_STAGING_ELEMENTS = 8ull * 1024ull * 1024ull;

inline std::size_t FilterBytesForMiB(int filterSizeMiB)
{
    if (filterSizeMiB <= 0) {
        throw std::invalid_argument("BloomFilter performance size must be positive");
    }
    if (static_cast<std::size_t>(filterSizeMiB) > std::numeric_limits<std::size_t>::max() / MIB_BYTES) {
        throw std::length_error("BloomFilter performance size overflows size_t");
    }
    return static_cast<std::size_t>(filterSizeMiB) * MIB_BYTES;
}

inline std::size_t BlocksForMiB(int filterSizeMiB)
{
    std::size_t const bytes = FilterBytesForMiB(filterSizeMiB);
    if (bytes % BYTES_PER_BLOCK != 0) {
        throw std::invalid_argument("BloomFilter performance size must contain an integral number of blocks");
    }
    return bytes / BYTES_PER_BLOCK;
}

inline std::size_t CheckedInputCount(std::uint64_t count)
{
    if (count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::length_error("BloomFilter performance input count exceeds size_t");
    }
    return static_cast<std::size_t>(count);
}

inline std::uint32_t RoutedAddProducerCores(std::size_t numInputs)
{
    std::uint32_t available = static_cast<std::uint32_t>(
        platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv());
    if (available == 0) {
        available = 1;
    }
    std::size_t const needed = (numInputs + BLOOM_FILTER_THREAD_NUM - 1) / BLOOM_FILTER_THREAD_NUM;
    return static_cast<std::uint32_t>(std::max<std::size_t>(1, std::min<std::size_t>(available, needed)));
}

inline constexpr std::size_t RoutedAddWorkspaceBytes(std::size_t numBlocks, std::size_t numInputs,
                                                     std::size_t producerCores = 56)
{
    // Keep this estimate aligned with BloomFilter::EnsureAddWorkspace so a
    // performance case either gets the intended route capacity or reports a
    // clear memory error instead of silently benchmarking the atomic fallback.
    constexpr std::size_t minimumAggregatedKeys = 16384;
    constexpr std::size_t maximumCounterWords = 1ULL << 26;
    constexpr std::size_t maximumSlotWords = 5ULL << 26;
    constexpr std::size_t maximumSlotsPerBlock = 95;
    if (numBlocks == 0 || numBlocks > maximumCounterWords || numInputs > std::numeric_limits<std::uint32_t>::max() ||
        numInputs < std::max(minimumAggregatedKeys, (numBlocks + 1) / 2)) {
        return 0;
    }
    constexpr std::size_t minimumPrivateKeys = 1ULL << 20;
    constexpr std::size_t maximumPrivateProducers = 128;
    constexpr std::size_t privateCounterWordsPerProducer = 8192;
    if (numInputs >= minimumPrivateKeys && producerCores != 0 && producerCores <= maximumPrivateProducers &&
        (numBlocks == (1ULL << 20) || numBlocks == (1ULL << 23) || numBlocks == (1ULL << 26))) {
        bool const denseLayout = numBlocks == (1ULL << 20);
        std::size_t const bucketCount = denseLayout ? (1ULL << 13) : (1ULL << 15);
        std::size_t const recordsPerSegment = denseLayout ? 240 : 80;
        std::size_t const words = privateCounterWordsPerProducer * producerCores +
                                  2 * bucketCount * producerCores * recordsPerSegment;
        if (words > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
            throw std::length_error("BloomFilter private Add workspace size overflows size_t");
        }
        return words * sizeof(std::uint32_t);
    }
    std::size_t desiredSlots = std::max<std::size_t>(1, (numInputs * 2 + numBlocks - 1) / numBlocks);
    if (numInputs >= numBlocks && numInputs < numBlocks * 2) {
        desiredSlots += 2;
    }
    std::size_t const slotsPerBlock = std::min(maximumSlotsPerBlock,
                                               std::min(desiredSlots, maximumSlotWords / numBlocks));
    std::size_t const recordWords = slotsPerBlock + 1;
    if (slotsPerBlock == 0 ||
        numBlocks > std::numeric_limits<std::size_t>::max() / recordWords / sizeof(std::uint32_t)) {
        throw std::length_error("BloomFilter routed Add workspace size overflows size_t");
    }
    return numBlocks * recordWords * sizeof(std::uint32_t);
}

static_assert(RoutedAddWorkspaceBytes(1048576, NUM_INPUTS) == 220659712ULL * 4,
              "32 MiB Add must reserve producer-private dense route segments");
static_assert(RoutedAddWorkspaceBytes(8388608, NUM_INPUTS) == 294060032ULL * 4,
              "256 MiB Add must reserve producer-private fine route segments");
static_assert(RoutedAddWorkspaceBytes(67108864, NUM_INPUTS) == 294060032ULL * 4,
              "2048 MiB Add must reserve producer-private fine route segments");

template <typename T>
std::size_t BytesForElements(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::length_error("BloomFilter performance element bytes overflow size_t");
    }
    return count * sizeof(T);
}

template <typename T>
class ObservableDeviceAllocator {
public:
    using ValueType = T;
    using value_type = T;

    T* Allocate(std::size_t count) const
    {
        void* memory = nullptr;
        if (aclrtMalloc(&memory, sizeof(T) * count, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS) {
            return nullptr;
        }
        ++allocationCount_;
        lastAllocationElements_ = count;
        return static_cast<T*>(memory);
    }

    void Deallocate(T* pointer)
    {
        if (pointer != nullptr) {
            (void)aclrtFree(pointer);
        }
    }

    static void ResetStatistics() noexcept
    {
        allocationCount_ = 0;
        lastAllocationElements_ = 0;
    }

    static std::size_t AllocationCount() noexcept { return allocationCount_; }

    static std::size_t LastAllocationElements() noexcept { return lastAllocationElements_; }

private:
    inline static std::size_t allocationCount_{0};
    inline static std::size_t lastAllocationElements_{0};
};

inline void RequireDeviceMemory(std::size_t requiredBytes)
{
    std::size_t freeBytes = 0;
    std::size_t totalBytes = 0;
    CheckAcl(aclrtGetMemInfo(ACL_HBM_MEM, &freeBytes, &totalBytes), "aclrtGetMemInfo");
    (void)totalBytes;
    if (requiredBytes > std::numeric_limits<std::size_t>::max() - DEVICE_MEMORY_RESERVE ||
        freeBytes < requiredBytes + DEVICE_MEMORY_RESERVE) {
        throw std::runtime_error("BloomFilter performance case requires " + std::to_string(requiredBytes) +
                                 " device bytes plus reserve, but only " + std::to_string(freeBytes) +
                                 " bytes are free");
    }
}

template <typename Key>
using H8V1Policy = aclco::BloomFilterPolicy<Key, aclco::xxhash_64<Key>, std::uint32_t, 8, 8, 8, 1, 8, 1, false, false>;

template <typename Key, typename Policy = aclco::BloomFilterPolicy<Key>,
          typename Allocator = aclco::DefaultAllocator<typename Policy::WordType>>
using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy, Allocator>;

class EventTimer {
public:
    EventTimer()
    {
        constexpr std::uint32_t flags = ACL_EVENT_TIME_LINE | ACL_EVENT_SYNC;
        CheckAcl(aclrtCreateEventWithFlag(&start_, flags), "aclrtCreateEventWithFlag start");
        try {
            CheckAcl(aclrtCreateEventWithFlag(&stop_, flags), "aclrtCreateEventWithFlag stop");
        } catch (...) {
            (void)aclrtDestroyEvent(start_);
            start_ = nullptr;
            throw;
        }
    }

    ~EventTimer()
    {
        if (stop_ != nullptr) {
            (void)aclrtDestroyEvent(stop_);
        }
        if (start_ != nullptr) {
            (void)aclrtDestroyEvent(start_);
        }
    }

    EventTimer(EventTimer const&) = delete;
    EventTimer& operator=(EventTimer const&) = delete;

    void RecordStart(aclrtStream stream) { CheckAcl(aclrtRecordEvent(start_, stream), "aclrtRecordEvent start"); }

    void RecordStop(aclrtStream stream) { CheckAcl(aclrtRecordEvent(stop_, stream), "aclrtRecordEvent stop"); }

    double ElapsedUs(aclrtStream stream)
    {
        float elapsedMs = 0.0F;
        CheckAcl(aclrtEventElapsedTime(&elapsedMs, start_, stop_), "aclrtEventElapsedTime");
        CheckAcl(aclrtResetEvent(start_, stream), "aclrtResetEvent start");
        CheckAcl(aclrtResetEvent(stop_, stream), "aclrtResetEvent stop");
        // Reset is asynchronous. Complete both resets before the next sample
        // records these reusable events; this runs after the measured interval.
        Sync(stream);
        return static_cast<double>(elapsedMs) * 1000.0;
    }

private:
    aclrtEvent start_{nullptr};
    aclrtEvent stop_{nullptr};
};

template <typename Submit>
TestResult MeasureAsync(EventTimer& timer, aclrtStream stream, Submit&& submit)
{
    auto const hostStart = std::chrono::steady_clock::now();
    timer.RecordStart(stream);
    std::forward<Submit>(submit)();
    timer.RecordStop(stream);
    Sync(stream);
    double const deviceTimeUs = timer.ElapsedUs(stream);
    auto const hostStop = std::chrono::steady_clock::now();
    double const hostTimeUs = std::chrono::duration<double, std::micro>(hostStop - hostStart).count();
    return TestResult(hostTimeUs, deviceTimeUs, 1);
}

inline TestResult HostResult(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point stop)
{
    double const hostTimeUs = std::chrono::duration<double, std::micro>(stop - start).count();
    // The existing framework uses deviceTimeUs for convergence. Mirror the host
    // duration for host-only lifecycle cases; no ACL event is used for these.
    return TestResult(hostTimeUs, hostTimeUs, 1);
}

template <typename Key>
void PrepareSequentialKeys(DeviceBuffer<Key>& deviceKeys, std::size_t count, aclrtStream stream)
{
    deviceKeys.Resize(count);
    if (count == 0) {
        return;
    }
    std::size_t const stagingCount = std::min(count, KEY_STAGING_ELEMENTS);
    std::vector<Key> hostKeys(stagingCount);
    for (std::size_t offset = 0; offset < count; offset += stagingCount) {
        std::size_t const chunk = std::min(stagingCount, count - offset);
        for (std::size_t i = 0; i < chunk; ++i) {
            hostKeys[i] = static_cast<Key>(offset + i);
        }
        CheckAcl(aclrtMemcpyAsync(deviceKeys.Data() + offset, BytesForElements<Key>(count - offset), hostKeys.data(),
                                  BytesForElements<Key>(chunk), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                 "aclrtMemcpyAsync BloomFilter performance keys H2D");
        // The pageable staging vector is reused by the next chunk.
        Sync(stream);
    }
}

template <typename FilterType, typename Key>
void BuildSequential(FilterType& filter, DeviceBuffer<Key>& stagingKeys, std::size_t count, aclrtStream stream)
{
    if (count == 0) {
        return;
    }
    std::size_t const stagingCount = std::min(count, KEY_STAGING_ELEMENTS);
    stagingKeys.Resize(stagingCount);
    std::vector<Key> hostKeys(stagingCount);
    for (std::size_t offset = 0; offset < count; offset += stagingCount) {
        std::size_t const chunk = std::min(stagingCount, count - offset);
        for (std::size_t i = 0; i < chunk; ++i) {
            hostKeys[i] = static_cast<Key>(offset + i);
        }
        CheckAcl(aclrtMemcpyAsync(stagingKeys.Data(), stagingKeys.Bytes(), hostKeys.data(),
                                  BytesForElements<Key>(chunk), ACL_MEMCPY_HOST_TO_DEVICE, stream),
                 "aclrtMemcpyAsync BloomFilter build keys H2D");
        filter.AddAsync(stagingKeys.Data(), aclco::Extent<std::size_t>{chunk}, stream);
        // Synchronizing each chunk bounds host/device staging storage and ensures
        // the pageable host vector can be overwritten safely.
        Sync(stream);
    }
}

template <typename Policy>
std::size_t ReferenceBuildKeyCount(std::size_t filterBytes)
{
    constexpr std::size_t divisor = 2ull * Policy::patternBits;
    if (filterBytes > std::numeric_limits<std::size_t>::max() / 8ull) {
        throw std::length_error("BloomFilter reference build count overflows size_t");
    }
    return (filterBytes * 8ull) / divisor;
}

template <typename FilterType>
void EmplaceFilter(std::optional<FilterType>& filter, std::size_t numBlocks, aclrtStream stream)
{
    filter.reset();
    filter.emplace(aclco::Extent<std::size_t>{numBlocks}, typename FilterType::PolicyType{},
                   typename FilterType::AllocatorType{}, stream);
}

template <typename FilterType>
void FillFilter(FilterType& filter, std::uint8_t value, aclrtStream stream)
{
    CheckAcl(
        aclrtMemsetAsync(filter.Data(), filter.SizeBytes(), static_cast<int32_t>(value), filter.SizeBytes(), stream),
        "aclrtMemsetAsync BloomFilter performance fill");
}

struct HostContext {
    AclStreamGuard streamGuard;
    aclrtStream stream{streamGuard.stream};
    std::size_t numBlocks{0};
};

template <typename FilterType>
struct SingleFilterContext {
    AclStreamGuard streamGuard;
    aclrtStream stream{streamGuard.stream};
    EventTimer timer;
    std::optional<FilterType> filter;
    std::size_t numBlocks{0};
};

template <typename Key, typename Policy, typename Allocator = aclco::DefaultAllocator<typename Policy::WordType>>
struct InputContext {
    using FilterType = Filter<Key, Policy, Allocator>;

    AclStreamGuard streamGuard;
    aclrtStream stream{streamGuard.stream};
    EventTimer timer;
    std::optional<FilterType> filter;
    DeviceBuffer<Key> keys;
    DeviceBuffer<std::uint8_t> output;
    std::size_t numBlocks{0};
    std::size_t numInputs{0};
};

template <typename FilterType>
struct PairFilterContext {
    AclStreamGuard streamGuard;
    aclrtStream stream{streamGuard.stream};
    EventTimer timer;
    std::optional<FilterType> destination;
    std::optional<FilterType> source;
    std::size_t numBlocks{0};
};

template <typename FilterType>
void Release(SingleFilterContext<FilterType>& context)
{
    Sync(context.stream);
    context.filter.reset();
    context.numBlocks = 0;
}

template <typename Key, typename Policy, typename Allocator>
void Release(InputContext<Key, Policy, Allocator>& context)
{
    Sync(context.stream);
    context.filter.reset();
    context.keys.Reset();
    context.output.Reset();
    context.numBlocks = 0;
    context.numInputs = 0;
}

template <typename FilterType>
void Release(PairFilterContext<FilterType>& context)
{
    Sync(context.stream);
    context.destination.reset();
    context.source.reset();
    context.numBlocks = 0;
}

template <typename Context>
Context& LeakedContext()
{
    // Performance contexts outlive main's AclGlobalGuard if represented as
    // ordinary function statics. Let device reset reclaim the final context so
    // no ACL destructor runs after aclFinalize.
    static Context* context = new Context();
    return *context;
}

template <typename Key>
PairFilterContext<Filter<Key>>& PreparePairFilterContext(int filterSizeMiB, std::uint64_t numInputs)
{
    using Context = PairFilterContext<Filter<Key>>;
    auto& context = LeakedContext<Context>();
    Release(context);
    if constexpr (std::is_same_v<Key, std::uint64_t>) {
        auto& previous = LeakedContext<PairFilterContext<Filter<std::uint32_t>>>();
        Release(previous);
    }

    // NumInput is part of the reference benchmark configuration. Pair-filter
    // operations consume two bit arrays, so no key buffer is allocated.
    (void)CheckedInputCount(numInputs);
    context.numBlocks = BlocksForMiB(filterSizeMiB);
    RequireDeviceMemory(2 * FilterBytesForMiB(filterSizeMiB));
    EmplaceFilter(context.destination, context.numBlocks, context.stream);
    EmplaceFilter(context.source, context.numBlocks, context.stream);
    return context;
}

} // namespace aclco::test::bloom_filter_perf
