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

#include <cstddef>
#include <cstdint>

#include "detail/roaring_bitmap/format.h"
#include "macros.h"

namespace aclco::detail {

constexpr uint32_t kRoaringBinarySearchThreshold = 8;

template <bool Aligned>
COLLECTION_SIMT_DEVICE uint16_t RoaringLoad16(__gm__ uint8_t const* ptr) noexcept
{
    if constexpr (Aligned) {
        return *reinterpret_cast<__gm__ uint16_t const*>(ptr);
    }
    return static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8U);
}

template <bool Aligned>
COLLECTION_SIMT_DEVICE uint32_t RoaringLoad32(__gm__ uint8_t const* ptr) noexcept
{
    if constexpr (Aligned) {
        return *reinterpret_cast<__gm__ uint32_t const*>(ptr);
    }
    return static_cast<uint32_t>(ptr[0]) | (static_cast<uint32_t>(ptr[1]) << 8U) |
           (static_cast<uint32_t>(ptr[2]) << 16U) | (static_cast<uint32_t>(ptr[3]) << 24U);
}

COLLECTION_SIMT_DEVICE bool RoaringCheckBit(__gm__ uint8_t const* bitmap, uint32_t index) noexcept
{
    return (bitmap[index / 8U] & static_cast<uint8_t>(1U << (index % 8U))) != 0;
}

template <bool Aligned>
COLLECTION_SIMT_DEVICE bool RoaringContainsArray(__gm__ uint8_t const* container, uint16_t lower,
                                                 uint32_t cardinality) noexcept
{
    if (cardinality < kRoaringBinarySearchThreshold) {
        for (uint32_t i = 0; i < cardinality; ++i) {
            uint16_t element = RoaringLoad16<Aligned>(container + i * sizeof(uint16_t));
            if (element == lower) {
                return true;
            }
        }
        return false;
    }

    uint32_t left = 0;
    uint32_t right = cardinality;
    while (left < right) {
        uint32_t mid = left + (right - left) / 2U;
        uint16_t element = RoaringLoad16<Aligned>(container + mid * sizeof(uint16_t));
        if (element == lower) {
            return true;
        }
        if (element < lower) {
            left = mid + 1U;
        } else {
            right = mid;
        }
    }
    return false;
}

template <bool Aligned>
COLLECTION_SIMT_DEVICE bool RoaringContainsRun(__gm__ uint8_t const* container, uint16_t lower) noexcept
{
    uint16_t numRuns = RoaringLoad16<Aligned>(container);
    for (uint32_t i = 0; i < numRuns; ++i) {
        __gm__ uint8_t const* run = container + (i * 2U + 1U) * sizeof(uint16_t);
        uint16_t start = RoaringLoad16<Aligned>(run);
        uint32_t end = static_cast<uint32_t>(start) + RoaringLoad16<Aligned>(run + sizeof(uint16_t));
        if (start <= lower && lower <= end) {
            return true;
        }
        if (start > lower) {
            return false;
        }
    }
    return false;
}

template <bool Aligned16, bool Aligned32>
COLLECTION_SIMT_DEVICE bool RoaringContainsContainer(__gm__ uint8_t const* bitmap,
                                                     __gm__ RoaringBitmapMetadata32 const* metadata, uint16_t lower,
                                                     uint32_t index) noexcept
{
    uint32_t offset = metadata->computedOffsets[index];
    if (metadata->offsetsInSerializedData != 0) {
        offset = RoaringLoad32<Aligned32>(bitmap + metadata->containerOffsets + index * sizeof(uint32_t));
    }
    __gm__ uint8_t const* container = bitmap + offset;

    bool isRun = metadata->hasRun != 0 && RoaringCheckBit(bitmap + metadata->runContainerBitmap, index);
    if (isRun) {
        return RoaringContainsRun<Aligned16>(container, lower);
    }

    uint32_t cardinality = static_cast<uint32_t>(RoaringLoad16<Aligned16>(bitmap + metadata->keyCards +
                                                                          (index * 2U + 1U) * sizeof(uint16_t))) +
                           1U;
    if (cardinality <= kRoaringMaxArrayCardinality) {
        return RoaringContainsArray<Aligned16>(container, lower, cardinality);
    }
    return RoaringCheckBit(container, lower);
}

template <bool Aligned16, bool Aligned32>
COLLECTION_SIMT_DEVICE bool RoaringContains32(__gm__ uint8_t const* bitmap,
                                              __gm__ RoaringBitmapMetadata32 const* metadata, uint32_t value) noexcept
{
    if (metadata->numKeys == 0) {
        return false;
    }

    uint16_t upper = static_cast<uint16_t>(value >> 16U);
    uint16_t lower = static_cast<uint16_t>(value & 0xffffU);
    uint32_t numContainers = metadata->numContainers;

    if (numContainers < kRoaringBinarySearchThreshold) {
        for (uint32_t i = 0; i < numContainers; ++i) {
            uint16_t key = RoaringLoad16<Aligned16>(bitmap + metadata->keyCards + i * 2U * sizeof(uint16_t));
            if (key == upper) {
                return RoaringContainsContainer<Aligned16, Aligned32>(bitmap, metadata, lower, i);
            }
            if (key > upper) {
                return false;
            }
        }
        return false;
    }

    uint32_t left = 0;
    uint32_t right = numContainers;
    while (left < right) {
        uint32_t mid = left + (right - left) / 2U;
        uint16_t key = RoaringLoad16<Aligned16>(bitmap + metadata->keyCards + mid * 2U * sizeof(uint16_t));
        if (key == upper) {
            return RoaringContainsContainer<Aligned16, Aligned32>(bitmap, metadata, lower, mid);
        }
        if (key < upper) {
            left = mid + 1U;
        } else {
            right = mid;
        }
    }
    return false;
}

} // namespace aclco::detail

namespace aclco {

template <typename Key>
class RoaringBitmapRef;

/**
 * @brief Non-owning device-query view of a 32-bit Roaring bitmap.
 *
 * The owning `RoaringBitmap` and all referenced device storage must outlive this view and any work
 * that uses it.
 */
template <>
class RoaringBitmapRef<uint32_t> {
public:
    using KeyType = uint32_t;

    /** @brief Constructs a view from device-resident serialized bytes and metadata. */
    COLLECTION_HOST_DEVICE explicit constexpr RoaringBitmapRef(__gm__ uint8_t const* bitmap,
                                                               __gm__ detail::RoaringBitmapMetadata32 const* metadata,
                                                               uint64_t numKeys, uint64_t sizeBytes) noexcept
        : bitmap_{bitmap}, metadata_{metadata}, numKeys_{numKeys}, sizeBytes_{sizeBytes}
    {}

    /** @return Whether `value` is present. */
    COLLECTION_SIMT_DEVICE bool Contains(uint32_t value) const noexcept
    {
        return detail::RoaringContains32<false, false>(bitmap_, metadata_, value);
    }

    /** @return Number of keys represented by the bitmap. */
    COLLECTION_HOST_DEVICE constexpr uint64_t Size() const noexcept { return numKeys_; }

    /** @return Whether the bitmap represents no keys. */
    COLLECTION_HOST_DEVICE constexpr bool Empty() const noexcept { return numKeys_ == 0; }

    /** @return Device pointer to the serialized bytes. */
    COLLECTION_HOST_DEVICE constexpr __gm__ uint8_t const* Data() const noexcept { return bitmap_; }

    /** @return Serialized byte count. */
    COLLECTION_HOST_DEVICE constexpr uint64_t SizeBytes() const noexcept { return sizeBytes_; }

private:
    __gm__ uint8_t const* bitmap_;
    __gm__ detail::RoaringBitmapMetadata32 const* metadata_;
    uint64_t numKeys_;
    uint64_t sizeBytes_;
};

/**
 * @brief Non-owning device-query view of a portable 64-bit Roaring bitmap.
 *
 * The owning `RoaringBitmap` and all referenced device storage must outlive this view and any work
 * that uses it.
 */
template <>
class RoaringBitmapRef<uint64_t> {
public:
    using KeyType = uint64_t;

    /** @brief Constructs a view from device-resident serialized bytes, metadata, and bucket descriptors. */
    COLLECTION_HOST_DEVICE explicit constexpr RoaringBitmapRef(__gm__ uint8_t const* bitmap,
                                                               __gm__ detail::RoaringBitmapMetadata64 const* metadata,
                                                               __gm__ detail::RoaringBitmapBucket const* buckets,
                                                               uint64_t numKeys, uint64_t sizeBytes) noexcept
        : bitmap_{bitmap}, metadata_{metadata}, buckets_{buckets}, numKeys_{numKeys}, sizeBytes_{sizeBytes}
    {}

    /** @return Whether `value` is present. */
    COLLECTION_SIMT_DEVICE bool Contains(uint64_t value) const noexcept
    {
        uint32_t bucketKey = static_cast<uint32_t>(value >> 32U);
        uint32_t bucketValue = static_cast<uint32_t>(value & 0xffffffffULL);
        uint64_t left = 0;
        uint64_t right = metadata_->numBuckets;
        while (left < right) {
            uint64_t mid = left + (right - left) / 2U;
            uint32_t key = buckets_[mid].key;
            if (key == bucketKey) {
                return detail::RoaringContains32<false, false>(bitmap_ + buckets_[mid].byteOffset,
                                                               &buckets_[mid].metadata, bucketValue);
            }
            if (key < bucketKey) {
                left = mid + 1U;
            } else {
                right = mid;
            }
        }
        return false;
    }

    /** @return Number of keys represented by the bitmap. */
    COLLECTION_HOST_DEVICE constexpr uint64_t Size() const noexcept { return numKeys_; }

    /** @return Whether the bitmap represents no keys. */
    COLLECTION_HOST_DEVICE constexpr bool Empty() const noexcept { return numKeys_ == 0; }

    /** @return Device pointer to the serialized bytes. */
    COLLECTION_HOST_DEVICE constexpr __gm__ uint8_t const* Data() const noexcept { return bitmap_; }

    /** @return Serialized byte count. */
    COLLECTION_HOST_DEVICE constexpr uint64_t SizeBytes() const noexcept { return sizeBytes_; }

private:
    __gm__ uint8_t const* bitmap_;
    __gm__ detail::RoaringBitmapMetadata64 const* metadata_;
    __gm__ detail::RoaringBitmapBucket const* buckets_;
    uint64_t numKeys_;
    uint64_t sizeBytes_;
};

} // namespace aclco
