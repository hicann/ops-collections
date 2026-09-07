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
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace aclco::detail {

constexpr uint32_t kRoaringSerialCookieNoRun = 12346;
constexpr uint32_t kRoaringSerialCookie = 12347;
constexpr uint32_t kRoaringMaxContainers = 1U << 16;
constexpr uint32_t kRoaringMaxArrayCardinality = 4096;
constexpr uint32_t kRoaringBitsetContainerBytes = 8192;
constexpr uint32_t kRoaringNoOffsetThreshold = 4;
constexpr uint32_t kRoaringAligned16 = 1U;
constexpr uint32_t kRoaringAligned32 = 2U;
constexpr size_t kRoaringUnboundedBytes = std::numeric_limits<size_t>::max();

struct RoaringBitmapMetadata32 {
    uint64_t sizeBytes{0};
    uint64_t numKeys{0};
    uint32_t runContainerBitmap{0};
    uint32_t keyCards{0};
    uint32_t containerOffsets{0};
    uint32_t computedOffsets[kRoaringNoOffsetThreshold]{};
    uint32_t numContainers{0};
    uint8_t hasRun{0};
    uint8_t offsetsInSerializedData{1};
    uint8_t aligned16{0};
    uint8_t offsetsAligned{0};
};

struct RoaringBitmapBucket {
    uint32_t key{0};
    uint32_t reserved{0};
    uint64_t byteOffset{0};
    RoaringBitmapMetadata32 metadata{};
};

struct RoaringBitmapMetadata64 {
    uint64_t sizeBytes{0};
    uint64_t numKeys{0};
    uint64_t numBuckets{0};
};

static_assert(std::is_trivially_copyable_v<RoaringBitmapMetadata32>);
static_assert(std::is_trivially_copyable_v<RoaringBitmapBucket>);
static_assert(std::is_trivially_copyable_v<RoaringBitmapMetadata64>);

class RoaringBitmapReader {
public:
    RoaringBitmapReader(uint8_t const* data, size_t sizeBytes) : data_{data}, sizeBytes_{sizeBytes}
    {
        if (data_ == nullptr) {
            throw std::invalid_argument("RoaringBitmap: bitmap must not be null");
        }
    }

    void Require(size_t offset, size_t bytes) const
    {
        if (sizeBytes_ == kRoaringUnboundedBytes) {
            return;
        }
        if (offset > sizeBytes_ || bytes > sizeBytes_ - offset) {
            throw std::invalid_argument("RoaringBitmap: truncated serialized bitmap");
        }
    }

    uint8_t Read8(size_t offset) const
    {
        Require(offset, sizeof(uint8_t));
        return data_[offset];
    }

    uint16_t Read16(size_t offset) const
    {
        Require(offset, sizeof(uint16_t));
        return static_cast<uint16_t>(data_[offset]) | (static_cast<uint16_t>(data_[offset + 1]) << 8U);
    }

    uint32_t Read32(size_t offset) const
    {
        Require(offset, sizeof(uint32_t));
        return static_cast<uint32_t>(data_[offset]) | (static_cast<uint32_t>(data_[offset + 1]) << 8U) |
               (static_cast<uint32_t>(data_[offset + 2]) << 16U) | (static_cast<uint32_t>(data_[offset + 3]) << 24U);
    }

    uint64_t Read64(size_t offset) const
    {
        uint64_t low = Read32(offset);
        uint64_t high = Read32(CheckedAdd(offset, sizeof(uint32_t)));
        return low | (high << 32U);
    }

    bool IsBounded() const noexcept { return sizeBytes_ != kRoaringUnboundedBytes; }

    size_t SizeBytes() const noexcept { return sizeBytes_; }

    static size_t CheckedAdd(size_t lhs, size_t rhs)
    {
        if (rhs > std::numeric_limits<size_t>::max() - lhs) {
            throw std::invalid_argument("RoaringBitmap: serialized size overflow");
        }
        return lhs + rhs;
    }

    static size_t CheckedMul(size_t lhs, size_t rhs)
    {
        if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
            throw std::invalid_argument("RoaringBitmap: serialized size overflow");
        }
        return lhs * rhs;
    }

private:
    uint8_t const* data_;
    size_t sizeBytes_;
};

inline bool RoaringRunContainerAt(RoaringBitmapReader const& reader, RoaringBitmapMetadata32 const& metadata,
                                  uint32_t index)
{
    if (metadata.hasRun == 0) {
        return false;
    }
    uint8_t bits = reader.Read8(RoaringBitmapReader::CheckedAdd(metadata.runContainerBitmap, index / 8U));
    return (bits & static_cast<uint8_t>(1U << (index % 8U))) != 0;
}

inline size_t RoaringContainerEnd(RoaringBitmapReader const& reader, size_t containerOffset, bool isRun,
                                  uint32_t cardinality)
{
    size_t containerBytes = 0;
    if (isRun) {
        uint16_t numRuns = reader.Read16(containerOffset);
        containerBytes = RoaringBitmapReader::CheckedAdd(
            sizeof(uint16_t), RoaringBitmapReader::CheckedMul(static_cast<size_t>(numRuns), 2U * sizeof(uint16_t)));
    } else if (cardinality <= kRoaringMaxArrayCardinality) {
        containerBytes = RoaringBitmapReader::CheckedMul(cardinality, sizeof(uint16_t));
    } else {
        containerBytes = kRoaringBitsetContainerBytes;
    }
    reader.Require(containerOffset, containerBytes);
    return RoaringBitmapReader::CheckedAdd(containerOffset, containerBytes);
}

inline RoaringBitmapMetadata32 ParseRoaringBitmap32(uint8_t const* bitmap, size_t bitmapBytes)
{
    RoaringBitmapReader reader{bitmap, bitmapBytes};
    RoaringBitmapMetadata32 metadata{};
    uint32_t cookie = reader.Read32(0);
    size_t cursor = sizeof(uint32_t);

    if ((cookie & 0xffffU) == kRoaringSerialCookie) {
        metadata.hasRun = 1;
        metadata.numContainers = (cookie >> 16U) + 1U;
        metadata.runContainerBitmap = static_cast<uint32_t>(cursor);
        cursor = RoaringBitmapReader::CheckedAdd(cursor, (static_cast<size_t>(metadata.numContainers) + 7U) / 8U);
    } else if (cookie == kRoaringSerialCookieNoRun) {
        metadata.numContainers = reader.Read32(cursor);
        cursor = RoaringBitmapReader::CheckedAdd(cursor, sizeof(uint32_t));
    } else {
        throw std::invalid_argument("RoaringBitmap: unsupported serialized cookie");
    }

    if (metadata.numContainers > kRoaringMaxContainers) {
        throw std::invalid_argument("RoaringBitmap: container count is out of range");
    }

    if (cursor > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument("RoaringBitmap: metadata offset exceeds uint32 range");
    }
    metadata.keyCards = static_cast<uint32_t>(cursor);
    size_t keyCardsBytes = RoaringBitmapReader::CheckedMul(metadata.numContainers, 2U * sizeof(uint16_t));
    reader.Require(cursor, keyCardsBytes);

    uint32_t previousKey = 0;
    for (uint32_t i = 0; i < metadata.numContainers; ++i) {
        size_t entry = RoaringBitmapReader::CheckedAdd(cursor,
                                                       RoaringBitmapReader::CheckedMul(i, 2U * sizeof(uint16_t)));
        uint32_t key = reader.Read16(entry);
        if (i != 0 && key <= previousKey) {
            throw std::invalid_argument("RoaringBitmap: container keys must be strictly increasing");
        }
        previousKey = key;
        uint64_t cardinality = static_cast<uint64_t>(reader.Read16(entry + sizeof(uint16_t))) + 1U;
        if (cardinality > std::numeric_limits<uint64_t>::max() - metadata.numKeys) {
            throw std::invalid_argument("RoaringBitmap: key count overflow");
        }
        metadata.numKeys += cardinality;
    }
    cursor = RoaringBitmapReader::CheckedAdd(cursor, keyCardsBytes);

    bool hasSerializedOffsets = metadata.hasRun == 0 || metadata.numContainers >= kRoaringNoOffsetThreshold;
    metadata.offsetsInSerializedData = hasSerializedOffsets ? 1 : 0;
    if (hasSerializedOffsets) {
        if (cursor > std::numeric_limits<uint32_t>::max()) {
            throw std::invalid_argument("RoaringBitmap: offset table exceeds uint32 range");
        }
        metadata.containerOffsets = static_cast<uint32_t>(cursor);
        size_t offsetsBytes = RoaringBitmapReader::CheckedMul(metadata.numContainers, sizeof(uint32_t));
        reader.Require(cursor, offsetsBytes);
        cursor = RoaringBitmapReader::CheckedAdd(cursor, offsetsBytes);
    }

    if (metadata.numContainers == 0) {
        metadata.sizeBytes = cursor;
        return metadata;
    }

    size_t previousEnd = cursor;
    for (uint32_t i = 0; i < metadata.numContainers; ++i) {
        size_t entry = RoaringBitmapReader::CheckedAdd(metadata.keyCards,
                                                       RoaringBitmapReader::CheckedMul(i, 2U * sizeof(uint16_t)));
        uint32_t cardinality = static_cast<uint32_t>(reader.Read16(entry + sizeof(uint16_t))) + 1U;
        size_t containerOffset = previousEnd;
        if (hasSerializedOffsets) {
            size_t offsetEntry = RoaringBitmapReader::CheckedAdd(metadata.containerOffsets,
                                                                 RoaringBitmapReader::CheckedMul(i, sizeof(uint32_t)));
            containerOffset = reader.Read32(offsetEntry);
            if (containerOffset < cursor || containerOffset < previousEnd) {
                throw std::invalid_argument("RoaringBitmap: invalid container offset");
            }
        } else {
            if (containerOffset > std::numeric_limits<uint32_t>::max()) {
                throw std::invalid_argument("RoaringBitmap: computed offset exceeds uint32 range");
            }
            metadata.computedOffsets[i] = static_cast<uint32_t>(containerOffset);
        }

        previousEnd = RoaringContainerEnd(reader, containerOffset, RoaringRunContainerAt(reader, metadata, i),
                                          cardinality);
    }

    metadata.sizeBytes = previousEnd;
    return metadata;
}

struct ParsedRoaringBitmap64 {
    RoaringBitmapMetadata64 metadata{};
    std::vector<RoaringBitmapBucket> buckets{};
};

inline ParsedRoaringBitmap64 ParseRoaringBitmap64(uint8_t const* bitmap, size_t bitmapBytes)
{
    RoaringBitmapReader reader{bitmap, bitmapBytes};
    ParsedRoaringBitmap64 parsed{};
    parsed.metadata.numBuckets = reader.Read64(0);
    size_t cursor = sizeof(uint64_t);

    if (parsed.metadata.numBuckets > parsed.buckets.max_size()) {
        throw std::invalid_argument("RoaringBitmap: bucket count is out of range");
    }
    if (reader.IsBounded()) {
        size_t minimumBucketBytes = sizeof(uint32_t) + 2U * sizeof(uint32_t);
        if (parsed.metadata.numBuckets > (reader.SizeBytes() - cursor) / minimumBucketBytes) {
            throw std::invalid_argument("RoaringBitmap: truncated 64-bit bucket table");
        }
    }

    parsed.buckets.reserve(static_cast<size_t>(parsed.metadata.numBuckets));
    uint32_t previousKey = 0;
    for (uint64_t i = 0; i < parsed.metadata.numBuckets; ++i) {
        uint32_t bucketKey = reader.Read32(cursor);
        if (i != 0 && bucketKey <= previousKey) {
            throw std::invalid_argument("RoaringBitmap: bucket keys must be strictly increasing");
        }
        previousKey = bucketKey;
        cursor = RoaringBitmapReader::CheckedAdd(cursor, sizeof(uint32_t));

        size_t remaining = kRoaringUnboundedBytes;
        if (reader.IsBounded()) {
            remaining = reader.SizeBytes() - cursor;
        }
        RoaringBitmapMetadata32 bucketMetadata = ParseRoaringBitmap32(bitmap + cursor, remaining);
        RoaringBitmapBucket bucket{};
        bucket.key = bucketKey;
        bucket.byteOffset = cursor;
        bucket.metadata = bucketMetadata;
        parsed.buckets.push_back(bucket);

        if (bucketMetadata.numKeys > std::numeric_limits<uint64_t>::max() - parsed.metadata.numKeys) {
            throw std::invalid_argument("RoaringBitmap: key count overflow");
        }
        parsed.metadata.numKeys += bucketMetadata.numKeys;
        cursor = RoaringBitmapReader::CheckedAdd(cursor, static_cast<size_t>(bucketMetadata.sizeBytes));
    }

    parsed.metadata.sizeBytes = cursor;
    return parsed;
}

template <typename Key>
struct ParsedRoaringBitmap;

template <>
struct ParsedRoaringBitmap<uint32_t> {
    RoaringBitmapMetadata32 metadata{};
};

template <>
struct ParsedRoaringBitmap<uint64_t> {
    RoaringBitmapMetadata64 metadata{};
    std::vector<RoaringBitmapBucket> buckets{};
};

template <typename Key>
inline ParsedRoaringBitmap<Key> ParseRoaringBitmap(uint8_t const* bitmap, size_t bitmapBytes)
{
    static_assert(std::is_same_v<Key, uint32_t> || std::is_same_v<Key, uint64_t>,
                  "RoaringBitmap key type must be uint32_t or uint64_t");
    if constexpr (std::is_same_v<Key, uint32_t>) {
        return ParsedRoaringBitmap<uint32_t>{ParseRoaringBitmap32(bitmap, bitmapBytes)};
    } else {
        auto parsed = ParseRoaringBitmap64(bitmap, bitmapBytes);
        return ParsedRoaringBitmap<uint64_t>{parsed.metadata, std::move(parsed.buckets)};
    }
}

} // namespace aclco::detail
