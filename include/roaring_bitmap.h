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
#include <type_traits>

#include <acl/acl.h>

#include "detail/roaring_bitmap/format.h"
#include "extent.h"
#include "roaring_bitmap_ref.h"
#include "utility/allocator.h"

namespace aclco {

/**
 * @brief Owns a portable serialized Roaring bitmap in device memory.
 *
 * The input is parsed on the host and copied to device memory during construction. `Key` must be
 * `uint32_t` or `uint64_t`; the 64-bit specialization consumes the portable bucketed format.
 *
 * @tparam Key Key type stored in the bitmap.
 * @tparam Extent Query-count extent type.
 * @tparam Allocator Allocator for device-resident bytes.
 */
template <typename Key, typename Extent = aclco::Extent<size_t>, typename Allocator = aclco::DefaultAllocator<uint8_t>>
class RoaringBitmap {
    static_assert(std::is_same_v<Key, uint32_t> || std::is_same_v<Key, uint64_t>,
                  "RoaringBitmap key type must be uint32_t or uint64_t");
    static_assert(std::is_same_v<typename Allocator::ValueType, uint8_t>,
                  "RoaringBitmap allocator value type must be uint8_t");

public:
    using KeyType = Key;
    using SizeType = typename Extent::ValueType;
    using AllocatorType = Allocator;
    using RefType = RoaringBitmapRef<Key>;

    /**
     * @brief Constructs from a complete serialized bitmap in host memory.
     *
     * The serialized length is inferred from the format. The caller must provide complete, trusted
     * input because this overload cannot detect reads beyond the host allocation.
     *
     * @param bitmap Host pointer to the serialized bitmap.
     * @param allocator Device-memory allocator.
     * @param stream Stream used for initialization; synchronized before this constructor returns.
     */
    explicit RoaringBitmap(void const* bitmap, Allocator const& allocator = {}, aclrtStream stream = nullptr);

    /**
     * @brief Constructs from complete serialized bitmap data using the default allocator.
     *
     * This overload preserves the `(bitmap, stream)` calling convention used by the
     * RoaringBitmap self-test suite.
     */
    RoaringBitmap(void const* bitmap, aclrtStream stream);

    /**
     * @brief Constructs from a bounded serialized bitmap in host memory.
     *
     * @param bitmap Host pointer to the serialized bitmap.
     * @param bitmapBytes Available bytes at `bitmap`; used for bounds validation.
     * @param allocator Device-memory allocator.
     * @param stream Stream used for initialization; synchronized before this constructor returns.
     */
    RoaringBitmap(void const* bitmap, size_t bitmapBytes, Allocator const& allocator = {},
                  aclrtStream stream = nullptr);

    RoaringBitmap(RoaringBitmap const&) = delete;
    RoaringBitmap& operator=(RoaringBitmap const&) = delete;

    RoaringBitmap(RoaringBitmap&& other) noexcept(std::is_nothrow_move_constructible_v<Allocator>);
    RoaringBitmap& operator=(RoaringBitmap&& other) noexcept(std::is_nothrow_move_assignable_v<Allocator>);

    ~RoaringBitmap();

    /**
     * @brief Queries membership and synchronizes `stream` before returning.
     *
     * @param keys Device pointer to `Key[keyNum]`.
     * @param outputValues Device pointer to `bool[keyNum]`.
     * @param keyNum Number of keys to query.
     * @param stream Stream used for the query.
     */
    void Contains(void const* keys, void* outputValues, Extent keyNum, aclrtStream stream) const;

    /**
     * @brief Enqueues a membership query without synchronizing.
     *
     * The bitmap, input, and output storage must remain alive until work on `stream` completes.
     * Null input and output pointers are accepted only when `keyNum` is zero.
     *
     * @param keys Device pointer to `Key[keyNum]`.
     * @param outputValues Device pointer to `bool[keyNum]`.
     * @param keyNum Number of keys to query.
     * @param stream Stream used for the query.
     */
    void ContainsAsync(void const* keys, void* outputValues, Extent keyNum, aclrtStream stream) const;

    /** @return Number of keys represented by the bitmap. */
    [[nodiscard]] uint64_t Size() const noexcept;
    /** @return Whether the bitmap represents no keys. */
    [[nodiscard]] bool Empty() const noexcept;
    /** @return Device pointer to the copied serialized bytes. */
    [[nodiscard]] uint8_t const* Data() const noexcept;
    /** @return Serialized byte count, excluding internal metadata. */
    [[nodiscard]] uint64_t SizeBytes() const noexcept;
    /** @return Allocator used by this bitmap. */
    [[nodiscard]] Allocator GetAllocator() const;
    /** @return Non-owning view for device-side single-key queries. */
    [[nodiscard]] RefType Ref() const noexcept;

private:
    void Initialize(void const* bitmap, size_t bitmapBytes, aclrtStream stream);
    void Reset() noexcept;
    void MoveFrom(RoaringBitmap&& other) noexcept(std::is_nothrow_move_assignable_v<Allocator>);

    static size_t AlignUp(size_t value, size_t alignment);
    static void CheckAcl(aclError status, char const* operation);

    Allocator allocator_{};
    uint8_t* allocation_{nullptr};
    size_t allocationBytes_{0};
    size_t metadataOffset_{0};
    size_t bucketsOffset_{0};
    uint64_t serializedBytes_{0};
    uint64_t numKeys_{0};
    uint32_t loadAlignment_{0};
};

} // namespace aclco

#include "detail/roaring_bitmap/roaring_bitmap.inl"
