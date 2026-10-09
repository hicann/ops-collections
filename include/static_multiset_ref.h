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

#include <cstdint>
#include "kernel_operator.h"
#include "macros.h"
#include "hash_functions.h"

namespace aclco {
/**
 * @brief Read-only device reference to an exact multiset representation.
 * @tparam Key int32_t or int64_t.
 * @details Host mutations must finish before queries.
 * Keys and uint64 counts occupy separate contiguous arrays.
 * The two pointers address the first key and count; capacity is a power of two.
 */
template <typename Key>
class StaticMultisetRef {
public:
    static constexpr uint32_t KEY_STRIDE = 1;
    static constexpr uint32_t COUNT_STRIDE = 1;
    /**
     * @brief Construct a non-owning reference; optional arrays may be null.
     * @param keys Hash key array, required only for a materialized hash table.
     * @param counts Multiplicity array, required for materialized representations.
     * @param capacity Power-of-two number of addressable slots.
     * @param empty Reserved sentinel key.
     * @param direct Whether keys use encoded offsets rather than hashing.
     * @param base Smallest encoded key in the direct representation.
     * @param uniformCount Interval multiplicity, or zero for a materialized table.
     * @param uniformLength Number of consecutive keys in the interval.
     */
    COLLECTION_SIMT_DEVICE StaticMultisetRef(__gm__ Key* keys, __gm__ uint64_t* counts, uint64_t capacity, Key empty,
                                             bool direct = false, uint64_t base = 0, uint64_t uniformCount = 0,
                                             uint64_t uniformLength = 0)
        : keys_(keys),
          counts_(counts),
          capacity_(capacity),
          empty_(empty),
          direct_(direct),
          base_(base),
          uniformCount_(uniformCount),
          uniformLength_(uniformLength)
    {}

    /**
     * @brief Return exact multiplicity, or zero for an absent or reserved key.
     * @param key Query key.
     * @return Number of stored occurrences.
     */
    COLLECTION_SIMT_DEVICE uint64_t Count(Key key) const
    {
        if (key == empty_) {
            return 0;
        }
        if (direct_) {
            uint64_t encoded = Encode(key);
            if (encoded < base_) {
                return 0;
            }
            if (uniformCount_) {
                return encoded - base_ < uniformLength_ ? uniformCount_ : 0;
            }
            if (!counts_) {
                return 0;
            }
            if (encoded - base_ >= capacity_) {
                return 0;
            }
            return counts_[(encoded - base_) * COUNT_STRIDE] - 1;
        }
        if (!keys_) {
            return 0;
        }
        uint64_t slot = Hash(key) & (capacity_ - 1);
        for (uint64_t visited = 0; visited < capacity_; ++visited) {
            Key stored = keys_[slot * KEY_STRIDE];
            if (stored == key) {
                return counts_[slot * COUNT_STRIDE];
            }
            if (stored == empty_) {
                return 0;
            }
            if (++slot == capacity_) {
                slot = 0;
            }
        }
        return 0;
    }
    /**
     * @brief Count using a hash and equality consistent with table construction.
     * @param key Query key.
     * @param equal Exact key equality predicate.
     * @param hash Hash consistent with Hash() for every valid key.
     * @return Number of occurrences; zero for absent or reserved keys.
     */
    template <typename Equal, typename HashFunction>
    COLLECTION_SIMT_DEVICE uint64_t Count(Key key, Equal equal, HashFunction hash) const
    {
        if (key == empty_) {
            return 0;
        }
        if (direct_) {
            return Count(key);
        }
        if (!keys_) {
            return 0;
        }
        uint64_t slot = static_cast<uint64_t>(hash(key)) & (capacity_ - 1);
        for (uint64_t visited = 0; visited < capacity_; ++visited) {
            Key stored = keys_[slot * KEY_STRIDE];
            if (stored == empty_) {
                return 0;
            }
            if (equal(stored, key)) {
                return counts_[slot * COUNT_STRIDE];
            }
            if (++slot == capacity_) {
                slot = 0;
            }
        }
        return 0;
    }
    /**
     * @brief Test membership without reading hash-table multiplicities.
     * @param key Query key.
     * @return Whether the key is present.
     */
    COLLECTION_SIMT_DEVICE bool Contains(Key key) const
    {
        if (key == empty_) {
            return false;
        }
        if (direct_) {
            return Count(key) != 0;
        }
        if (!keys_) {
            return false;
        }
        uint64_t slot = Hash(key) & (capacity_ - 1);
        for (uint64_t visited = 0; visited < capacity_; ++visited) {
            Key stored = keys_[slot * KEY_STRIDE];
            if (stored == key) {
                return true;
            }
            if (stored == empty_) {
                return false;
            }
            if (++slot == capacity_) {
                slot = 0;
            }
        }
        return false;
    }
    /** @brief Return the matching key or the reserved sentinel.
     * @param key Query key.
     * @return key when present, otherwise empty.
     */
    COLLECTION_SIMT_DEVICE Key Find(Key key) const { return Contains(key) ? key : empty_; }
    COLLECTION_SIMT_DEVICE static uint64_t Encode(Key key)
    {
        return static_cast<uint64_t>(static_cast<int64_t>(key)) ^ (uint64_t{1} << 63);
    }
    COLLECTION_SIMT_DEVICE static uint64_t Hash(Key key)
    {
        if constexpr (sizeof(Key) == 4) {
            return murmurhash3_fmix32<Key>{}(key);
        } else {
            return murmurhash3_fmix64<Key>{}(key);
        }
    }

private:
    __gm__ Key* keys_;
    __gm__ uint64_t* counts_;
    uint64_t capacity_;
    Key empty_;
    bool direct_;
    uint64_t base_;
    uint64_t uniformCount_, uniformLength_;
};
} // namespace aclco
