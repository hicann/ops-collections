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
#include <type_traits>
#include "kernel_operator.h"
#include "hash_functions.h"
#include "macros.h"
#include "simt_api/device_atomic_functions.h"
#include "simt_api/device_functions.h"

namespace aclco {
/** Non-owning device reference. The owner must outlive all kernels using this reference. */
template <class Key>
class HyperLogLogRef {
    static_assert(std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::int64_t>);

public:
    COLLECTION_SIMT_DEVICE constexpr HyperLogLogRef(__gm__ std::uint32_t* words, std::uint32_t precision)
        : words_(words), precision_(precision)
    {}

    COLLECTION_SIMT_DEVICE void Add(Key const& key) const noexcept
    {
        auto const hash = aclco::xxhash_64<Key>{0}(key);
        auto const index = static_cast<std::uint32_t>(hash >> (64U - precision_));
        // The sentinel makes CLZ well-defined and caps rank at 65 - precision.
        auto const tail = (hash << precision_) | (std::uint64_t{1} << (precision_ - 1U));
        auto const rank = static_cast<std::uint32_t>(__clz(static_cast<long long>(tail))) + 1U;
        auto const shift = (index & 3U) * 8U;
        auto* word = words_ + (index >> 2U);
        auto observed = *word;
        while (((observed >> shift) & 255U) < rank) {
            auto const desired = (observed & ~(255U << shift)) | (rank << shift);
            auto const previous = asc_atomic_cas(word, observed, desired);
            if (previous == observed)
                break;
            observed = previous;
        }
    }
    COLLECTION_SIMT_DEVICE __gm__ std::uint32_t* Data() const noexcept { return words_; }

private:
    __gm__ std::uint32_t* words_;
    std::uint32_t precision_;
};
} // namespace aclco
