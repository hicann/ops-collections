/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/*
 * Portions adapted from NVIDIA cuCollections.
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

#include "bloom_filter_policy.h"
#include "detail/bloom_filter/bloom_filter_impl.h"
#include "macros.h"

namespace aclco {

/**
 * @brief Non-owning device reference to a BloomFilter.
 */
template <class Key, class SizeType = std::uint64_t, class Policy = BloomFilterPolicy<Key>>
class BloomFilterRef {
public:
    using KeyType = Key;
    using SizeTypeT = SizeType;
    using PolicyType = Policy;
    using WordType = typename PolicyType::WordType;
    static constexpr std::uint32_t wordsPerBlock = PolicyType::wordsPerBlock;

    COLLECTION_SIMT_DEVICE constexpr BloomFilterRef(__gm__ WordType* words, SizeType numBlocks,
                                                    PolicyType policy = PolicyType{}) noexcept
        : impl_{words, numBlocks, policy}
    {}

    COLLECTION_SIMT_DEVICE void Add(KeyType const& key) const noexcept { impl_.Add(key); }

    COLLECTION_SIMT_DEVICE bool Contains(KeyType const& key) const noexcept { return impl_.Contains(key); }

    template <std::uint32_t WordIndex>
    COLLECTION_SIMT_DEVICE void AddWord(std::uint32_t block, std::uint32_t lowerHash) const noexcept
    {
        impl_.template AddWord<WordIndex>(block, lowerHash);
    }

    template <std::uint32_t WordIndex>
    COLLECTION_SIMT_DEVICE bool ContainsWord(std::uint32_t block, std::uint32_t lowerHash) const noexcept
    {
        return impl_.template ContainsWord<WordIndex>(block, lowerHash);
    }

    COLLECTION_SIMT_DEVICE constexpr __gm__ WordType* Data() const noexcept { return impl_.Data(); }

    COLLECTION_SIMT_DEVICE constexpr SizeType BlockExtent() const noexcept { return impl_.BlockExtent(); }

    COLLECTION_SIMT_DEVICE constexpr PolicyType const& GetPolicy() const noexcept { return impl_.GetPolicy(); }

private:
    detail::BloomFilterImpl<KeyType, SizeType, PolicyType> impl_;
};

} // namespace aclco
