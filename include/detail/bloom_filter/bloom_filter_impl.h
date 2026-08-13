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

// CANN's public kernel entry must precede the lower-level SIMT headers.
#include "kernel_operator.h"

#include "bloom_filter_policy.h"
#include "macros.h"
#include "simt_api/device_atomic_functions.h"

namespace aclco::detail {

template <class Key, class SizeType, class Policy>
class BloomFilterImpl {
public:
    using KeyType = Key;
    using WordType = typename Policy::WordType;
    using PolicyType = Policy;

    COLLECTION_SIMT_DEVICE constexpr BloomFilterImpl(__gm__ WordType* words, SizeType numBlocks,
                                                     PolicyType policy = PolicyType{}) noexcept
        : words_{words}, numBlocks_{numBlocks}, policy_{policy}
    {}

    COLLECTION_SIMT_DEVICE void Add(KeyType const& key) const noexcept
    {
        std::uint64_t const hash = policy_.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
        std::uint32_t const block = policy_.BlockIndex(upperHash, numBlocks_);
        AddWords<0>(block, lowerHash);
    }

    COLLECTION_SIMT_DEVICE bool Contains(KeyType const& key) const noexcept
    {
        std::uint64_t const hash = policy_.HashKey(key);
        std::uint32_t const upperHash = static_cast<std::uint32_t>(hash >> 32);
        std::uint32_t const lowerHash = static_cast<std::uint32_t>(hash);
        std::uint32_t const block = policy_.BlockIndex(upperHash, numBlocks_);
        return ContainsWords<0>(block, lowerHash);
    }

    template <std::uint32_t WordIndex>
    COLLECTION_SIMT_DEVICE void AddWord(std::uint32_t block, std::uint32_t lowerHash) const noexcept
    {
        static_assert(WordIndex < PolicyType::wordsPerBlock, "word index out of range");
        WordType const pattern = policy_.template WordPattern<WordIndex>(lowerHash);
        SizeType const offset = static_cast<SizeType>(block) * PolicyType::wordsPerBlock + WordIndex;
        if constexpr (PolicyType::conditionalAdd) {
            if ((words_[offset] & pattern) != pattern) {
                asc_atomic_or(words_ + offset, pattern);
            }
        } else {
            asc_atomic_or(words_ + offset, pattern);
        }
    }

    template <std::uint32_t WordIndex>
    COLLECTION_SIMT_DEVICE bool ContainsWord(std::uint32_t block, std::uint32_t lowerHash) const noexcept
    {
        static_assert(WordIndex < PolicyType::wordsPerBlock, "word index out of range");
        WordType const pattern = policy_.template WordPattern<WordIndex>(lowerHash);
        SizeType const offset = static_cast<SizeType>(block) * PolicyType::wordsPerBlock + WordIndex;
        return (words_[offset] & pattern) == pattern;
    }

    COLLECTION_SIMT_DEVICE constexpr __gm__ WordType* Data() const noexcept { return words_; }

    COLLECTION_SIMT_DEVICE constexpr SizeType BlockExtent() const noexcept { return numBlocks_; }

    COLLECTION_SIMT_DEVICE constexpr PolicyType const& GetPolicy() const noexcept { return policy_; }

private:
    template <std::uint32_t WordIndex>
    COLLECTION_SIMT_DEVICE void AddWords(std::uint32_t block, std::uint32_t lowerHash) const noexcept
    {
        if constexpr (WordIndex < PolicyType::wordsPerBlock) {
            AddWord<WordIndex>(block, lowerHash);
            AddWords<WordIndex + 1>(block, lowerHash);
        }
    }

    template <std::uint32_t WordIndex>
    COLLECTION_SIMT_DEVICE bool ContainsWords(std::uint32_t block, std::uint32_t lowerHash) const noexcept
    {
        if constexpr (WordIndex < PolicyType::wordsPerBlock) {
            bool const match = ContainsWord<WordIndex>(block, lowerHash);
            if constexpr (PolicyType::earlyExitContains) {
                if (!match) {
                    return false;
                }
                return ContainsWords<WordIndex + 1>(block, lowerHash);
            } else {
                return ContainsWords<WordIndex + 1>(block, lowerHash) && match;
            }
        } else {
            return true;
        }
    }

    __gm__ WordType* words_;
    SizeType numBlocks_;
    PolicyType policy_;
};

} // namespace aclco::detail
