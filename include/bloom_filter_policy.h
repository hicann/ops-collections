/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the LICENSE.
 */
/*
 * Portions adapted from NVIDIA cuCollections.
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

#include "hash_functions.h"
#include "macros.h"

namespace aclco {
namespace detail {

COLLECTION_HOST_DEVICE constexpr bool IsPowerOfTwo(std::uint32_t value) noexcept
{
    return value != 0 && (value & (value - 1)) == 0;
}

COLLECTION_HOST_DEVICE constexpr std::uint32_t Log2PowerOfTwo(std::uint32_t value) noexcept
{
    std::uint32_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

template <class Policy>
struct UsesMultiplyHighBlockIndex : std::false_type {};

} // namespace detail

/**
 * @brief Sectorized Bloom filter policy.
 *
 * A 64-bit key hash is split into two halves. The upper half selects one
 * fixed-size filter block; the lower half generates an equal number of
 * fingerprint bits for every word in that block.
 */
template <class Key, class Hash = aclco::xxhash_64<Key>, class Word = std::uint32_t,
          std::uint32_t WordsPerBlock = 32 / sizeof(Word), std::uint32_t PatternBits = WordsPerBlock,
          std::uint32_t AddHorizontalLayout = WordsPerBlock, std::uint32_t AddVerticalLayout = 1,
          std::uint32_t ContainsHorizontalLayout = 1, std::uint32_t ContainsVerticalLayout = WordsPerBlock,
          bool ConditionalAdd = false, bool EarlyExitContains = false>
class BloomFilterPolicy {
public:
    using KeyType = Key;
    using Hasher = Hash;
    using WordType = Word;
    using HashResultType = typename Hasher::ResultType;

    static constexpr std::uint32_t wordsPerBlock = WordsPerBlock;
    static constexpr std::uint32_t patternBits = PatternBits;
    static constexpr std::uint32_t addHorizontalLayout = AddHorizontalLayout;
    static constexpr std::uint32_t addVerticalLayout = AddVerticalLayout;
    static constexpr std::uint32_t containsHorizontalLayout = ContainsHorizontalLayout;
    static constexpr std::uint32_t containsVerticalLayout = ContainsVerticalLayout;
    static constexpr bool conditionalAdd = ConditionalAdd;
    static constexpr bool earlyExitContains = EarlyExitContains;
    static constexpr std::uint32_t wordBits = sizeof(WordType) * 8;
    static constexpr std::uint32_t bitsPerWord = patternBits / wordsPerBlock;
    static constexpr std::uint32_t maxFilterBlocks = std::numeric_limits<std::uint32_t>::max();

    COLLECTION_HOST_DEVICE constexpr BloomFilterPolicy(Hasher hash = Hasher{}) noexcept : hash_{hash}
    {
        static_assert(std::is_unsigned_v<WordType>, "BloomFilter word type must be unsigned");
        static_assert(sizeof(WordType) == 4, "BloomFilter currently supports uint32_t words only");
        static_assert(std::is_same_v<HashResultType, std::uint64_t>, "BloomFilter requires a 64-bit hash result");
        static_assert(detail::IsPowerOfTwo(wordsPerBlock), "WordsPerBlock must be a non-zero power of two");
        static_assert(wordsPerBlock <= 32, "WordsPerBlock must not exceed the reference policy limit");
        static_assert(detail::IsPowerOfTwo(wordBits), "Word bit count must be a power of two");
        static_assert(patternBits >= wordsPerBlock, "PatternBits must provide at least one bit per word");
        static_assert(patternBits <= 64, "PatternBits exceeds the built-in salt table");
        static_assert(patternBits <= wordBits * wordsPerBlock, "PatternBits exceeds the number of bits in a block");
        static_assert(patternBits % wordsPerBlock == 0, "PatternBits must be evenly distributed across all words");
        static_assert(addHorizontalLayout != 0 && addVerticalLayout != 0, "Add layout dimensions must be non-zero");
        static_assert(containsHorizontalLayout != 0 && containsVerticalLayout != 0,
                      "Contains layout dimensions must be non-zero");
        static_assert(wordsPerBlock % (addHorizontalLayout * addVerticalLayout) == 0,
                      "Add layout must evenly divide WordsPerBlock");
        static_assert(wordsPerBlock % (containsHorizontalLayout * containsVerticalLayout) == 0,
                      "Contains layout must evenly divide WordsPerBlock");
    }

    COLLECTION_HOST_DEVICE constexpr std::uint64_t HashKey(KeyType const& key) const noexcept { return hash_(key); }

    template <typename SizeType>
    COLLECTION_HOST_DEVICE constexpr std::uint32_t BlockIndex(std::uint32_t upperHash,
                                                              SizeType numBlocks) const noexcept
    {
        return static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(upperHash) * static_cast<std::uint32_t>(numBlocks)) >> 32);
    }

    template <std::uint32_t WordIndex>
    COLLECTION_HOST_DEVICE constexpr WordType WordPattern(std::uint32_t lowerHash) const noexcept
    {
        static_assert(WordIndex < wordsPerBlock, "word index out of range");
        return WordPatternImpl<WordIndex, 0>(lowerHash);
    }

    /**
     * @brief Generates the word pattern owned by a cooperative lane.
     *
     * @pre lane < wordsPerBlock
     */
    COLLECTION_HOST_DEVICE constexpr WordType WordPatternForLane(std::uint32_t lane,
                                                                 std::uint32_t lowerHash) const noexcept
    {
        return RuntimeWordPatternImpl<0, wordsPerBlock>(lane, lowerHash);
    }

    COLLECTION_HOST_DEVICE constexpr Hasher const& GetHasher() const noexcept { return hash_; }

    COLLECTION_HOST_DEVICE constexpr bool operator==(BloomFilterPolicy const& other) const noexcept
    {
        return hash_ == other.hash_;
    }

    COLLECTION_HOST_DEVICE constexpr bool operator!=(BloomFilterPolicy const& other) const noexcept
    {
        return !(*this == other);
    }

private:
    template <std::uint32_t Begin, std::uint32_t End>
    COLLECTION_HOST_DEVICE static constexpr WordType RuntimeWordPatternImpl(std::uint32_t wordIndex,
                                                                            std::uint32_t lowerHash) noexcept
    {
        static_assert(Begin < End, "invalid word-pattern dispatch range");
        if constexpr (End - Begin == 1) {
            return WordPatternImpl<Begin, 0>(lowerHash);
        } else {
            constexpr std::uint32_t middle = Begin + (End - Begin) / 2;
            if (wordIndex < middle) {
                return RuntimeWordPatternImpl<Begin, middle>(wordIndex, lowerHash);
            }
            return RuntimeWordPatternImpl<middle, End>(wordIndex, lowerHash);
        }
    }

    template <std::uint32_t WordIndex, std::uint32_t BitIndex>
    COLLECTION_HOST_DEVICE static constexpr WordType WordPatternImpl(std::uint32_t lowerHash) noexcept
    {
        if constexpr (BitIndex < bitsPerWord) {
            constexpr std::uint32_t saltIndex = WordIndex * bitsPerWord + BitIndex;
            constexpr std::uint32_t bitIndexWidth = detail::Log2PowerOfTwo(wordBits);
            std::uint32_t const wordBit = (salts_[saltIndex] * lowerHash) >> (32 - bitIndexWidth);
            return static_cast<WordType>((WordType{1} << wordBit) |
                                         WordPatternImpl<WordIndex, BitIndex + 1>(lowerHash));
        } else {
            return WordType{0};
        }
    }

    inline static constexpr std::uint32_t salts_[64] = {
        0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU, 0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U,
        0xb24bcdffU, 0xb6843d6dU, 0x6db04543U, 0x3a12efddU, 0xb0ddd463U, 0x8d22f6e7U, 0xb82f1e53U, 0x7db9f86bU,
        0xc7afe639U, 0xfb135cd7U, 0x693256e1U, 0x9466d871U, 0x23d3d02fU, 0x6461d049U, 0x66a91621U, 0xbaa3006fU,
        0x52fb8d99U, 0x3ea88b4fU, 0x0f470cfdU, 0xb1db79a5U, 0x9809fcd1U, 0xbced4445U, 0x2eb7c737U, 0x2cea6803U,
        0x156f1955U, 0x8813c027U, 0xa26819f9U, 0x4c3b57bdU, 0x7df94487U, 0xb975e769U, 0xb8f20cb5U, 0x5c9e2e77U,
        0x5fb1735fU, 0x3a6f759bU, 0x3c090923U, 0xfced424dU, 0xa187a6a9U, 0x6f070a41U, 0x2c85233bU, 0x7e62258bU,
        0x2771ef17U, 0x13bbf093U, 0x4ff059e5U, 0xe3ce3d0fU, 0xf1b4789fU, 0x9fbb6173U, 0x6a320cf5U, 0x1be2c481U,
        0x7ba8222bU, 0x6fd619b3U, 0x7b1bbf0dU, 0x8b8993adU, 0x448eca95U, 0x82ab09d9U, 0x2ce53909U, 0x4f548685U};

    Hasher hash_;
};

namespace detail {

template <class Key, class Hash, class Word, std::uint32_t WordsPerBlock, std::uint32_t PatternBits,
          std::uint32_t AddHorizontalLayout, std::uint32_t AddVerticalLayout, std::uint32_t ContainsHorizontalLayout,
          std::uint32_t ContainsVerticalLayout, bool ConditionalAdd, bool EarlyExitContains>
struct UsesMultiplyHighBlockIndex<
    BloomFilterPolicy<Key, Hash, Word, WordsPerBlock, PatternBits, AddHorizontalLayout, AddVerticalLayout,
                      ContainsHorizontalLayout, ContainsVerticalLayout, ConditionalAdd, EarlyExitContains>>
    : std::true_type {};

} // namespace detail

/**
 * @brief Compatibility name for the default sectorized Bloom filter policy.
 *
 * This is an exact type alias to BloomFilterPolicy and does not introduce a
 * wrapper, additional state, or a separate execution path.
 */
template <class Hash, class Word, std::uint32_t WordsPerBlock>
using DefaultFilterPolicy = BloomFilterPolicy<typename Hash::ArgumentType, Hash, Word, WordsPerBlock>;

/**
 * @brief Compatibility name for the Arrow-compatible Bloom filter policy.
 *
 * This aliases the current 256-bit block layout explicitly so future changes
 * to BloomFilterPolicy defaults cannot alter its compatibility contract.
 */
template <class Key>
using ArrowFilterPolicy = BloomFilterPolicy<Key, aclco::xxhash_64<Key>, std::uint32_t, 8, 8, 8, 1, 1, 8, false, false>;

} // namespace aclco
