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

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "tests/common/object_representation.h"

namespace aclco::test {

namespace bloom_golden_detail {

constexpr std::uint64_t prime1 = 11400714785074694791ull;
constexpr std::uint64_t prime2 = 14029467366897019727ull;
constexpr std::uint64_t prime3 = 1609587929392839161ull;
constexpr std::uint64_t prime4 = 9650029242287828579ull;
constexpr std::uint64_t prime5 = 2870177450012600261ull;
constexpr std::uint32_t salts[64] = {
    0x47b6137bU, 0x44974d91U, 0x8824ad5bU, 0xa2b7289dU, 0x705495c7U, 0x2df1424bU, 0x9efc4947U, 0x5c6bfb31U,
    0xb24bcdffU, 0xb6843d6dU, 0x6db04543U, 0x3a12efddU, 0xb0ddd463U, 0x8d22f6e7U, 0xb82f1e53U, 0x7db9f86bU,
    0xc7afe639U, 0xfb135cd7U, 0x693256e1U, 0x9466d871U, 0x23d3d02fU, 0x6461d049U, 0x66a91621U, 0xbaa3006fU,
    0x52fb8d99U, 0x3ea88b4fU, 0x0f470cfdU, 0xb1db79a5U, 0x9809fcd1U, 0xbced4445U, 0x2eb7c737U, 0x2cea6803U,
    0x156f1955U, 0x8813c027U, 0xa26819f9U, 0x4c3b57bdU, 0x7df94487U, 0xb975e769U, 0xb8f20cb5U, 0x5c9e2e77U,
    0x5fb1735fU, 0x3a6f759bU, 0x3c090923U, 0xfced424dU, 0xa187a6a9U, 0x6f070a41U, 0x2c85233bU, 0x7e62258bU,
    0x2771ef17U, 0x13bbf093U, 0x4ff059e5U, 0xe3ce3d0fU, 0xf1b4789fU, 0x9fbb6173U, 0x6a320cf5U, 0x1be2c481U,
    0x7ba8222bU, 0x6fd619b3U, 0x7b1bbf0dU, 0x8b8993adU, 0x448eca95U, 0x82ab09d9U, 0x2ce53909U, 0x4f548685U};

constexpr std::uint64_t Rotl64(std::uint64_t value, std::uint32_t shift)
{
    return (value << shift) | (value >> (64 - shift));
}

constexpr std::uint64_t Finalize(std::uint64_t hash)
{
    hash ^= hash >> 33;
    hash *= prime2;
    hash ^= hash >> 29;
    hash *= prime3;
    hash ^= hash >> 32;
    return hash;
}

inline std::uint32_t LoadLittleEndian32(std::uint8_t const* bytes)
{
    return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8) |
           (static_cast<std::uint32_t>(bytes[2]) << 16) | (static_cast<std::uint32_t>(bytes[3]) << 24);
}

inline std::uint64_t LoadLittleEndian64(std::uint8_t const* bytes)
{
    return static_cast<std::uint64_t>(LoadLittleEndian32(bytes)) |
           (static_cast<std::uint64_t>(LoadLittleEndian32(bytes + 4)) << 32);
}

template <typename Key>
std::uint64_t XXHash64(Key const& key, std::uint64_t seed = 0)
{
    static_assert(sizeof(Key) == 4 || sizeof(Key) == 8, "BloomFilter golden supports 4-byte and 8-byte keys");
    auto const bytes = ObjectRepresentationCast<std::array<std::uint8_t, sizeof(Key)>>(key);
    std::uint64_t hash = seed + prime5 + sizeof(Key);
    if constexpr (sizeof(Key) == 8) {
        std::uint64_t mixed = LoadLittleEndian64(bytes.data()) * prime2;
        mixed = Rotl64(mixed, 31) * prime1;
        hash ^= mixed;
        hash = Rotl64(hash, 27) * prime1 + prime4;
    } else {
        hash ^= static_cast<std::uint64_t>(LoadLittleEndian32(bytes.data())) * prime1;
        hash = Rotl64(hash, 23) * prime2 + prime3;
    }
    return Finalize(hash);
}

inline std::uint32_t BlockIndex(std::uint32_t upperHash, std::size_t numBlocks)
{
    return static_cast<std::uint32_t>((static_cast<std::uint64_t>(upperHash) * static_cast<std::uint32_t>(numBlocks)) >>
                                      32);
}

template <std::uint32_t WordsPerBlock = 8, std::uint32_t PatternBits = WordsPerBlock>
inline std::uint32_t WordPattern(std::uint32_t lowerHash, std::uint32_t word)
{
    static_assert(WordsPerBlock != 0, "WordsPerBlock must be non-zero");
    static_assert(PatternBits >= WordsPerBlock, "PatternBits must provide at least one bit per word");
    static_assert(PatternBits <= 64, "PatternBits exceeds the golden salt table");
    static_assert(PatternBits % WordsPerBlock == 0, "PatternBits must be evenly distributed across all words");

    constexpr std::uint32_t bitsPerWord = PatternBits / WordsPerBlock;
    std::uint32_t pattern = 0;
    for (std::uint32_t i = 0; i < bitsPerWord; ++i) {
        std::uint32_t const saltIndex = word * bitsPerWord + i;
        std::uint32_t const mixed = salts[saltIndex] * lowerHash;
        pattern |= std::uint32_t{1} << (mixed >> 27);
    }
    return pattern;
}

} // namespace bloom_golden_detail

template <class Key, std::uint32_t WordsPerBlock = 8, std::uint32_t PatternBits = WordsPerBlock>
void BloomGoldenAdd(std::vector<std::uint32_t>& words, std::size_t numBlocks, Key const& key, std::uint64_t seed = 0)
{
    std::uint64_t const hash = bloom_golden_detail::XXHash64(key, seed);
    std::uint32_t const block = bloom_golden_detail::BlockIndex(static_cast<std::uint32_t>(hash >> 32), numBlocks);
    std::uint32_t const lower = static_cast<std::uint32_t>(hash);
    for (std::uint32_t word = 0; word < WordsPerBlock; ++word) {
        words[static_cast<std::size_t>(block) * WordsPerBlock +
              word] |= bloom_golden_detail::WordPattern<WordsPerBlock, PatternBits>(lower, word);
    }
}

template <class Key, std::uint32_t WordsPerBlock = 8, std::uint32_t PatternBits = WordsPerBlock>
bool BloomGoldenContains(std::vector<std::uint32_t> const& words, std::size_t numBlocks, Key const& key,
                         std::uint64_t seed = 0)
{
    std::uint64_t const hash = bloom_golden_detail::XXHash64(key, seed);
    std::uint32_t const block = bloom_golden_detail::BlockIndex(static_cast<std::uint32_t>(hash >> 32), numBlocks);
    std::uint32_t const lower = static_cast<std::uint32_t>(hash);
    for (std::uint32_t word = 0; word < WordsPerBlock; ++word) {
        std::uint32_t const pattern = bloom_golden_detail::WordPattern<WordsPerBlock, PatternBits>(lower, word);
        if ((words[static_cast<std::size_t>(block) * WordsPerBlock + word] & pattern) != pattern) {
            return false;
        }
    }
    return true;
}

template <class Key, std::uint32_t WordsPerBlock = 8, std::uint32_t PatternBits = WordsPerBlock>
std::vector<std::uint32_t> MakeBloomGolden(std::size_t numBlocks, std::vector<Key> const& keys, std::uint64_t seed = 0)
{
    std::vector<std::uint32_t> words(numBlocks * WordsPerBlock, 0);
    for (auto const& key : keys) {
        BloomGoldenAdd<Key, WordsPerBlock, PatternBits>(words, numBlocks, key, seed);
    }
    return words;
}

} // namespace aclco::test
