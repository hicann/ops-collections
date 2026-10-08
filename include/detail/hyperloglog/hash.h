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
#include "macros.h"
#include "simt_api/device_functions.h"
#include <cstdint>
#include <type_traits>

namespace aclco::detail::hyperloglog {
struct HashWord {
    std::uint32_t lo;
    std::uint32_t hi;
};
constexpr std::uint64_t xxhashPrime3 = 1609587929392839161ULL;
template <std::uint64_t C>
COLLECTION_SIMT_DEVICE HashWord Multiply(HashWord x)
{
    constexpr auto low = static_cast<std::uint32_t>(C);
    constexpr auto high = static_cast<std::uint32_t>(C >> 32);
    return {x.lo * low, __umulhi(x.lo, low) + x.hi * low + x.lo * high};
}
template <unsigned N>
COLLECTION_SIMT_DEVICE HashWord Rotate(HashWord x)
{
    static_assert(N > 0 && N < 32);
    return {(x.lo << N) | (x.hi >> (32 - N)), (x.hi << N) | (x.lo >> (32 - N))};
}
template <std::uint64_t C>
COLLECTION_SIMT_DEVICE HashWord AddConstant(HashWord x)
{
    auto const lo = x.lo + static_cast<std::uint32_t>(C);
    return {lo, x.hi + static_cast<std::uint32_t>(C >> 32) + static_cast<unsigned>(lo < x.lo)};
}
template <class Key>
COLLECTION_SIMT_DEVICE HashWord HashKeyBeforeFinalMultiply(Key key)
{
    static_assert(std::is_same_v<Key, std::int32_t> || std::is_same_v<Key, std::int64_t>);
    // Exact xxhash64 arithmetic, split into native 32-bit products and carry words.
    constexpr std::uint64_t p1 = 11400714785074694791ULL;
    constexpr std::uint64_t p2 = 14029467366897019727ULL;
    constexpr std::uint64_t p4 = 9650029242287828579ULL;
    constexpr std::uint64_t p5 = 2870177450012600261ULL;
    HashWord x;
    if constexpr (sizeof(Key) == 4) {
        x = Multiply<p1>({static_cast<std::uint32_t>(key), 0});
        x.lo ^= static_cast<std::uint32_t>(p5 + 4);
        x.hi ^= static_cast<std::uint32_t>((p5 + 4) >> 32);
        x = AddConstant<xxhashPrime3>(Multiply<p2>(Rotate<23>(x)));
    } else {
        auto const raw = static_cast<std::uint64_t>(key);
        x = Multiply<p1>(
            Rotate<31>(Multiply<p2>({static_cast<std::uint32_t>(raw), static_cast<std::uint32_t>(raw >> 32)})));
        x.lo ^= static_cast<std::uint32_t>(p5 + 8);
        x.hi ^= static_cast<std::uint32_t>((p5 + 8) >> 32);
        x = AddConstant<p4>(Multiply<p1>(Rotate<27>(x)));
    }
    x.lo ^= x.hi >> 1;
    x = Multiply<p2>(x);
    x.lo ^= (x.lo >> 29) | (x.hi << 3);
    x.hi ^= x.hi >> 29;
    return x;
}
COLLECTION_SIMT_DEVICE std::uint32_t FinalHashHigh(HashWord x)
{
    constexpr auto low = static_cast<std::uint32_t>(xxhashPrime3);
    constexpr auto high = static_cast<std::uint32_t>(xxhashPrime3 >> 32);
    return __umulhi(x.lo, low) + x.hi * low + x.lo * high;
}
COLLECTION_SIMT_DEVICE std::uint32_t FinalHashLow(HashWord x, std::uint32_t finalHigh)
{
    return x.lo * static_cast<std::uint32_t>(xxhashPrime3) ^ finalHigh;
}
template <class Key>
COLLECTION_SIMT_DEVICE HashWord HashKeyWords(Key key)
{
    auto const state = HashKeyBeforeFinalMultiply(key);
    auto const high = FinalHashHigh(state);
    return {FinalHashLow(state, high), high};
}
template <class Key>
COLLECTION_SIMT_DEVICE std::uint64_t HashKey(Key key)
{
    auto const x = HashKeyWords(key);
    return (static_cast<std::uint64_t>(x.hi) << 32) | x.lo;
}
} // namespace aclco::detail::hyperloglog
