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

#include <cassert>
#include <cstring>

#include "macros.h"

namespace aclco {
namespace detail {

template <typename T, typename U, typename Extent, typename SizeType>
COLLECTION_HOST_DEVICE constexpr T LoadChunk(U const* const data, Extent index, SizeType dataSize) noexcept
{
  assert((index + 1) * sizeof(T) <= static_cast<std::size_t>(dataSize));
  auto const bytes = reinterpret_cast<std::byte const*>(data);
  T chunk;
  memcpy(&chunk, bytes + index * sizeof(T), sizeof(T));
  return chunk;
}

COLLECTION_HOST_DEVICE constexpr std::uint32_t Rotl32(std::uint32_t x, std::int8_t r) noexcept
{
  return (x << r) | (x >> (32 - r));
}

template <typename Key>
COLLECTION_HOST_DEVICE constexpr std::uint32_t Fmix32(Key key, std::uint32_t seed = 0) noexcept
{
    static_assert(sizeof(Key) == 4, "Key type must be 4 bytes in size.");
    std::uint32_t h = static_cast<std::uint32_t>(key) ^ seed;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

template <typename Key>
COLLECTION_HOST_DEVICE constexpr std::uint64_t Fmix64(Key key, std::uint64_t seed = 0) noexcept
{
    static_assert(sizeof(Key) == 8, "Key type must be 8 bytes in size.");
    std::uint64_t h = static_cast<std::uint64_t>(key) ^ seed;
    h ^= h >> 33;
    h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33;
    h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
}
}