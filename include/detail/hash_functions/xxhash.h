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

#include "detail/hash_functions/utils.h"
#include "extent.h"
#include "macros.h"

namespace aclco::detail {
template <typename Key>
struct XXHash_32 {
 private:
  static constexpr std::uint32_t prime1 = 0x9e3779b1u;
  static constexpr std::uint32_t prime2 = 0x85ebca77u;
  static constexpr std::uint32_t prime3 = 0xc2b2ae3du;
  static constexpr std::uint32_t prime4 = 0x27d4eb2fu;
  static constexpr std::uint32_t prime5 = 0x165667b1u;

 public:
  using ArgumentType = Key;            ///< The type of the values taken as argument
  using ResultType   = std::uint32_t;  ///< The type of the hash values produced

  /**
   * @brief Constructs a XXH32 hash function with the given `seed`.
   *
   * @param seed A custom number to randomize the resulting hash value
   */
  COLLECTION_HOST_DEVICE constexpr XXHash_32(std::uint32_t seed = 0) : seed_{seed} {}

  /**
   * @brief Returns a hash value for its argument, as a value of type `ResultType`.
   *
   * @param key The input argument to hash
   * @return The resulting hash value for `key`
   */
  constexpr ResultType COLLECTION_HOST_DEVICE operator()(Key const& key) const noexcept
  {
    if constexpr (sizeof(Key) <= 16) {
      Key const key_copy = key;
      return ComputeHash(reinterpret_cast<std::byte const*>(&key_copy),
                          aclco::Extent<std::size_t, sizeof(Key)>{});
    } else {
      return ComputeHash(reinterpret_cast<std::byte const*>(&key),
                          aclco::Extent<std::size_t, sizeof(Key)>{});
    }
  }

  /**
   * @brief Returns a hash value for its argument, as a value of type `ResultType`.
   *
   * @tparam Extent The Extent type
   *
   * @param bytes The input argument to hash
   * @param size The Extent of the data in bytes
   * @return The resulting hash value
   */
  template <typename Extent>
  constexpr ResultType COLLECTION_HOST_DEVICE ComputeHash(std::byte const* bytes,
                                                         Extent size) const noexcept
  {
    size_t offset = 0;
    std::uint32_t h32{};

    // data can be processed in 16-byte chunks
    if (size >= 16) {
      auto const limit = size - 16;
      std::uint32_t v1 = seed_ + prime1 + prime2;
      std::uint32_t v2 = seed_ + prime2;
      std::uint32_t v3 = seed_;
      std::uint32_t v4 = seed_ - prime1;

      do {
        // pipeline 4*4byte computations
        auto const pipeline_offset = offset / 4;
        v1 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 0, size) * prime2;
        v1 = Rotl32(v1, 13);
        v1 *= prime1;
        v2 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 1, size) * prime2;
        v2 = Rotl32(v2, 13);
        v2 *= prime1;
        v3 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 2, size) * prime2;
        v3 = Rotl32(v3, 13);
        v3 *= prime1;
        v4 += LoadChunk<std::uint32_t>(bytes, pipeline_offset + 3, size) * prime2;
        v4 = Rotl32(v4, 13);
        v4 *= prime1;
        offset += 16;
      } while (offset <= limit);

      h32 = Rotl32(v1, 1) + Rotl32(v2, 7) + Rotl32(v3, 12) + Rotl32(v4, 18);
    } else {
      h32 = seed_ + prime5;
    }

    h32 += static_cast<std::uint32_t>(size);

    // remaining data can be processed in 4-byte chunks
    if ((size % 16) >= 4) {
      for (; offset <= size - 4; offset += 4) {
        h32 += LoadChunk<std::uint32_t>(bytes, offset / 4, size) * prime3;
        h32 = Rotl32(h32, 17) * prime4;
      }
    }

    // the following loop is only needed if the size of the key is not a multiple of the block size
    if (size % 4) {
      while (offset < size) {
        h32 += (static_cast<std::uint32_t>(bytes[offset]) & 255) * prime5;
        h32 = Rotl32(h32, 11) * prime1;
        ++offset;
      }
    }

    return Finalize(h32);
  }


 private:
  // avalanche helper
  constexpr COLLECTION_HOST_DEVICE std::uint32_t Finalize(std::uint32_t h) const noexcept
  {
    h ^= h >> 15;
    h *= prime2;
    h ^= h >> 13;
    h *= prime3;
    h ^= h >> 16;
    return h;
  }

  std::uint32_t seed_;
};
}  // namespace aclco::detail
