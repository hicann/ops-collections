/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "detail/hash_functions/murmurhash3.h"
#include "detail/hash_functions/xxhash.h"

TEST_CASE("murmurhash3 functors produce expected values", "[hash]")
{
  constexpr std::uint32_t key  = 0x41424344u;   // 'ABCD' as a 32-bit integer
  constexpr std::uint32_t seed = 0xffffeeeeu;

  aclco::detail::MurmurHash3Fmix32<std::uint32_t> fmix32{seed};
  REQUIRE(fmix32(key) == 0x570df661u);

  // In the legacy test, the 32-bit key was cast to int64_t and hashed with Fmix64.
  aclco::detail::MurmurHash3Fmix64<std::uint64_t> fmix64{static_cast<std::uint64_t>(seed)};
  REQUIRE(fmix64(static_cast<std::uint64_t>(key)) == 0x5ee45a8eb612f708ULL);

  aclco::detail::MurmurHash3_32<std::uint32_t> murmur32{seed};
  REQUIRE(murmur32(key) == 0xd6bdbbb4u);
}

TEST_CASE("xxhash functors produce expected values", "[hash]")
{
  constexpr std::uint32_t key1  = 0x75bcd15u;   // 123456789 as a 32-bit integer
  constexpr std::uint32_t seed1 = 0x0u;

  aclco::detail::XXHash_32<std::uint32_t> xxhash1{seed1};
  REQUIRE(xxhash1(key1) == 0xb20a85eeu);    //2987034094

  constexpr std::uint32_t key2  = 0x2au;   // 42 as a 32-bit integer
  constexpr std::uint32_t seed2 = 0x0u;

  aclco::detail::XXHash_32<std::uint32_t> xxhash2{seed2};
  REQUIRE(xxhash2(key2) == 0x454235d1);    // 1161967057
}