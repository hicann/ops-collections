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
#include <cstring>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

#include "pair.h"
#include "detail/open_addressing/kernels.h"
#include "utility/kernel_launch_utils.h"

extern "C" COLLECTION_GLOBAL void CreatePair(__gm__ uint64_t* pairAddr)
{
    aclco::Pair<int8_t, uint8_t> devicePair = aclco::MakePair<int8_t, uint8_t>((int8_t)-48, (uint8_t)48);
    void* addr = (void*)&devicePair;
    AscendC::WriteGmByPassDCache<uint64_t>(pairAddr, *((uint64_t*)addr));
    AscendC::WriteGmByPassDCache<uint64_t>((pairAddr + 1), *((uint64_t*)addr + 1));
}

struct EmptyType
{
    //定义一个空结构体
};

TEST_CASE("Pair construction and equality (host)", "[utility][pair][host]")
{
  using First  = char;
  using Second = unsigned char;

  aclco::Pair<First, Second> p1 = aclco::MakePair<First, Second>(static_cast<First>(-48),
                                                          static_cast<Second>(48));
  aclco::Pair<First, Second> p2 = aclco::MakePair<First, Second>(static_cast<First>(-48),
                                                          static_cast<Second>(48));

  REQUIRE(static_cast<int>(p1.first) == -48);
  REQUIRE(static_cast<int>(p1.second) == 48);
  REQUIRE(p1 == p2);
}

TEST_CASE("Pair construction on device and D2H copy (device)", "[utility][pair][device]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using First  = char;
  using Second = unsigned char;
  using P      = aclco::Pair<First, Second>;

  const int expectedFirst  = -48;
  const int expectedSecond = 48;

  constexpr std::size_t u64s = 2;
  aclco::test::DeviceBuffer<uint64_t> dOut(u64s);
  dOut.MemsetZero(stream);

  CreatePair<<<1, nullptr, stream>>>(dOut.Data());
  aclco::test::Sync(stream);

  auto host_u64 = dOut.CopyToHost(stream);
  REQUIRE(host_u64.size() == 2);

  P hostPair{};
  std::memcpy(&hostPair, host_u64.data(), sizeof(P));

  REQUIRE(static_cast<int>(hostPair.first) == expectedFirst);
  REQUIRE(static_cast<int>(hostPair.second) == expectedSecond);
}

TEST_CASE("PairAlignment returns expected alignment", "[utility][pair][alignment]")
{
  // char + unsigned char
  {
    using T1 = char;
    using T2 = unsigned char;
    constexpr std::size_t expected = 2;
    const std::size_t actual = aclco::PairAlignment<T1, T2>();
    CAPTURE(expected, actual);
    REQUIRE(actual == expected);
  }

  // int8_t + int16_t
  {
    using T3 = int8_t;
    using T4 = int16_t;
    constexpr std::size_t expected = 4;
    const std::size_t actual = aclco::PairAlignment<T3, T4>();
    CAPTURE(expected, actual);
    REQUIRE(actual == expected);
  }

  // double + int32_t
  {
    using T5 = double;
    using T6 = int32_t;
    constexpr std::size_t expected = 16;
    const std::size_t actual = aclco::PairAlignment<T5, T6>();
    CAPTURE(expected, actual);
    REQUIRE(actual == expected);
  }

  // EmptyType + EmptyType
  {
    constexpr std::size_t expected = 2;
    const std::size_t actual = aclco::PairAlignment<EmptyType, EmptyType>();
    CAPTURE(expected, actual);
    REQUIRE(actual == expected);
  }
}