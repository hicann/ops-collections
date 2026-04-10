/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>

#include <acl/acl.h>
#include <catch2/catch_test_macros.hpp>

#include "detail/open_addressing/kernels.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

struct TestBlock
{
  // Use fixed-width types to avoid host/device signed-char differences.
  uint8_t first;
  uint8_t second;
};

struct TestFBlock
{
  uint8_t GetFirst() const
  {
    return this->first;
  }
  uint8_t GetSecond() const
  {
    return this->second;
  }
  uint8_t first;
  uint8_t second;
};

// NOTE: Mark kernels as aiv to ensure they are actually emitted as exeaclcotable aicore kernels.
extern "C" __global__ __aicore__ void CreateTestBlock(__gm__ uint16_t* blockAddr)
{
  TestBlock blk;
  blk.first  = 49;
  blk.second = 50;
  void* addr = (void*)&blk;
  AscendC::WriteGmByPassDCache<uint16_t>((__gm__ uint16_t*)blockAddr, *((uint16_t*)addr));
}

extern "C" __global__ __aicore__ void CreateTestFBlock(__gm__ uint16_t* blockAddr)
{
  TestFBlock blk;
  blk.first  = 49;
  blk.second = 50;
  void* addr = (void*)&blk;
  AscendC::WriteGmByPassDCache<uint16_t>((__gm__ uint16_t*)blockAddr, *((uint16_t*)addr));
}

TEST_CASE("Device memory layout: POD struct TestBlock (device->host)", "[utility][mem][layout]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  const uint8_t expectedFirst  = static_cast<uint8_t>('1');
  const uint8_t expectedSecond = static_cast<uint8_t>('2');

  constexpr size_t bytes = sizeof(TestFBlock);
  constexpr size_t words = (bytes + sizeof(uint16_t) - 1) / sizeof(uint16_t);
  aclco::test::DeviceBuffer<uint16_t> dWords(words);
  dWords.MemsetZero(stream);

  CreateTestBlock<<<1, nullptr, stream>>>(dWords.Data());
  aclco::test::Sync(stream);

  auto hostWords = dWords.CopyToHost(stream);
  REQUIRE(hostWords.size() == 1);

  TestBlock hostBlk{};
  std::memcpy(&hostBlk, hostWords.data(), sizeof(TestBlock));

  REQUIRE(static_cast<int>(hostBlk.first) == static_cast<int>(expectedFirst));
  REQUIRE(static_cast<int>(hostBlk.second) == static_cast<int>(expectedSecond));
}

TEST_CASE("Device memory layout: struct with methods TestFBlock (device->host)", "[utility][mem][layout]")
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  const uint8_t expectedFirst  = static_cast<uint8_t>('1');
  const uint8_t expectedSecond = static_cast<uint8_t>('2');

  constexpr size_t bytes = sizeof(TestFBlock);
  constexpr size_t words = (bytes + sizeof(uint16_t) - 1) / sizeof(uint16_t);
  aclco::test::DeviceBuffer<uint16_t> dWords(words);
  dWords.MemsetZero(stream);

  CreateTestFBlock<<<1, nullptr, stream>>>(dWords.Data());
  aclco::test::Sync(stream);

  auto hostWords = dWords.CopyToHost(stream);
  REQUIRE(hostWords.size() == words);

  TestFBlock hostBlk{};
  std::memcpy(&hostBlk, hostWords.data(), sizeof(TestFBlock));

  REQUIRE(static_cast<int>(hostBlk.first) == static_cast<int>(expectedFirst));
  REQUIRE(static_cast<int>(hostBlk.second) == static_cast<int>(expectedSecond));
  REQUIRE(static_cast<int>(hostBlk.GetFirst()) == static_cast<int>(expectedFirst));
  REQUIRE(static_cast<int>(hostBlk.GetSecond()) == static_cast<int>(expectedSecond));
}