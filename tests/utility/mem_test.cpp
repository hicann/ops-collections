/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <acl/acl.h>
#include <catch2/catch_test_macros.hpp>

#include "detail/open_addressing/kernels.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

struct TestBlock {
    // Use fixed-width types to avoid host/device signed-char differences.
    uint8_t first;
    uint8_t second;
};

struct TestFBlock {
    uint8_t GetFirst() const { return this->first; }
    uint8_t GetSecond() const { return this->second; }
    uint8_t first;
    uint8_t second;
};

// Mark kernels as AIV so Bisheng can emit executable vector-core kernels.
extern "C" COLLECTION_AIV_GLOBAL void CreateTestBlock(__gm__ uint16_t* blockAddr)
{
    TestBlock blk;
    blk.first = 49;
    blk.second = 50;
    void* addr = (void*)&blk;
    AscendC::WriteGmByPassDCache<uint16_t>((__gm__ uint16_t*)blockAddr, *((uint16_t*)addr));
}

extern "C" COLLECTION_AIV_GLOBAL void CreateTestFBlock(__gm__ uint16_t* blockAddr)
{
    TestFBlock blk;
    blk.first = 49;
    blk.second = 50;
    void* addr = (void*)&blk;
    AscendC::WriteGmByPassDCache<uint16_t>((__gm__ uint16_t*)blockAddr, *((uint16_t*)addr));
}

template <typename Block>
Block ReadDeviceBlock()
{
    static_assert(std::is_same_v<Block, TestBlock> || std::is_same_v<Block, TestFBlock>);
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    constexpr std::size_t words = (sizeof(Block) + sizeof(uint16_t) - 1) / sizeof(uint16_t);
    aclco::test::DeviceBuffer<uint16_t> deviceWords(words);
    deviceWords.MemsetZero(stream);
    if constexpr (std::is_same_v<Block, TestBlock>) {
        CreateTestBlock<<<1, nullptr, stream>>>(deviceWords.Data());
    } else {
        CreateTestFBlock<<<1, nullptr, stream>>>(deviceWords.Data());
    }
    aclco::test::Sync(stream);

    auto const hostWords = deviceWords.CopyToHost(stream);
    REQUIRE(hostWords.size() == words);
    Block block{};
    auto* destination = reinterpret_cast<unsigned char*>(&block);
    auto const* source = reinterpret_cast<unsigned char const*>(hostWords.data());
    std::copy_n(source, sizeof(Block), destination);
    return block;
}

TEST_CASE("Device memory layout: POD struct TestBlock (device->host)", "[utility][mem][layout]")
{
    const uint8_t expectedFirst = static_cast<uint8_t>('1');
    const uint8_t expectedSecond = static_cast<uint8_t>('2');
    TestBlock const hostBlk = ReadDeviceBlock<TestBlock>();

    REQUIRE(static_cast<int>(hostBlk.first) == static_cast<int>(expectedFirst));
    REQUIRE(static_cast<int>(hostBlk.second) == static_cast<int>(expectedSecond));
}

TEST_CASE("Device memory layout: struct with methods TestFBlock (device->host)", "[utility][mem][layout]")
{
    const uint8_t expectedFirst = static_cast<uint8_t>('1');
    const uint8_t expectedSecond = static_cast<uint8_t>('2');
    TestFBlock const hostBlk = ReadDeviceBlock<TestFBlock>();

    REQUIRE(static_cast<int>(hostBlk.first) == static_cast<int>(expectedFirst));
    REQUIRE(static_cast<int>(hostBlk.second) == static_cast<int>(expectedSecond));
    REQUIRE(static_cast<int>(hostBlk.GetFirst()) == static_cast<int>(expectedFirst));
    REQUIRE(static_cast<int>(hostBlk.GetSecond()) == static_cast<int>(expectedSecond));
}
