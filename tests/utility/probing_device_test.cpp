/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include "tiling/platform/platform_ascendc.h"
#include "probing_scheme.h"
#include "hash_functions.h"
#include "detail/open_addressing/kernels.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco {

struct TestHash1 {
    COLLECTION_HOST_DEVICE constexpr uint32_t operator()(uint32_t key) const { return key * 16777619; }
};

struct TestHash2 {
    COLLECTION_HOST_DEVICE constexpr uint32_t operator()(uint32_t key) const { return key * 2654435761ULL; }
};

// Mark the kernel as AIV so Bisheng can emit an executable vector-core kernel.
extern "C" COLLECTION_AIV_GLOBAL void TestDoubleHashingIterator(__gm__ uint32_t* result, __gm__ uint32_t* probeKey,
                                                                __gm__ uint32_t* upperBound)
{
    aclco::DoubleHashing<TestHash1, TestHash2> probing(TestHash1{}, TestHash2{});
    uint32_t key = *probeKey;
    uint32_t upper = *upperBound;
    auto it = probing.template MakeIterator<4>(key, upper);

    uint32_t pos1 = *it;
    ++it;
    uint32_t pos2 = *it;

    *result = pos1 << 16 | pos2;
}

TEST_CASE("DoubleHashing Test")
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;

    SECTION("DoubleHashing Iterator Test")
    {
        uint32_t probeKey = 12345;
        uint32_t upperBound = 1024;

        aclco::test::DeviceBuffer<uint32_t> deviceResult(1);
        aclco::test::DeviceBuffer<uint32_t> deviceKey(1);
        aclco::test::DeviceBuffer<uint32_t> deviceUpper(1);

        deviceKey.MemsetZero(stream);
        deviceResult.MemsetZero(stream);
        deviceUpper.MemsetZero(stream);

        deviceKey.CopyFromHostAsync(&probeKey, 1, stream);
        deviceUpper.CopyFromHostAsync(&upperBound, 1, stream);

        auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();

        TestDoubleHashingIterator<<<aivCoreNum, 256, stream>>>(deviceResult.Data(), deviceKey.Data(),
                                                               deviceUpper.Data());

        aclco::test::Sync(stream);
        auto hostResult = deviceResult.CopyToHost(stream);

        REQUIRE(hostResult.size() == 1);
        uint32_t combined = hostResult[0];

        uint32_t pos1 = (combined >> 16) & 0xFFFF;
        uint32_t pos2 = combined & 0xFFFF;

        REQUIRE(pos1 < upperBound);
        REQUIRE(pos2 < upperBound);
        REQUIRE(pos1 != pos2);

        uint32_t hash1Val = TestHash1{}(probeKey);
        uint32_t hash2Val = TestHash2{}(probeKey);

        uint32_t bucketSize = 4;
        uint32_t groups = upperBound / bucketSize;
        uint32_t step = (hash2Val % (groups - 1) + 1) * bucketSize;
        uint32_t expected1 = hash1Val % groups * bucketSize;
        uint32_t expected2 = (expected1 + step) % upperBound;
        REQUIRE(pos1 == expected1);
        REQUIRE(pos2 == expected2);
    }

    SECTION("DoubleHashing RebindHash Test")
    {
        aclco::DoubleHashing<TestHash1, TestHash2> original(TestHash1{}, TestHash2{});

        auto newHash = std::make_tuple(TestHash1{}, TestHash2{});
        auto rebound = original.RebindHash(newHash);

        REQUIRE((std::is_same_v<decltype(rebound), aclco::DoubleHashing<TestHash1, TestHash2>>));
    }
}

} // namespace aclco
