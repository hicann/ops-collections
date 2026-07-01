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
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/dmap_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
}

TEMPLATE_TEST_CASE_SIG("dynamic_map destroy correctness", "[dynamic_map][destroy]",
                       ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize,
                        ProbingScheme),
                       (uint32_t, uint32_t, 1, aclco::test::dmap_factory::LinearProbing<uint32_t>),
                       (uint64_t, uint64_t, 5, aclco::test::dmap_factory::LinearProbing<uint64_t>),
                       (uint16_t, uint16_t, 5, aclco::test::dmap_factory::LinearProbing<uint16_t>),
                       (float, float, 5, aclco::test::dmap_factory::LinearProbing<float>))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    using Probe = ProbingScheme;
    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();
    std::size_t initCap = 1024;

    PRINT_BEFORE_EXEC_WITH_PROBE("destroy correctness", Key, Value, BS, initCap, 0, "", Probe);

    SECTION("construct + destruct (empty), no crash/leak")
    {
        PRINT_SECTION("construct + destruct (empty), no crash/leak");
        {
            auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);
            aclco::test::Sync(stream);
        }
        aclco::test::Sync(stream);
        REQUIRE_PRINT(true);
    }

    SECTION("destruct after growth (multi-submap), no crash/leak")
    {
        PRINT_SECTION("destruct after growth (multi-submap), no crash/leak");
        std::size_t n = (sizeof(Key) <= 2) ? 60000u : 100000u;
        auto hostPairs = aclco::test::MakeExamples<Key, Value>(1u, n, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs for Key type");
        }
        n = hostPairs.size();
        {
            auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);
            aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
            dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
            aclco::test::Sync(stream);
            map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
            aclco::test::Sync(stream);
            REQUIRE_PRINT(map.NumSubmaps() > 1);
        }
        aclco::test::Sync(stream);
        REQUIRE_PRINT(true);
    }
}
