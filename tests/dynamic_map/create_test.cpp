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
#include "tests/common/dmap_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE_SIG("dynamic_map create correctness", "[dynamic_map][create]",
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
    std::size_t initCap = GENERATE(1024u, 40000000u);

    std::string params = "initCap=" + std::to_string(initCap);
    PRINT_BEFORE_EXEC_WITH_PROBE("create correctness", Key, Value, BS, initCap, 0, params, Probe);

    SECTION("construct empty dynamic_map")
    {
        PRINT_SECTION("construct empty dynamic_map");
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(map.Size() == 0);
        REQUIRE_PRINT(map.NumSubmaps() == 1);
        REQUIRE_PRINT(map.Capacity() >= static_cast<decltype(map.Capacity())>(initCap));
        REQUIRE_PRINT(map.Capacity() < static_cast<decltype(map.Capacity())>(initCap) * 2);
    }

    SECTION("emptyKey == erasedKey must throw")
    {
        PRINT_SECTION("emptyKey == erasedKey must throw");
        auto badSent = sent;
        badSent.erasedKey = badSent.emptyKey;
        REQUIRE_THROWS_AS((aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, badSent, stream)),
                          std::invalid_argument);
    }
}
