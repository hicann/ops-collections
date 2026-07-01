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
#include <unordered_set>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/dmap_factory.h"
#include "tests/common/generators.h"
#include "tests/common/golden.h"
#include "tests/common/test_print.h"

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
}

TEMPLATE_TEST_CASE_SIG("dynamic_map contains correctness", "[dynamic_map][contains]",
                       ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize,
                        ProbingScheme),
                       (uint32_t, uint32_t, 1, aclco::test::dmap_factory::LinearProbing<uint32_t>),
                       (uint32_t, uint32_t, 5, aclco::test::dmap_factory::DoubleHashing<uint32_t>),
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
    auto seed = GENERATE(1u);
    std::size_t n = (sizeof(Key) <= 2) ? 60000u : 100000u;

    std::string params = "seed=" + std::to_string(seed) + ", n=" + std::to_string(n);
    PRINT_BEFORE_EXEC_WITH_PROBE("contains correctness", Key, Value, BS, initCap, n, params, Probe);

    auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty())
    {
        SKIP("Can not create enough pairs for Key type");
    }
    n = hostPairs.size();
    std::vector<Key> keys(n);
    for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
    aclco::test::Sync(stream);
    map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
    aclco::test::Sync(stream);

    SECTION("contains existing keys → all true")
    {
        PRINT_SECTION("contains existing keys → all true");
        auto flags = aclco::test::dmap_factory::RetrieveViaContains<Key>(map, keys, stream);
        std::size_t missing = 0;
        for (auto f : flags)
            if (f == 0) ++missing;
        INFO("missing = " << missing);
        REQUIRE_PRINT(missing == 0);
    }

    SECTION("contains mixed present + absent keys")
    {
        PRINT_SECTION("contains mixed present + absent keys");
        auto other = aclco::test::MakeExamples<Key, Value>(seed + 777u, n, sent, "uniform", true);
        std::unordered_set<Key> present(keys.begin(), keys.end());
        std::vector<Key> q;
        std::vector<char> expect;
        for (std::size_t i = 0; i < n; ++i)
        {
            q.push_back(keys[i]);
            expect.push_back(1);
        }
        for (auto const& p : other)
            if (!present.count(p.first))
            {
                q.push_back(p.first);
                expect.push_back(0);
            }
        auto flags = aclco::test::dmap_factory::RetrieveViaContains<Key>(map, q, stream);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < q.size(); ++i)
            if ((flags[i] != 0) != (expect[i] != 0)) ++wrong;
        INFO("wrong判定 = " << wrong);
        REQUIRE_PRINT(wrong == 0);
    }
}
