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
#include "tests/common/golden.h"
#include "tests/common/test_print.h"

namespace {
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
}

TEMPLATE_TEST_CASE_SIG("dynamic_map reserve correctness", "[dynamic_map][reserve]",
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
    auto seed = GENERATE(1u);
    std::size_t n = (sizeof(Key) <= 2) ? 60000u : 100000u;

    std::string params = "seed=" + std::to_string(seed) + ", reserve_n=" + std::to_string(n);
    PRINT_BEFORE_EXEC_WITH_PROBE("reserve correctness", Key, Value, BS, 1024, n, params, Probe);

    SECTION("reserve(n) then insert n → correct")
    {
        PRINT_SECTION("reserve(n) then insert n → correct");
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(1024, sent, stream);
        map.Reserve(static_cast<typename decltype(map)::SizeType>(n), stream);
        aclco::test::Sync(stream);

        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
        if (hostPairs.empty()) {
            SKIP("Can not create enough pairs for Key type");
        }
        std::size_t m = hostPairs.size();
        std::vector<Key> keys(m);
        for (std::size_t i = 0; i < m; ++i)
            keys[i] = hostPairs[i].first;
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(m);
        dPairs.CopyFromHostAsync(hostPairs.data(), m, stream);
        aclco::test::Sync(stream);
        auto ins = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(m), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(ins == static_cast<decltype(ins)>(m));

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < m; ++i)
            if (observed[i] != golden[keys[i]])
                ++mismatch;
        INFO("mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }
}

TEST_CASE("dynamic_map zero initial capacity grows for reserve and insert", "[dynamic_map][zero_capacity]")
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    using Key = uint32_t;
    using Value = uint32_t;
    using Probe = aclco::test::dmap_factory::LinearProbing<Key>;
    constexpr int BS = 1;

    auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();
    auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(0, sent, stream);
    map.Reserve(1, stream);
    aclco::test::Sync(stream);
    REQUIRE(map.Capacity() > 0);

    std::vector<Pair<Key, Value>> hostPairs = {{1, 2}};
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(hostPairs.size());
    dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);
    aclco::test::Sync(stream);
    auto inserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(hostPairs.size()), stream);
    aclco::test::Sync(stream);
    REQUIRE(inserted == 1);
    REQUIRE(map.Size() == 1);
}
