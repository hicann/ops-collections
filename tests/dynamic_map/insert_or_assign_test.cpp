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

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
}

TEMPLATE_TEST_CASE_SIG("dynamic_map insert_or_assign correctness", "[dynamic_map][insert_or_assign]",
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
    auto seed = GENERATE(1u);
    std::size_t n = (sizeof(Key) <= 2) ? 60000u : 100000u;

    std::string params = "seed=" + std::to_string(seed) + ", n=" + std::to_string(n);
    PRINT_BEFORE_EXEC_WITH_PROBE("insert_or_assign correctness", Key, Value, BS, n, n, params, Probe);

    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty())
    {
        SKIP("Can not create enough pairs for Key type");
    }
    n = hostPairs.size();
    std::vector<Key> keys(n);
    for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;

    SECTION("insert_or_assign new keys (auto-grow)")
    {
        PRINT_SECTION("insert_or_assign new keys (auto-grow)");
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(1024, sent, stream);
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);
        auto ins = map.InsertOrAssign(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(ins == static_cast<decltype(ins)>(n));
        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("insert_or_assign updates existing key value (single submap)")
    {
        PRINT_SECTION("insert_or_assign updates existing key value (single submap)");
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(n * 4, sent, stream);
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);
        map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(map.NumSubmaps() == 1);
        std::vector<Pair<Key, Value>> upd(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            upd[i].first = hostPairs[i].first;
            upd[i].second = hostPairs[(i + 1) % n].second;
        }
        aclco::test::DeviceBuffer<Pair<Key, Value>> dUpd(n);
        dUpd.CopyFromHostAsync(upd.data(), n, stream);
        aclco::test::Sync(stream);
        map.InsertOrAssign(static_cast<void*>(dUpd.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != upd[i].second) ++mismatch;
        INFO("updated value mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("insert_or_assign updates old-submap keys after growth")
    {
        PRINT_SECTION("insert_or_assign updates old-submap keys after growth");
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(1024, sent, stream);
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);
        map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(map.NumSubmaps() > 1);
        auto sizeBefore = map.Size();

        std::vector<Pair<Key, Value>> upd(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            upd[i].first = hostPairs[i].first;
            upd[i].second = hostPairs[(i + 1) % n].second;
        }
        aclco::test::DeviceBuffer<Pair<Key, Value>> dUpd(n);
        dUpd.CopyFromHostAsync(upd.data(), n, stream);
        aclco::test::Sync(stream);
        auto ins = map.InsertOrAssign(static_cast<void*>(dUpd.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(ins == 0);
        REQUIRE_PRINT(map.Size() == sizeBefore);

        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != upd[i].second) ++mismatch;
        INFO("cross-submap updated value mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }
}
