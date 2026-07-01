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
}  // namespace

TEMPLATE_TEST_CASE_SIG("dynamic_map insert correctness", "[dynamic_map][insert]",
                       ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize,
                        ProbingScheme),
                       (uint32_t, uint32_t, 1, aclco::test::dmap_factory::LinearProbing<uint32_t>),
                       (uint32_t, uint32_t, 5, aclco::test::dmap_factory::LinearProbing<uint32_t>),
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

    std::size_t initCap = GENERATE(1024u, 200000u);
    auto seed = GENERATE(1u);
    auto ratio = GENERATE(0.5f, 2.0f);
    auto n = static_cast<std::size_t>(ratio * initCap);
    if (sizeof(Key) <= 2 && n > 60000u) n = 60000u;

    std::string params =
        "initCap=" + std::to_string(initCap) + ", seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
    PRINT_BEFORE_EXEC_WITH_PROBE("insert correctness", Key, Value, BS, initCap, n, params, Probe);

    SECTION("normal insert unique pairs")
    {
        PRINT_SECTION("normal insert unique pairs");
        CAPTURE(seed, n, initCap, BS);
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);

        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
        }
        n = hostPairs.size();

        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);

        auto inserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(inserted == static_cast<decltype(inserted)>(n));
        REQUIRE_PRINT(map.Size() == static_cast<decltype(map.Size())>(n));

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        std::vector<Key> keys(n);
        for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("value mismatch count = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("insert triggering growth (multi-submap)")
    {
        PRINT_SECTION("insert triggering growth (multi-submap)");
        CAPTURE(seed, n, initCap, BS);
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(1024, sent, stream);

        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
        }
        n = hostPairs.size();
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);

        auto inserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(inserted == static_cast<decltype(inserted)>(n));
        REQUIRE_PRINT(map.NumSubmaps() > 1);

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        std::vector<Key> keys(n);
        for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("cross-submap Find mismatch count = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("re-insert old keys with new values after growth → 0 inserted, old values kept")
    {
        PRINT_SECTION("re-insert old keys with new values after growth");
        CAPTURE(seed, n, initCap, BS);
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(1024, sent, stream);

        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
        }
        n = hostPairs.size();
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);

        auto inserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(inserted == static_cast<decltype(inserted)>(n));
        REQUIRE_PRINT(map.NumSubmaps() > 1);

        auto newPairs = hostPairs;
        for (auto& p : newPairs) p.second = static_cast<Value>(p.second + 1);
        dPairs.CopyFromHostAsync(newPairs.data(), n, stream);
        aclco::test::Sync(stream);
        auto reInserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(reInserted == 0);
        REQUIRE_PRINT(map.Size() == static_cast<decltype(map.Size())>(n));

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        std::vector<Key> keys(n);
        for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("old-value mismatch count after re-insert = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("chunked insert when activeRoom < n (batch split across submaps)")
    {
        PRINT_SECTION("chunked insert when activeRoom < n");
        if (sizeof(Key) <= 2)
        {
            SKIP("u16 key domain too small for chunk-split scale");
        }
        std::size_t bigN = 300000;
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(200000, sent, stream);
        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, bigN, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs");
        }
        bigN = hostPairs.size();
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(bigN);
        dPairs.CopyFromHostAsync(hostPairs.data(), bigN, stream);
        aclco::test::Sync(stream);

        auto inserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(bigN), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(inserted == static_cast<decltype(inserted)>(bigN));
        REQUIRE_PRINT(map.Size() == static_cast<decltype(map.Size())>(bigN));
        REQUIRE_PRINT(map.NumSubmaps() > 1);

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        std::vector<Key> keys(bigN);
        for (std::size_t i = 0; i < bigN; ++i) keys[i] = hostPairs[i].first;
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < bigN; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("chunk-split Find mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("kAppendUnique multi-batch insert (optimistic fast path)")
    {
        PRINT_SECTION("kAppendUnique multi-batch insert");
        CAPTURE(seed, n, initCap, BS);
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(1024, sent, stream);

        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
        }
        n = hostPairs.size();
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);

        std::size_t batch = (n + 3) / 4;
        if (batch == 0) batch = n;
        std::size_t total = 0;
        for (std::size_t off = 0; off < n; off += batch)
        {
            std::size_t cur = (batch < n - off) ? batch : (n - off);
            total += map.Insert(static_cast<void*>(dPairs.Data() + off), aclco::Extent<std::size_t>(cur), stream,
                                decltype(map)::InsertMode::kAppendUnique);
        }
        aclco::test::Sync(stream);
        REQUIRE_PRINT(total == static_cast<decltype(total)>(n));
        REQUIRE_PRINT(map.Size() == static_cast<decltype(map.Size())>(n));

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        std::vector<Key> keys(n);
        for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("kAppendUnique Find mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }

    SECTION("InsertAsync unique pairs (optimistic, deferred count)")
    {
        PRINT_SECTION("InsertAsync unique pairs");
        CAPTURE(seed, n, initCap, BS);
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);

        auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
        }
        n = hostPairs.size();
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);

        map.InsertAsync(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(map.Size() == static_cast<decltype(map.Size())>(n));

        auto golden = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);
        std::vector<Key> keys(n);
        for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;
        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t mismatch = 0;
        for (std::size_t i = 0; i < n; ++i)
            if (observed[i] != golden[keys[i]]) ++mismatch;
        INFO("InsertAsync Find mismatch = " << mismatch);
        REQUIRE_PRINT(mismatch == 0);
    }
}

TEMPLATE_TEST_CASE_SIG("dynamic_map insert negative tests", "[dynamic_map][insert]",
                       ((typename K, typename V, int BucketSize), K, V, BucketSize), (uint32_t, uint32_t, 1),
                       (uint32_t, uint32_t, 5), (uint64_t, uint64_t, 5), (float, float, 1))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();
    std::size_t initCap = 100000;
    auto seed = GENERATE(1u);

    std::string params = "seed=" + std::to_string(seed);
    PRINT_BEFORE_EXEC_WITH_PARAMS("insert negative tests", Key, Value, BS, initCap, 0, params);

    SECTION("test n == 0")
    {
        PRINT_SECTION("test n == 0");
        auto map =
            aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, aclco::test::dmap_factory::LinearProbing<Key>>(
                initCap, sent, stream);
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(1);
        auto inserted = map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(0), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(inserted == 0);
        REQUIRE_PRINT(map.Size() == 0);
    }

    SECTION("insert duplicate keys")
    {
        PRINT_SECTION("insert duplicate keys");
        std::size_t n = 40000, uniqueN = 10000;
        auto hostPairs = aclco::test::MakeExamplesWithDuplicates<Key, Value>(seed, n, uniqueN, sent);
        if (hostPairs.empty())
        {
            SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
        }
        std::size_t uniq = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent).size();
        auto map =
            aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, aclco::test::dmap_factory::LinearProbing<Key>>(
                initCap, sent, stream);
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(hostPairs.size());
        dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);
        aclco::test::Sync(stream);
        map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(hostPairs.size()), stream);
        aclco::test::Sync(stream);
        CAPTURE(n, uniqueN, uniq);
        REQUIRE_PRINT(map.Size() == static_cast<decltype(map.Size())>(uniq));
    }
}
