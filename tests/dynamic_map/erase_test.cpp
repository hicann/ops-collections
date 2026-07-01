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

TEMPLATE_TEST_CASE_SIG("dynamic_map erase correctness", "[dynamic_map][erase]",
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
    PRINT_BEFORE_EXEC_WITH_PROBE("erase correctness", Key, Value, BS, initCap, n, params, Probe);

    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty())
    {
        SKIP("Can not create enough pairs for Key type");
    }
    n = hostPairs.size();
    std::vector<Key> keys(n);
    for (std::size_t i = 0; i < n; ++i) keys[i] = hostPairs[i].first;

    auto rebuild = [&]()
    {
        auto map = aclco::test::dmap_factory::MakeDynamicMap<Key, Value, BS, Probe>(initCap, sent, stream);
        aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
        dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
        aclco::test::Sync(stream);
        map.Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), stream);
        aclco::test::Sync(stream);
        return map;
    };

    SECTION("erase first half, second half remains")
    {
        PRINT_SECTION("erase first half, second half remains");
        auto map = rebuild();
        std::size_t half = n / 2;
        aclco::test::DeviceBuffer<Key> dKeys(n);
        dKeys.CopyFromHostAsync(keys.data(), n, stream);
        aclco::test::Sync(stream);
        auto del = map.Erase(static_cast<void*>(dKeys.Data()), aclco::Extent<std::size_t>(half), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(del == static_cast<decltype(del)>(half));
        auto flags = aclco::test::dmap_factory::RetrieveViaContains<Key>(map, keys, stream);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            bool exp = (i >= half);
            if ((flags[i] != 0) != exp) ++wrong;
        }
        INFO("wrong = " << wrong);
        REQUIRE_PRINT(wrong == 0);
    }

    SECTION("erase non-existent keys → 0 deleted")
    {
        PRINT_SECTION("erase non-existent keys → 0 deleted");
        auto map = rebuild();
        auto other = aclco::test::MakeExamples<Key, Value>(seed + 99u, n, sent, "uniform", true);
        std::unordered_set<Key> present(keys.begin(), keys.end());
        std::vector<Key> absent;
        for (auto const& p : other)
            if (!present.count(p.first)) absent.push_back(p.first);
        if (absent.empty())
        {
            SKIP("no absent keys");
        }
        aclco::test::DeviceBuffer<Key> dAbsent(absent.size());
        dAbsent.CopyFromHostAsync(absent.data(), absent.size(), stream);
        aclco::test::Sync(stream);
        auto del = map.Erase(static_cast<void*>(dAbsent.Data()), aclco::Extent<std::size_t>(absent.size()), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(del == 0);
    }

    SECTION("erase n == 0")
    {
        PRINT_SECTION("erase n == 0");
        auto map = rebuild();
        aclco::test::DeviceBuffer<Key> dKeys(1);
        auto del = map.Erase(static_cast<void*>(dKeys.Data()), aclco::Extent<std::size_t>(0), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(del == 0);
    }

    SECTION("erase old-submap keys then re-insert lands in active submap, Find returns new value")
    {
        PRINT_SECTION("erase old keys then re-insert with new value");
        auto map = rebuild();
        REQUIRE_PRINT(map.NumSubmaps() > 1);
        auto sizeBefore = map.Size();

        std::size_t half = n / 2;
        aclco::test::DeviceBuffer<Key> dKeys(n);
        dKeys.CopyFromHostAsync(keys.data(), n, stream);
        aclco::test::Sync(stream);
        auto del = map.Erase(static_cast<void*>(dKeys.Data()), aclco::Extent<std::size_t>(half), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(del == static_cast<decltype(del)>(half));

        std::vector<Pair<Key, Value>> reins(half);
        for (std::size_t i = 0; i < half; ++i)
        {
            reins[i].first = keys[i];
            reins[i].second = hostPairs[(i + 1) % n].second;
        }
        aclco::test::DeviceBuffer<Pair<Key, Value>> dReins(half);
        dReins.CopyFromHostAsync(reins.data(), half, stream);
        aclco::test::Sync(stream);
        auto ins = map.Insert(static_cast<void*>(dReins.Data()), aclco::Extent<std::size_t>(half), stream);
        aclco::test::Sync(stream);
        REQUIRE_PRINT(ins == static_cast<decltype(ins)>(half));
        REQUIRE_PRINT(map.Size() == sizeBefore);

        auto observed = aclco::test::dmap_factory::RetrieveViaFind<Key, Value>(map, keys, stream);
        std::size_t wrong = 0;
        for (std::size_t i = 0; i < half; ++i)
            if (observed[i] != reins[i].second) ++wrong;
        for (std::size_t i = half; i < n; ++i)
            if (observed[i] != hostPairs[i].second) ++wrong;
        INFO("erase+reinsert mismatch = " << wrong);
        REQUIRE_PRINT(wrong == 0);
    }
}
