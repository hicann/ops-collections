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
#include <cstdint>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dynamic_map.h"
#include "pair.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"

using K = uint32_t;
using V = uint32_t;
using PairKV = aclco::Pair<K, V>;
using DMap = aclco::DynamicMap<K, V, aclco::Extent<std::size_t>, aclco::EqualTo<K>,
                               aclco::LinearProbing<aclco::murmurhash3_32<K>>, aclco::Storage<1>>;

TEST_CASE("async batched insert unique single-submap safety", "[dynamic_map][asyncsafe]")
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sent = aclco::test::MakeDefaultSentinels<K, V>();

    const std::size_t n = 2000000;
    std::mt19937 rng(123u);
    std::uniform_int_distribution<K> kd(1, 200000000);
    std::vector<PairKV> pairs;
    pairs.reserve(n);
    std::unordered_set<K> seen;
    seen.reserve(n * 2);
    std::unordered_map<K, V> golden;
    golden.reserve(n);
    std::uniform_int_distribution<V> vd(1, 200000000);
    while (pairs.size() < n)
    {
        K k = kd(rng);
        if (k == sent.emptyKey || k == sent.erasedKey || !seen.insert(k).second) continue;
        V v = vd(rng);
        golden[k] = v;
        pairs.emplace_back(k, v);
    }

    DMap map(aclco::Extent<std::size_t>(8000000), sent.emptyKey, sent.emptyValue, sent.erasedKey, {}, {}, {}, stream);
    aclco::test::DeviceBuffer<PairKV> dPairs(n);
    dPairs.CopyFromHostAsync(pairs.data(), n, stream);

    const std::size_t batch = 800000;
    for (std::size_t off = 0; off < n; off += batch)
    {
        std::size_t cur = std::min(batch, n - off);
        map.InsertAsync(static_cast<void*>(dPairs.Data() + off), aclco::Extent<std::size_t>(cur), stream);
    }
    aclrtSynchronizeStream(stream);
    REQUIRE(map.NumSubmaps() == 1);

    std::vector<K> qk(n);
    for (std::size_t i = 0; i < n; ++i) qk[i] = pairs[i].first;
    aclco::test::DeviceBuffer<K> dK(n);
    dK.CopyFromHostAsync(qk.data(), n, stream);
    aclco::test::DeviceBuffer<V> dV(n);
    map.Find(static_cast<void*>(dK.Data()), static_cast<void*>(dV.Data()), aclco::Extent<std::size_t>(n), stream);
    std::vector<V> outV(n);
    dV.CopyToHostAsync(outV.data(), n, stream);
    aclrtSynchronizeStream(stream);
    std::size_t notFound = 0, wrongVal = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (outV[i] == golden[qk[i]]) continue;
        if (outV[i] == sent.emptyValue)
            ++notFound;
        else
            ++wrongVal;
    }
    INFO("async-safe notFound=" << notFound << " wrongVal=" << wrongVal);
    REQUIRE(notFound == 0);
    REQUIRE(wrongVal == 0);
}

TEST_CASE("async batched insert unique WITH growth safety", "[dynamic_map][asyncsafe][grow]")
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sent = aclco::test::MakeDefaultSentinels<K, V>();

    const std::size_t n = 2000000;
    std::mt19937 rng(321u);
    std::uniform_int_distribution<K> kd(1, 200000000);
    std::vector<PairKV> pairs;
    pairs.reserve(n);
    std::unordered_set<K> seen;
    seen.reserve(n * 2);
    std::unordered_map<K, V> golden;
    golden.reserve(n);
    std::uniform_int_distribution<V> vd(1, 200000000);
    while (pairs.size() < n)
    {
        K k = kd(rng);
        if (k == sent.emptyKey || k == sent.erasedKey || !seen.insert(k).second) continue;
        V v = vd(rng);
        golden[k] = v;
        pairs.emplace_back(k, v);
    }

    DMap map(aclco::Extent<std::size_t>(100000), sent.emptyKey, sent.emptyValue, sent.erasedKey, {}, {}, {}, stream);
    aclco::test::DeviceBuffer<PairKV> dPairs(n);
    dPairs.CopyFromHostAsync(pairs.data(), n, stream);
    const std::size_t batch = 800000;
    for (std::size_t off = 0; off < n; off += batch)
    {
        std::size_t cur = std::min(batch, n - off);
        map.InsertAsync(static_cast<void*>(dPairs.Data() + off), aclco::Extent<std::size_t>(cur), stream);
    }
    aclrtSynchronizeStream(stream);
    REQUIRE(map.NumSubmaps() > 1);

    std::vector<K> qk(n);
    for (std::size_t i = 0; i < n; ++i) qk[i] = pairs[i].first;
    aclco::test::DeviceBuffer<K> dK(n);
    dK.CopyFromHostAsync(qk.data(), n, stream);
    aclco::test::DeviceBuffer<V> dV(n);
    map.Find(static_cast<void*>(dK.Data()), static_cast<void*>(dV.Data()), aclco::Extent<std::size_t>(n), stream);
    std::vector<V> outV(n);
    dV.CopyToHostAsync(outV.data(), n, stream);
    aclrtSynchronizeStream(stream);
    std::size_t notFound = 0, wrongVal = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        if (outV[i] == golden[qk[i]]) continue;
        if (outV[i] == sent.emptyValue)
            ++notFound;
        else
            ++wrongVal;
    }
    INFO("async-grow notFound=" << notFound << " wrongVal=" << wrongVal << " submaps=" << map.NumSubmaps());
    REQUIRE(notFound == 0);
    REQUIRE(wrongVal == 0);
}
