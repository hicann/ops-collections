/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../performance_test_framework.h"
#include "dynamic_map.h"
#include "tests/common/dmap_factory.h"

namespace aclco::test
{
template <typename K, typename V>
using DMapT =
    aclco::test::dmap_factory::DynamicMapT<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>, aclco::EqualTo<K>>;

template <typename K, typename V>
struct DMapEraseCtx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<K, V> sent;
    std::size_t initCap{0};
    std::size_t n{0};
    std::size_t en{0};
    DeviceBuffer<aclco::Pair<K, V>> dPairs;
    DeviceBuffer<K> dKeys;
};

template <typename K, typename V>
DMapEraseCtx<K, V>& EraseCtx()
{
    static DMapEraseCtx<K, V>* c = new DMapEraseCtx<K, V>();
    return *c;
}

template <typename K, typename V>
void SetupErase(int numKeys, int initSize, int matchPct, int seed, std::string dist)
{
    auto& c = EraseCtx<K, V>();
    c.stream = c.sg.stream;
    c.sent = MakeDefaultSentinels<K, V>();
    c.initCap = static_cast<std::size_t>(initSize);

    auto pairs = MakeExamples<K, V>(static_cast<uint32_t>(seed), static_cast<std::size_t>(numKeys), c.sent, dist, true);
    c.n = pairs.size();
    c.dPairs = DeviceBuffer<aclco::Pair<K, V>>(c.n);
    c.dPairs.CopyFromHostAsync(pairs.data(), c.n, c.stream);

    c.en = c.n;
    std::size_t hit = static_cast<std::size_t>(c.en * (static_cast<double>(matchPct) / 100.0));
    if (hit > c.en) hit = c.en;
    const uint64_t domainTop = static_cast<uint64_t>(c.n) * 2ULL;  // unique key ∈ [1, 2n]
    std::vector<K> ek(c.en);
    for (std::size_t i = 0; i < hit; ++i) ek[i] = pairs[i].first;
    for (std::size_t i = hit; i < c.en; ++i) ek[i] = static_cast<K>(domainTop + 1 + (i - hit));
    c.dKeys = DeviceBuffer<K>(c.en);
    c.dKeys.CopyFromHostAsync(ek.data(), c.en, c.stream);
    Sync(c.stream);
}

template <typename K, typename V>
TestResult TestErase()
{
    auto& c = EraseCtx<K, V>();
    auto map = aclco::test::dmap_factory::MakeDynamicMap<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>>(
        c.initCap, c.sent, c.stream);
    map.Insert(static_cast<void*>(c.dPairs.Data()), aclco::Extent<std::size_t>(c.n), c.stream);
    Sync(c.stream);

    auto start = std::chrono::high_resolution_clock::now();
    map.Erase(static_cast<void*>(c.dKeys.Data()), aclco::Extent<std::size_t>(c.en), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

REGISTER_PERFORMANCE_TEST(dmapEraseI32, (TestErase<uint32_t, uint32_t>), (SetupErase<uint32_t, uint32_t>), int, int,
                          int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapEraseI64, (TestErase<uint64_t, uint64_t>), (SetupErase<uint64_t, uint64_t>), int, int,
                          int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapEraseI32, "erase i32 mr0.1/0.5/1.0 (A100: 38.41/41.15/40.15ms)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 10, 200, "uniform"},
                              {80000000, 40000000, 50, 200, "uniform"},
                              {80000000, 40000000, 100, 200, "uniform"}}),
                          int, int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapEraseI64, "erase i64 mr0.1/0.5/1.0 (A100: 40.1/42.99/41.98ms)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 10, 200, "uniform"},
                              {80000000, 40000000, 50, 200, "uniform"},
                              {80000000, 40000000, 100, 200, "uniform"}}),
                          int, int, int, int, std::string);

}  // namespace aclco::test
