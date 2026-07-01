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
struct DMapResvCtx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<K, V> sent;
    std::size_t initCap{0};
    std::size_t n{0};
    std::size_t batch{0};
    DeviceBuffer<aclco::Pair<K, V>> dPairs;
};

template <typename K, typename V>
DMapResvCtx<K, V>& ResvCtx()
{
    static DMapResvCtx<K, V>* c = new DMapResvCtx<K, V>();
    return *c;
}

template <typename K, typename V>
void SetupResv(int numInputs, int initSize, int batchSize, int seed, std::string dist)
{
    auto& c = ResvCtx<K, V>();
    c.stream = c.sg.stream;
    c.sent = MakeDefaultSentinels<K, V>();
    c.initCap = static_cast<std::size_t>(initSize);
    c.batch = static_cast<std::size_t>(batchSize);
    auto pairs =
        MakeExamples<K, V>(static_cast<uint32_t>(seed), static_cast<std::size_t>(numInputs), c.sent, dist, true);
    c.n = pairs.size();
    c.dPairs = DeviceBuffer<aclco::Pair<K, V>>(c.n);
    c.dPairs.CopyFromHostAsync(pairs.data(), c.n, c.stream);
    Sync(c.stream);
}

template <typename K, typename V>
TestResult TestResv()
{
    auto& c = ResvCtx<K, V>();
    auto map = aclco::test::dmap_factory::MakeDynamicMap<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>>(
        c.initCap, c.sent, c.stream);
    Sync(c.stream);
    auto* base = c.dPairs.Data();
    auto start = std::chrono::high_resolution_clock::now();
    map.Reserve(static_cast<typename DMapT<K, V>::SizeType>(c.n), c.stream);
    for (std::size_t off = 0; off < c.n; off += c.batch)
    {
        std::size_t cur = std::min(c.batch, c.n - off);
        map.InsertAsync(static_cast<void*>(base + off), aclco::Extent<std::size_t>(cur), c.stream);
    }
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

REGISTER_PERFORMANCE_TEST(dmapResvI32, (TestResv<uint32_t, uint32_t>), (SetupResv<uint32_t, uint32_t>), int, int, int,
                          int, std::string);
REGISTER_PERFORMANCE_TEST(dmapResvI64, (TestResv<uint64_t, uint64_t>), (SetupResv<uint64_t, uint64_t>), int, int, int,
                          int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapResvI32, "reserve i32 bs800k (A100: 22.85@40M / 15.61@80M)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 80000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapResvI64, "reserve i64 bs800k (A100: 46.09@40M / 30.88@80M)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 80000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);

}  // namespace aclco::test
