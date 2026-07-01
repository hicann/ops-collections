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
struct DMapFindCtx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<K, V> sent;
    std::size_t fn{0};
    DeviceBuffer<K> dKeys;
    DeviceBuffer<V> dVals;
    std::optional<DMapT<K, V>> map;
};

template <typename K, typename V>
DMapFindCtx<K, V>& FindCtx()
{
    static DMapFindCtx<K, V>* c = new DMapFindCtx<K, V>();
    return *c;
}

template <typename K, typename V>
void SetupDMapFind(int numKeys, int initSize, int findKeys, int seed, std::string dist)
{
    auto& c = FindCtx<K, V>();
    c.stream = c.sg.stream;
    c.sent = MakeDefaultSentinels<K, V>();

    auto pairs = MakeExamples<K, V>(static_cast<uint32_t>(seed), static_cast<std::size_t>(numKeys), c.sent, dist, true);
    std::size_t n = pairs.size();
    DeviceBuffer<aclco::Pair<K, V>> dPairs(n);
    dPairs.CopyFromHostAsync(pairs.data(), n, c.stream);

    c.map.emplace(aclco::Extent<std::size_t>(static_cast<std::size_t>(initSize)), c.sent.emptyKey, c.sent.emptyValue,
                  c.sent.erasedKey, aclco::EqualTo<K>{}, aclco::LinearProbing<aclco::murmurhash3_32<K>>{},
                  aclco::Storage<1>{}, c.stream);
    c.map->Insert(static_cast<void*>(dPairs.Data()), aclco::Extent<std::size_t>(n), c.stream);
    Sync(c.stream);

    c.fn = std::min<std::size_t>(static_cast<std::size_t>(findKeys), n);
    std::vector<K> qk(c.fn);
    for (std::size_t i = 0; i < c.fn; ++i) qk[i] = pairs[i].first;
    c.dKeys = DeviceBuffer<K>(c.fn);
    c.dKeys.CopyFromHostAsync(qk.data(), c.fn, c.stream);
    c.dVals = DeviceBuffer<V>(c.fn);
    c.dVals.MemsetZero(c.stream);
    Sync(c.stream);
}

template <typename K, typename V>
TestResult TestDMapFind()
{
    auto& c = FindCtx<K, V>();
    auto start = std::chrono::high_resolution_clock::now();
    c.map->Find(static_cast<void*>(c.dKeys.Data()), static_cast<void*>(c.dVals.Data()),
                aclco::Extent<std::size_t>(c.fn), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

REGISTER_PERFORMANCE_TEST(dmapFindI32, (TestDMapFind<uint32_t, uint32_t>), (SetupDMapFind<uint32_t, uint32_t>), int,
                          int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapFindI64, (TestDMapFind<uint64_t, uint64_t>), (SetupDMapFind<uint64_t, uint64_t>), int,
                          int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapFindI32, "dmap_find_i32 mr1.0 (A100: 23.27ms)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 80000000, 200, "uniform"}}),
                          int, int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapFindI64, "dmap_find_i64 mr1.0 (A100: 24.58ms)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 80000000, 200, "uniform"}}),
                          int, int, int, int, std::string);

}  // namespace aclco::test
