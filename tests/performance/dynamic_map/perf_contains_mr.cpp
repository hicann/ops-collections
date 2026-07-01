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
struct DMapCtsMrCtx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<K, V> sent;
    std::size_t qn{0};
    DeviceBuffer<K> dKeys;
    DeviceBuffer<uint8_t> dOut;
    std::optional<DMapT<K, V>> map;
};

template <typename K, typename V>
DMapCtsMrCtx<K, V>& CmCtx()
{
    static DMapCtsMrCtx<K, V>* c = new DMapCtsMrCtx<K, V>();
    return *c;
}

template <typename K, typename V>
void SetupCtsMr(int numKeys, int initSize, int queryKeys, int matchPct, int seed, std::string dist)
{
    auto& c = CmCtx<K, V>();
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

    c.qn = std::min<std::size_t>(static_cast<std::size_t>(queryKeys), n);
    std::size_t hit = static_cast<std::size_t>(c.qn * (static_cast<double>(matchPct) / 100.0));
    if (hit > c.qn) hit = c.qn;
    const uint64_t domainTop = static_cast<uint64_t>(n) * 2ULL;
    std::vector<K> qk(c.qn);
    for (std::size_t i = 0; i < hit; ++i) qk[i] = pairs[i].first;
    for (std::size_t i = hit; i < c.qn; ++i) qk[i] = static_cast<K>(domainTop + 1 + (i - hit));
    c.dKeys = DeviceBuffer<K>(c.qn);
    c.dKeys.CopyFromHostAsync(qk.data(), c.qn, c.stream);
    c.dOut = DeviceBuffer<uint8_t>(c.qn);
    c.dOut.MemsetZero(c.stream);
    Sync(c.stream);
}

template <typename K, typename V>
TestResult TestCtsMr()
{
    auto& c = CmCtx<K, V>();
    auto start = std::chrono::high_resolution_clock::now();
    c.map->Contains(static_cast<void*>(c.dKeys.Data()), static_cast<void*>(c.dOut.Data()),
                    aclco::Extent<std::size_t>(c.qn), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

REGISTER_PERFORMANCE_TEST(dmapCtsMrI32, (TestCtsMr<uint32_t, uint32_t>), (SetupCtsMr<uint32_t, uint32_t>), int, int,
                          int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapCtsMrI64, (TestCtsMr<uint64_t, uint64_t>), (SetupCtsMr<uint64_t, uint64_t>), int, int,
                          int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapCtsMrI32, "contains i32 mr0.1/0.5 (A100: 21.42/19.41ms)",
                          (std::initializer_list<std::tuple<int, int, int, int, int, std::string>>{
                              {80000000, 40000000, 80000000, 10, 200, "uniform"},
                              {80000000, 40000000, 80000000, 50, 200, "uniform"}}),
                          int, int, int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapCtsMrI64, "contains i64 mr0.1/0.5 (A100: 26.82/24.46ms)",
                          (std::initializer_list<std::tuple<int, int, int, int, int, std::string>>{
                              {80000000, 40000000, 80000000, 10, 200, "uniform"},
                              {80000000, 40000000, 80000000, 50, 200, "uniform"}}),
                          int, int, int, int, int, std::string);

}  // namespace aclco::test
