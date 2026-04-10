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

namespace aclco::test {

template <typename K, typename V, int BucketSize>
struct ContainsTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::vector<K> findKeysVec;
    DeviceBuffer<K> dKeys;
    DeviceBuffer<unsigned char> dVals;
    std::optional<map_factory::StaticMapT<K, V, BucketSize, map_factory::LinearProbing<K>, aclco::EqualTo<K>>> map;
};

template <typename K, typename V, int BucketSize>
ContainsTestContext<K, V, BucketSize>& GetContext() {
    static ContainsTestContext<K, V, BucketSize> ctx;
    return ctx;
}

template <typename K, typename V, int BucketSize>
void SetupContainsTest(int numKeys, int findKeys, double loadFactor, int seed, std::string keyDistribution) {
    auto& ctx = GetContext<K, V, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;

    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key, Value>();
    auto capacity = static_cast<size_t>(numKeys / loadFactor);
    ctx.map = map_factory::MakeStaticMap<Key, Value, BS, map_factory::LinearProbing<K>>(capacity, sent, ctx.stream);
    auto hostPairs = MakeExamples<Key, Value>(seed, numKeys, sent, keyDistribution);
    auto dPairs = DeviceBuffer<aclco::Pair<K, V>>(hostPairs.size());
    dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), ctx.stream);

    ctx.map->Clear(ctx.stream);
    ctx.map->Insert(static_cast<void*>(dPairs.Data()),
                    aclco::Extent<std::size_t>(hostPairs.size()), ctx.stream);

    ctx.findKeysVec.reserve(findKeys);
    for (std::size_t i = 0; i < findKeys && i < hostPairs.size(); ++i) {
        ctx.findKeysVec.push_back(hostPairs[i].first);
    }

    ctx.dKeys = DeviceBuffer<K>(ctx.findKeysVec.size());
    ctx.dKeys.CopyFromHostAsync(ctx.findKeysVec.data(), ctx.findKeysVec.size(), ctx.stream);

    ctx.dVals = DeviceBuffer<unsigned char>(ctx.findKeysVec.size());
    ctx.dVals.MemsetZero(ctx.stream);
}

template <typename K, typename V, int BucketSize>
TestResult TestContains() {
    auto& ctx = GetContext<K, V, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.map->Contains(static_cast<void*>(ctx.dKeys.Data()),
                 static_cast<void*>(ctx.dVals.Data()),
                 aclco::Extent<std::size_t>(ctx.findKeysVec.size()),
                 ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(containsCase1, (TestContains<uint32_t,uint32_t,1>), (SetupContainsTest<uint32_t,uint32_t,1>), int, int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(containsCase2, (TestContains<uint64_t,uint64_t,1>), (SetupContainsTest<uint64_t,uint64_t,1>), int, int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(containsCase1, "contains_unit32",
    (std::initializer_list<std::tuple<int,int,double,int,std::string>>{
        {80000000, 80000000, 0.5, 200, "uniform"}
    }),
    int, int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(containsCase2, "contains_unit64",
    (std::initializer_list<std::tuple<int,int,double,int,std::string>>{
        {80000000, 80000000, 0.5, 200, "uniform"}
    }),
    int, int, double, int, std::string);

}
