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
struct InsertTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::vector<aclco::Pair<K, V>> hostPairs;
    DeviceBuffer<aclco::Pair<K, V>> dPairs;
    std::optional<map_factory::StaticMapT<K, V, BucketSize, map_factory::LinearProbing<K>, aclco::EqualTo<K>>> map;
};

template <typename K, typename V, int BucketSize>
InsertTestContext<K, V, BucketSize>& GetContext() {
    static InsertTestContext<K, V, BucketSize> ctx;
    return ctx;
}

template <typename K, typename V, int BucketSize>
void SetupInsertTest(int numKeys, double loadFactor, int seed, std::string keyDistribution) {
    auto& ctx = GetContext<K, V, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;

    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key, Value>();
    auto capacity = static_cast<size_t>(numKeys / loadFactor);
    ctx.map = map_factory::MakeStaticMap<Key, Value, BS, map_factory::LinearProbing<Key>>(capacity, sent, ctx.stream);
    ctx.hostPairs = MakeExamples<Key, Value>(seed, numKeys, sent, keyDistribution);
    ctx.dPairs = DeviceBuffer<aclco::Pair<K, V>>(ctx.hostPairs.size());
    ctx.dPairs.CopyFromHostAsync(ctx.hostPairs.data(), ctx.hostPairs.size(), ctx.stream);

    ctx.map->Clear(ctx.stream);
}

template <typename K, typename V, int BucketSize>
TestResult TestInsert() {
    auto& ctx = GetContext<K, V, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.map->Insert(static_cast<void*>(ctx.dPairs.Data()),
                    aclco::Extent<std::size_t>(ctx.hostPairs.size()), ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    ctx.map->Clear(ctx.stream);

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(insertCase1, (TestInsert<uint32_t,uint32_t,1>), (SetupInsertTest<uint32_t,uint32_t,1>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(insertCase2, (TestInsert<uint64_t,uint64_t,1>), (SetupInsertTest<uint64_t,uint64_t,1>), int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(insertCase1, "insert_uint32",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {80000000,0.5,200,"uniform"}
    }),
    int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(insertCase2, "insert_uint64",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {80000000,0.5,200,"uniform"}
    }),
    int, double, int, std::string);

}
