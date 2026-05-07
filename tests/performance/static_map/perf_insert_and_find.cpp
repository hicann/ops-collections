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
struct InsertAndFindTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::vector<aclco::Pair<K, V>> hostPairs;
    DeviceBuffer<aclco::Pair<K, V>> dPairs;
    DeviceBuffer<V> dOutputFind;
    DeviceBuffer<unsigned char> dOutputInsert;
    std::optional<map_factory::StaticMapT<K, V, BucketSize, map_factory::LinearProbing<K>, aclco::EqualTo<K>>> map;
};

template <typename K, typename V, int BucketSize>
InsertAndFindTestContext<K, V, BucketSize>& GetContext() {
    static InsertAndFindTestContext<K, V, BucketSize> ctx;
    return ctx;
}

template <typename K, typename V, int BucketSize>
void SetupInsertAndFindTest(int numKeys, double loadFactor, int seed, std::string keyDistribution) {
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

    ctx.dOutputFind = DeviceBuffer<V>(ctx.hostPairs.size());
    ctx.dOutputInsert = DeviceBuffer<unsigned char>(ctx.hostPairs.size());

    ctx.map->Clear(ctx.stream);
}

template <typename K, typename V, int BucketSize>
TestResult TestInsertAndFind() {
    auto& ctx = GetContext<K, V, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.map->InsertAndFind(static_cast<void*>(ctx.dPairs.Data()),
                           static_cast<void*>(ctx.dOutputFind.Data()),
                           static_cast<void*>(ctx.dOutputInsert.Data()),
                           aclco::Extent<std::size_t>(ctx.hostPairs.size()), ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    ctx.map->Clear(ctx.stream);

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(insertAndFindCase1, (TestInsertAndFind<uint32_t,uint32_t,1>), (SetupInsertAndFindTest<uint32_t,uint32_t,1>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(insertAndFindCase2, (TestInsertAndFind<uint64_t,uint64_t,1>), (SetupInsertAndFindTest<uint64_t,uint64_t,1>), int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(insertAndFindCase1, "insert_and_find_uint32",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {80000000,0.5,200,"uniform"}
    }),
    int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(insertAndFindCase2, "insert_and_find_uint64",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {80000000,0.5,200,"uniform"}
    }),
    int, double, int, std::string);

}
