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

template <typename K, int BucketSize>
struct FindTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::vector<K> findKeysVec;
    DeviceBuffer<K> dKeys;
    DeviceBuffer<K> dFindKeys;   
    std::optional<set_factory::StaticSetT<K, BucketSize, set_factory::DoubleHashing<K>, aclco::EqualTo<K>>> set;
};

template <typename K, int BucketSize>
FindTestContext<K, BucketSize>& GetContext() {
    static FindTestContext<K, BucketSize> ctx;
    return ctx;
}

template <typename K, int BucketSize>
void SetupFindTest(int numKeys, int findKeys, double loadFactor, int seed, std::string keyDistribution) {
    auto& ctx = GetContext<K, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;

    using Key = K;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key>();
    auto capacity = static_cast<size_t>(numKeys / loadFactor);
    ctx.set = set_factory::MakeStaticSet<Key, BS, set_factory::DoubleHashing<Key>>(capacity, sent, ctx.stream);
    auto hostKeys = MakeExamples<Key>(seed, numKeys, sent, keyDistribution);
    auto dKeys = DeviceBuffer<K>(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), ctx.stream);

    ctx.set->Clear(ctx.stream);
    ctx.set->Insert(static_cast<void*>(dKeys.Data()),
                    aclco::Extent<std::size_t>(hostKeys.size()), ctx.stream);

    ctx.findKeysVec.reserve(findKeys);
    for (std::size_t i = 0; i < findKeys && i < hostKeys.size(); ++i) {
        ctx.findKeysVec.push_back(hostKeys[i]);
    }

    ctx.dKeys = DeviceBuffer<K>(ctx.findKeysVec.size());
    ctx.dKeys.CopyFromHostAsync(ctx.findKeysVec.data(), ctx.findKeysVec.size(), ctx.stream);

    ctx.dFindKeys = DeviceBuffer<K>(ctx.findKeysVec.size());
    ctx.dFindKeys.MemsetZero(ctx.stream);
}

template <typename K, int BucketSize>
TestResult TestFind() {
    auto& ctx = GetContext<K, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.set->Find(static_cast<void*>(ctx.dKeys.Data()),
             static_cast<void*>(ctx.dFindKeys.Data()),
             aclco::Extent<std::size_t>(ctx.findKeysVec.size()),
             ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(findCase1, (TestFind<uint32_t,1>), (SetupFindTest<uint32_t,1>), int, int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(findCase2, (TestFind<uint64_t,1>), (SetupFindTest<uint64_t,1>), int, int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(findCase1, "find_unit32",
    (std::initializer_list<std::tuple<int,int,double,int,std::string>>{
        {80000000, 80000000, 0.5, 200, "uniform"}
    }),
    int, int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(findCase2, "find_unit64",
    (std::initializer_list<std::tuple<int,int,double,int,std::string>>{
        {80000000, 80000000, 0.5, 200, "uniform"}
    }),
    int, int, double, int, std::string);

}
