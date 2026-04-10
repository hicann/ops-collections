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
struct EraseTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::size_t numKeys;
    std::vector<K> hostKeys;
    DeviceBuffer<K> deviceKeys;
    std::optional<set_factory::StaticSetT<K, BucketSize, set_factory::DoubleHashing<K>, aclco::EqualTo<K>>> set;
};

template <typename K, int BucketSize>
EraseTestContext<K, BucketSize>& GetContext() {
    static EraseTestContext<K, BucketSize> ctx;
    return ctx;
}

template <typename K, int BucketSize>
void SetupEraseTest(int numKeys, double loadFactor, int seed, std::string keyDistribution) {
    auto& ctx = GetContext<K, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;
    ctx.numKeys = numKeys;

    using Key = K;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key>();
    auto capacity = static_cast<size_t>(numKeys / loadFactor);
    ctx.set = set_factory::MakeStaticSet<Key, BS, set_factory::DoubleHashing<Key>>(capacity, sent, ctx.stream);
    ctx.hostKeys = MakeExamples<Key>(seed, numKeys, sent, keyDistribution);
    ctx.deviceKeys = DeviceBuffer<K>(ctx.hostKeys.size());
    ctx.deviceKeys.CopyFromHostAsync(ctx.hostKeys.data(), ctx.hostKeys.size(), ctx.stream);

    std::vector<K> eraseKeys(numKeys);
    for (std::size_t i = 0; i < numKeys; ++i) {
        eraseKeys[i] = ctx.hostKeys[i];
    }

    ctx.deviceKeys = DeviceBuffer<K>(numKeys);
    ctx.deviceKeys.CopyFromHostAsync(eraseKeys.data(), numKeys, ctx.stream);
}

template <typename K, int BucketSize>
TestResult TestErase() {
    auto& ctx = GetContext<K, BucketSize>();

    ctx.set->Clear(ctx.stream);
    ctx.set->Insert(static_cast<void*>(ctx.deviceKeys.Data()),
                    aclco::Extent<std::size_t>(ctx.hostKeys.size()), ctx.stream);

    auto start = std::chrono::high_resolution_clock::now();
    ctx.set->Erase(static_cast<void*>(ctx.deviceKeys.Data()), 
                             aclco::Extent<std::size_t>(ctx.numKeys), ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(eraseCase1, (TestErase<uint32_t,1>), (SetupEraseTest<uint32_t,1>), int, double, int, std::string);
REGISTER_PERFORMANCE_TEST(eraseCase2, (TestErase<uint64_t,1>), (SetupEraseTest<uint64_t,1>), int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(eraseCase1, "erase_unit32",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {80000000,0.5,200,"uniform"}
    }),
    int, double, int, std::string);

REGISTER_PERFORMANCE_ARGS(eraseCase2, "erase_unit64",
    (std::initializer_list<std::tuple<int,double,int,std::string>>{
        {80000000,0.5,200,"uniform"}
    }),
    int, double, int, std::string);

}
