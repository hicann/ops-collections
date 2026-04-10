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
struct DestroyTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    int capacity;
};

template <typename K, int BucketSize>
DestroyTestContext<K, BucketSize>& GetContext() {
    static DestroyTestContext<K, BucketSize> ctx;
    return ctx;
}

template <typename K, int BucketSize>
void SetupDestroyTest(int capacity) {
    auto& ctx = GetContext<K, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;
    ctx.capacity = capacity;
}

template <typename K, int BucketSize>
TestResult TestDestroy() {
    auto& ctx = GetContext<K, BucketSize>();

    auto sent = MakeDefaultSentinels<K>();
    auto setPtr = std::make_unique<set_factory::StaticSetT<K, BucketSize, set_factory::DoubleHashing<K>, aclco::EqualTo<K>>>(
        set_factory::MakeStaticSet<K, BucketSize, set_factory::DoubleHashing<K>>(
            ctx.capacity, sent, ctx.stream));

    auto start = std::chrono::high_resolution_clock::now();
    setPtr.reset();
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(destroyCase1, (TestDestroy<uint32_t,1>), (SetupDestroyTest<uint32_t,1>), int);
REGISTER_PERFORMANCE_TEST(destroyCase2, (TestDestroy<uint64_t,1>), (SetupDestroyTest<uint64_t,1>), int);

REGISTER_PERFORMANCE_ARGS(destroyCase1, "destroy_unit32",
    (std::initializer_list<std::tuple<int>>{
        {160000000}
    }),
    int);

REGISTER_PERFORMANCE_ARGS(destroyCase2, "destroy_unit64",
    (std::initializer_list<std::tuple<int>>{
        {160000000}
    }),
    int);

}
