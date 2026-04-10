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
struct CreateTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    int capacity;
};

template <typename K, int BucketSize>
CreateTestContext<K, BucketSize>& GetContext() {
    static CreateTestContext<K, BucketSize> ctx;
    return ctx;
}

template <typename K, int BucketSize>
void SetupCreateTest(int capacity) {
    auto& ctx = GetContext<K, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;
    ctx.capacity = capacity;
}

template <typename K, int BucketSize>
TestResult TestCreate() {
    auto& ctx = GetContext<K, BucketSize>();
    using Key   = K;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key>();

    auto start = std::chrono::high_resolution_clock::now();
    auto set = set_factory::MakeStaticSet<Key, BS, set_factory::DoubleHashing<Key>>(ctx.capacity, sent, ctx.stream);

    auto end = std::chrono::high_resolution_clock::now();

    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(createCase1, (TestCreate<uint32_t,1>), (SetupCreateTest<uint32_t,1>), int);
REGISTER_PERFORMANCE_TEST(createCase2, (TestCreate<uint64_t,1>), (SetupCreateTest<uint64_t,1>), int);

REGISTER_PERFORMANCE_ARGS(createCase1, "create_unit32",
    (std::initializer_list<std::tuple<int>>{
        {160000000}
    }),
    int);

REGISTER_PERFORMANCE_ARGS(createCase2, "create_unit64",
    (std::initializer_list<std::tuple<int>>{
        {160000000}
    }),
    int);

}