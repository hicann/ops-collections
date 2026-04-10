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
struct ClearTestContext {
    AclStreamGuard streamGuard;
    aclrtStream stream;
    std::optional<map_factory::StaticMapT<K, V, BucketSize, map_factory::LinearProbing<K>, aclco::EqualTo<K>>> map;
};

template <typename K, typename V, int BucketSize>
ClearTestContext<K, V, BucketSize>& GetContext() {
    static ClearTestContext<K, V, BucketSize> ctx;
    return ctx;
}

template <typename K, typename V, int BucketSize>
void SetupClearTest(int capacity) {
    auto& ctx = GetContext<K, V, BucketSize>();
    ctx.stream = ctx.streamGuard.stream;

    using Key = K;
    using Value = V;
    constexpr int BS = BucketSize;

    auto sent = MakeDefaultSentinels<Key, Value>();
    ctx.map = map_factory::MakeStaticMap<Key, Value, BS, map_factory::LinearProbing<Key>>(capacity, sent, ctx.stream);
}

template <typename K, typename V, int BucketSize>
TestResult TestClear() {
    auto& ctx = GetContext<K, V, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.map->Clear(ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

template <typename K, typename V, int BucketSize>
TestResult TestClearAsync() {
    auto& ctx = GetContext<K, V, BucketSize>();

    auto start = std::chrono::high_resolution_clock::now();
    ctx.map->ClearAsync(ctx.stream);
    Sync(ctx.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTimeUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return TestResult(cpuTimeUs, cpuTimeUs, 0);
}

REGISTER_PERFORMANCE_TEST(clearCase1, (TestClear<uint32_t,uint32_t,1>), (SetupClearTest<uint32_t,uint32_t,1>), int);
REGISTER_PERFORMANCE_TEST(clearCase2, (TestClear<uint64_t,uint64_t,1>), (SetupClearTest<uint64_t,uint64_t,1>), int);
REGISTER_PERFORMANCE_TEST(clearAsyncCase1, (TestClearAsync<uint32_t,uint32_t,1>), (SetupClearTest<uint32_t,uint32_t,1>), int);
REGISTER_PERFORMANCE_TEST(clearAsyncCase2, (TestClearAsync<uint64_t,uint64_t,1>), (SetupClearTest<uint64_t,uint64_t,1>), int);

REGISTER_PERFORMANCE_ARGS(clearCase1, "clear_unit32",
    (std::initializer_list<std::tuple<int>>{
        {160000000}
    }),
    int);

REGISTER_PERFORMANCE_ARGS(clearCase2, "clear_unit64",
    (std::initializer_list<std::tuple<int>>{
        {160000000}
    }),
    int);    

REGISTER_PERFORMANCE_ARGS(clearAsyncCase1, "clear_async_unit32",
    (std::initializer_list<std::tuple<int>>{
         {160000000}
    }),
    int);

REGISTER_PERFORMANCE_ARGS(clearAsyncCase2, "clear_async_unit64",
    (std::initializer_list<std::tuple<int>>{
         {160000000}
    }),
    int);

}
