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
struct DMapCDCtx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<K, V> sent;
    std::size_t initCap{0};
};

template <typename K, typename V>
DMapCDCtx<K, V>& CDCtx()
{
    static DMapCDCtx<K, V>* c = new DMapCDCtx<K, V>();
    return *c;
}

template <typename K, typename V>
void SetupCD(int initSize, int seed)
{
    auto& c = CDCtx<K, V>();
    (void)seed;
    c.stream = c.sg.stream;
    c.sent = MakeDefaultSentinels<K, V>();
    c.initCap = static_cast<std::size_t>(initSize);
}

template <typename K, typename V>
TestResult TestCreate()
{
    auto& c = CDCtx<K, V>();
    auto start = std::chrono::high_resolution_clock::now();
    {
        auto map = aclco::test::dmap_factory::MakeDynamicMap<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>>(
            c.initCap, c.sent, c.stream);
        Sync(c.stream);
        auto end = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        return TestResult(us, us, 0);
    }
}

template <typename K, typename V>
TestResult TestDestroy()
{
    auto& c = CDCtx<K, V>();
    std::optional<DMapT<K, V>> m;
    m.emplace(aclco::Extent<std::size_t>(c.initCap), c.sent.emptyKey, c.sent.emptyValue, c.sent.erasedKey,
              aclco::EqualTo<K>{}, aclco::LinearProbing<aclco::murmurhash3_32<K>>{}, aclco::Storage<1>{}, c.stream);
    Sync(c.stream);
    auto start = std::chrono::high_resolution_clock::now();
    m.reset();
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

#define CD_ARGS(name, a100) \
    REGISTER_PERFORMANCE_ARGS(name, a100, (std::initializer_list<std::tuple<int, int>>{{40000000, 200}}), int, int)

REGISTER_PERFORMANCE_TEST(dmapCreateI32, (TestCreate<uint32_t, uint32_t>), (SetupCD<uint32_t, uint32_t>), int, int);
REGISTER_PERFORMANCE_TEST(dmapCreateI64, (TestCreate<uint64_t, uint64_t>), (SetupCD<uint64_t, uint64_t>), int, int);
REGISTER_PERFORMANCE_TEST(dmapCreateI16, (TestCreate<uint16_t, uint16_t>), (SetupCD<uint16_t, uint16_t>), int, int);
REGISTER_PERFORMANCE_TEST(dmapDestroyI32, (TestDestroy<uint32_t, uint32_t>), (SetupCD<uint32_t, uint32_t>), int, int);
REGISTER_PERFORMANCE_TEST(dmapDestroyI64, (TestDestroy<uint64_t, uint64_t>), (SetupCD<uint64_t, uint64_t>), int, int);
REGISTER_PERFORMANCE_TEST(dmapDestroyI16, (TestDestroy<uint16_t, uint16_t>), (SetupCD<uint16_t, uint16_t>), int, int);

CD_ARGS(dmapCreateI32, "create i32 @40M (A100: 8.22ms)");
CD_ARGS(dmapCreateI64, "create i64 @40M (A100: 16.34ms)");
CD_ARGS(dmapCreateI16, "create i16 @40M (A100: 4.14ms)");
CD_ARGS(dmapDestroyI32, "destroy i32 @40M (A100: 1.04ms)");
CD_ARGS(dmapDestroyI64, "destroy i64 @40M (A100: 1.95ms)");
CD_ARGS(dmapDestroyI16, "destroy i16 @40M (A100: 3.52us)");

}  // namespace aclco::test
