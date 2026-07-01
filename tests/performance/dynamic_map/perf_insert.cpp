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
struct DMapInsertCtx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<K, V> sent;
    std::size_t initCap{0};
    std::size_t n{0};
    std::size_t batch{0};
    DeviceBuffer<aclco::Pair<K, V>> dPairs;
};

template <typename K, typename V>
DMapInsertCtx<K, V>& InsCtx()
{
    static DMapInsertCtx<K, V>* c = new DMapInsertCtx<K, V>();
    return *c;
}

template <typename K, typename V>
void SetupDMapInsert(int numKeys, int initSize, int batchSize, int seed, std::string dist)
{
    auto& c = InsCtx<K, V>();
    c.stream = c.sg.stream;
    c.sent = MakeDefaultSentinels<K, V>();
    c.initCap = static_cast<std::size_t>(initSize);
    c.batch = static_cast<std::size_t>(batchSize);
    auto pairs = MakeExamples<K, V>(static_cast<uint32_t>(seed), static_cast<std::size_t>(numKeys), c.sent, dist, true);
    c.n = pairs.size();
    c.dPairs = DeviceBuffer<aclco::Pair<K, V>>(c.n);
    c.dPairs.CopyFromHostAsync(pairs.data(), c.n, c.stream);
    Sync(c.stream);
}

template <typename K, typename V>
TestResult TestDMapInsert()
{
    auto& c = InsCtx<K, V>();
    auto map = aclco::test::dmap_factory::MakeDynamicMap<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>>(
        c.initCap, c.sent, c.stream);
    Sync(c.stream);
    auto start = std::chrono::high_resolution_clock::now();
    auto inserted = map.Insert(static_cast<void*>(c.dPairs.Data()), aclco::Extent<std::size_t>(c.n), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (inserted != static_cast<decltype(inserted)>(c.n))
        std::cerr << "[WARN] sync Insert count mismatch: " << inserted << " != " << c.n << std::endl;
    return TestResult(us, us, 0);
}

template <typename K, typename V>
TestResult TestDMapInsertAsync()
{
    auto& c = InsCtx<K, V>();
    auto map = aclco::test::dmap_factory::MakeDynamicMap<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>>(
        c.initCap, c.sent, c.stream);
    Sync(c.stream);
    auto start = std::chrono::high_resolution_clock::now();
    map.InsertAsync(static_cast<void*>(c.dPairs.Data()), aclco::Extent<std::size_t>(c.n), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

template <typename K, typename V>
TestResult TestDMapInsertSyncBatch()
{
    auto& c = InsCtx<K, V>();
    auto map = aclco::test::dmap_factory::MakeDynamicMap<K, V, 1, aclco::test::dmap_factory::LinearProbing<K>>(
        c.initCap, c.sent, c.stream);
    Sync(c.stream);
    auto* base = c.dPairs.Data();
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t off = 0; off < c.n; off += c.batch)
    {
        std::size_t cur = std::min(c.batch, c.n - off);
        map.Insert(static_cast<void*>(base + off), aclco::Extent<std::size_t>(cur), c.stream,
                   decltype(map)::InsertMode::kAppendUnique);
    }
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

REGISTER_PERFORMANCE_TEST(dmapInsertI32, (TestDMapInsert<uint32_t, uint32_t>), (SetupDMapInsert<uint32_t, uint32_t>),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapInsertI64, (TestDMapInsert<uint64_t, uint64_t>), (SetupDMapInsert<uint64_t, uint64_t>),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapInsertAsyncI32, (TestDMapInsertAsync<uint32_t, uint32_t>),
                          (SetupDMapInsert<uint32_t, uint32_t>), int, int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapInsertAsyncI64, (TestDMapInsertAsync<uint64_t, uint64_t>),
                          (SetupDMapInsert<uint64_t, uint64_t>), int, int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapInsertSyncBatchI32, (TestDMapInsertSyncBatch<uint32_t, uint32_t>),
                          (SetupDMapInsert<uint32_t, uint32_t>), int, int, int, int, std::string);
REGISTER_PERFORMANCE_TEST(dmapInsertSyncBatchI64, (TestDMapInsertSyncBatch<uint64_t, uint64_t>),
                          (SetupDMapInsert<uint64_t, uint64_t>), int, int, int, int, std::string);

REGISTER_PERFORMANCE_ARGS(dmapInsertI32, "dmap_insert_i32 SYNC bulk (A100: 8.46ms@40M / 18.84ms@160M)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 160000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_ARGS(dmapInsertI64, "dmap_insert_i64 SYNC bulk (A100: 142.29ms@40M / 20.22ms@160M)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 160000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_ARGS(dmapInsertAsyncI32, "dmap_insert_i32 ASYNC bulk (A100: 8.46ms@40M / 18.84ms@160M)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 160000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_ARGS(dmapInsertAsyncI64, "dmap_insert_i64 ASYNC bulk (A100: 142.29ms@40M / 20.22ms@160M)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 160000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_ARGS(dmapInsertSyncBatchI32, "dmap_insert_i32 SYNC batched800k (A100: 8.46/18.84)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 160000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);
REGISTER_PERFORMANCE_ARGS(dmapInsertSyncBatchI64, "dmap_insert_i64 SYNC batched800k (A100: 142.29/20.22)",
                          (std::initializer_list<std::tuple<int, int, int, int, std::string>>{
                              {80000000, 40000000, 800000, 200, "uniform"},
                              {80000000, 160000000, 800000, 200, "uniform"}}),
                          int, int, int, int, std::string);

}  // namespace aclco::test
