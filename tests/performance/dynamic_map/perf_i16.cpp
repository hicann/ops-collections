/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <random>

#include "../performance_test_framework.h"
#include "dynamic_map.h"
#include "tests/common/dmap_factory.h"

namespace aclco::test
{
static constexpr uint16_t I16_UNIQUE_DOMAIN = 32000;
static constexpr uint16_t I16_MISS_TOP = 65533;

template <int BS = 1>
using DMapI16 =
    aclco::test::dmap_factory::DynamicMapT<uint16_t, uint16_t, BS, aclco::test::dmap_factory::LinearProbing<uint16_t>,
                                           aclco::EqualTo<uint16_t>>;

inline std::vector<aclco::Pair<uint16_t, uint16_t>> MakeI16Pairs(uint32_t seed, std::size_t n)
{
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> kd(1, I16_UNIQUE_DOMAIN);
    std::vector<aclco::Pair<uint16_t, uint16_t>> out;
    out.reserve(n);
    for (std::size_t i = 0; i < n; ++i)
    {
        out.emplace_back(static_cast<uint16_t>(kd(rng)), static_cast<uint16_t>(kd(rng)));
    }
    return out;
}

struct I16Ctx
{
    AclStreamGuard sg;
    aclrtStream stream;
    Sentinels<uint16_t, uint16_t> sent;
    std::size_t initCap{0};
    std::size_t n{0};
    std::size_t qn{0};
    DeviceBuffer<aclco::Pair<uint16_t, uint16_t>> dPairs;
    DeviceBuffer<uint16_t> dKeys;
    DeviceBuffer<uint16_t> dFindOut;
    DeviceBuffer<uint8_t> dContainsOut;
    std::optional<DMapI16<>> map;
};

inline I16Ctx& Ctx()
{
    static I16Ctx& c = *(new I16Ctx());
    return c;
}

inline void SetupCommon(int numKeys, int initSize, int matchPct, int seed, bool buildQuery)
{
    auto& c = Ctx();
    c.stream = c.sg.stream;
    c.sent = MakeDefaultSentinels<uint16_t, uint16_t>();
    c.initCap = static_cast<std::size_t>(initSize);
    auto pairs = MakeI16Pairs(static_cast<uint32_t>(seed), static_cast<std::size_t>(numKeys));
    c.n = pairs.size();
    c.dPairs = DeviceBuffer<aclco::Pair<uint16_t, uint16_t>>(c.n);
    c.dPairs.CopyFromHostAsync(pairs.data(), c.n, c.stream);

    if (buildQuery)
    {
        c.qn = c.n;
        std::size_t hit = static_cast<std::size_t>(c.qn * (static_cast<double>(matchPct) / 100.0));
        if (hit > c.qn) hit = c.qn;
        std::mt19937 rng(seed ^ 0x55aa);
        std::uniform_int_distribution<int> hd(1, I16_UNIQUE_DOMAIN);
        std::vector<uint16_t> qk(c.qn);
        for (std::size_t i = 0; i < hit; ++i) qk[i] = static_cast<uint16_t>(hd(rng));
        std::uniform_int_distribution<int> md(I16_UNIQUE_DOMAIN + 1, I16_MISS_TOP);
        for (std::size_t i = hit; i < c.qn; ++i) qk[i] = static_cast<uint16_t>(md(rng));
        c.dKeys = DeviceBuffer<uint16_t>(c.qn);
        c.dKeys.CopyFromHostAsync(qk.data(), c.qn, c.stream);
    }
    Sync(c.stream);
}

inline void SetupWithMap(int numKeys, int initSize, int matchPct, int seed)
{
    SetupCommon(numKeys, initSize, matchPct, seed, /*buildQuery=*/true);
    auto& c = Ctx();
    c.map.emplace(aclco::Extent<std::size_t>(c.initCap), c.sent.emptyKey, c.sent.emptyValue, c.sent.erasedKey,
                  aclco::EqualTo<uint16_t>{}, aclco::LinearProbing<aclco::murmurhash3_32<uint16_t>>{},
                  aclco::Storage<1>{}, c.stream);
    c.map->Insert(static_cast<void*>(c.dPairs.Data()), aclco::Extent<std::size_t>(c.n), c.stream);
    Sync(c.stream);
}

void SetupI16Insert(int numKeys, int initSize, int matchPct, int seed)
{
    SetupCommon(numKeys, initSize, matchPct, seed, false);
}
TestResult TestI16Insert()
{
    auto& c = Ctx();
    DMapI16<> map(aclco::Extent<std::size_t>(c.initCap), c.sent.emptyKey, c.sent.emptyValue, c.sent.erasedKey, {}, {},
                  {}, c.stream);
    Sync(c.stream);
    auto* base = c.dPairs.Data();
    const std::size_t batch = 800000;
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t off = 0; off < c.n; off += batch)
    {
        std::size_t cur = std::min(batch, c.n - off);
        map.InsertAsync(static_cast<void*>(base + off), aclco::Extent<std::size_t>(cur), c.stream);
    }
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

// ---- Find ----
void SetupI16Find(int numKeys, int initSize, int matchPct, int seed)
{
    SetupWithMap(numKeys, initSize, matchPct, seed);
}
TestResult TestI16Find()
{
    auto& c = Ctx();
    auto& dOut = c.dFindOut;
    if (dOut.Size() < c.qn) dOut.Resize(c.qn);
    auto start = std::chrono::high_resolution_clock::now();
    c.map->Find(static_cast<void*>(c.dKeys.Data()), static_cast<void*>(dOut.Data()), aclco::Extent<std::size_t>(c.qn),
                c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

// ---- Contains ----
void SetupI16Contains(int numKeys, int initSize, int matchPct, int seed)
{
    SetupWithMap(numKeys, initSize, matchPct, seed);
}
TestResult TestI16Contains()
{
    auto& c = Ctx();
    auto& dOut = c.dContainsOut;
    if (dOut.Size() < c.qn) dOut.Resize(c.qn);
    auto start = std::chrono::high_resolution_clock::now();
    c.map->Contains(static_cast<void*>(c.dKeys.Data()), static_cast<void*>(dOut.Data()),
                    aclco::Extent<std::size_t>(c.qn), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

void SetupI16Erase(int numKeys, int initSize, int matchPct, int seed)
{
    SetupCommon(numKeys, initSize, matchPct, seed, true);
}
TestResult TestI16Erase()
{
    auto& c = Ctx();
    DMapI16<> map(aclco::Extent<std::size_t>(c.initCap), c.sent.emptyKey, c.sent.emptyValue, c.sent.erasedKey, {}, {},
                  {}, c.stream);
    map.Insert(static_cast<void*>(c.dPairs.Data()), aclco::Extent<std::size_t>(c.n), c.stream);
    Sync(c.stream);
    auto start = std::chrono::high_resolution_clock::now();
    map.Erase(static_cast<void*>(c.dKeys.Data()), aclco::Extent<std::size_t>(c.qn), c.stream);
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

void SetupI16Ioa(int numKeys, int initSize, int matchPct, int seed)
{
    SetupCommon(numKeys, initSize, matchPct, seed, false);
}
TestResult TestI16Ioa()
{
    auto& c = Ctx();
    DMapI16<> map(aclco::Extent<std::size_t>(c.initCap), c.sent.emptyKey, c.sent.emptyValue, c.sent.erasedKey, {}, {},
                  {}, c.stream);
    Sync(c.stream);
    auto* base = c.dPairs.Data();
    const std::size_t batch = 800000;
    auto start = std::chrono::high_resolution_clock::now();
    for (std::size_t off = 0; off < c.n; off += batch)
    {
        std::size_t cur = std::min(batch, c.n - off);
        map.InsertOrAssign(static_cast<void*>(base + off), aclco::Extent<std::size_t>(cur), c.stream);
    }
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

void SetupI16Reserve(int numKeys, int initSize, int matchPct, int seed)
{
    SetupCommon(numKeys, initSize, matchPct, seed, false);
}
TestResult TestI16Reserve()
{
    auto& c = Ctx();
    DMapI16<> map(aclco::Extent<std::size_t>(c.initCap), c.sent.emptyKey, c.sent.emptyValue, c.sent.erasedKey, {}, {},
                  {}, c.stream);
    Sync(c.stream);
    auto* base = c.dPairs.Data();
    const std::size_t batch = 800000;
    auto start = std::chrono::high_resolution_clock::now();
    map.Reserve(static_cast<DMapI16<>::SizeType>(c.n), c.stream);
    for (std::size_t off = 0; off < c.n; off += batch)
    {
        std::size_t cur = std::min(batch, c.n - off);
        map.InsertAsync(static_cast<void*>(base + off), aclco::Extent<std::size_t>(cur), c.stream);
    }
    Sync(c.stream);
    auto end = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return TestResult(us, us, 0);
}

REGISTER_PERFORMANCE_TEST(dmapI16Insert, (TestI16Insert), (SetupI16Insert), int, int, int, int);
REGISTER_PERFORMANCE_TEST(dmapI16Find, (TestI16Find), (SetupI16Find), int, int, int, int);
REGISTER_PERFORMANCE_TEST(dmapI16Contains, (TestI16Contains), (SetupI16Contains), int, int, int, int);
REGISTER_PERFORMANCE_TEST(dmapI16Erase, (TestI16Erase), (SetupI16Erase), int, int, int, int);

REGISTER_PERFORMANCE_ARGS(dmapI16Insert, "i16 insert bs800k (A100: 8.45@40M / 8.78@160M)",
                          (std::initializer_list<std::tuple<int, int, int, int>>{{80000000, 40000000, 100, 200},
                                                                                 {80000000, 160000000, 100, 200}}),
                          int, int, int, int);

REGISTER_PERFORMANCE_ARGS(dmapI16Find, "i16 find mr1.0/0.5/0.1 (A100: 5.4/5.4/5.42)",
                          (std::initializer_list<std::tuple<int, int, int, int>>{{80000000, 40000000, 100, 200},
                                                                                 {80000000, 40000000, 50, 200},
                                                                                 {80000000, 40000000, 10, 200}}),
                          int, int, int, int);

REGISTER_PERFORMANCE_ARGS(dmapI16Contains, "i16 contains mr1.0/0.5/0.1 (A100: 4.52/4.52/4.53)",
                          (std::initializer_list<std::tuple<int, int, int, int>>{{80000000, 40000000, 100, 200},
                                                                                 {80000000, 40000000, 50, 200},
                                                                                 {80000000, 40000000, 10, 200}}),
                          int, int, int, int);

REGISTER_PERFORMANCE_ARGS(dmapI16Erase, "i16 erase mr0.1/0.5/1.0 (A100: 15.47/15.6/15.71)",
                          (std::initializer_list<std::tuple<int, int, int, int>>{{80000000, 40000000, 10, 200},
                                                                                 {80000000, 40000000, 50, 200},
                                                                                 {80000000, 40000000, 100, 200}}),
                          int, int, int, int);

REGISTER_PERFORMANCE_TEST(dmapI16Ioa, (TestI16Ioa), (SetupI16Ioa), int, int, int, int);
REGISTER_PERFORMANCE_TEST(dmapI16Reserve, (TestI16Reserve), (SetupI16Reserve), int, int, int, int);

REGISTER_PERFORMANCE_ARGS(dmapI16Ioa, "i16 insert_or_assign bs800k (A100: 51.98@40M / 158.18@160M)",
                          (std::initializer_list<std::tuple<int, int, int, int>>{{65536, 40000000, 100, 200},
                                                                                 {65536, 160000000, 100, 200},
                                                                                 {80000000, 40000000, 100, 200},
                                                                                 {80000000, 160000000, 100, 200}}),
                          int, int, int, int);

REGISTER_PERFORMANCE_ARGS(dmapI16Reserve, "i16 reserve bs800k (A100: 12.28@40M / 8.215@80M)",
                          (std::initializer_list<std::tuple<int, int, int, int>>{{80000000, 40000000, 100, 200},
                                                                                 {80000000, 80000000, 100, 200}}),
                          int, int, int, int);

}  // namespace aclco::test
