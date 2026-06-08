/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/map_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;

template <typename Key, typename Value>
std::vector<Key> PickFirstKeys(const std::vector<aclco::Pair<Key, Value>>& hostPairs, std::size_t findN)
{
  std::vector<Key> keys;
  keys.reserve(findN);
  for (std::size_t i = 0; i < hostPairs.size() && keys.size() < findN; ++i)
  {
    keys.push_back(hostPairs[i].first);
  }
  return keys;
}

template <typename Key, typename Value>
std::vector<Value> ExpectedValuesForKeys(const std::vector<aclco::Pair<Key, Value>>& hostPairs,
                                         const std::vector<Key>& keys)
{
  std::unordered_map<Key, Value> pairMap;
  pairMap.reserve(hostPairs.size());
  for (const auto& pair : hostPairs) {
    pairMap[pair.first] = pair.second;
  }

  std::vector<Value> out;
  out.reserve(keys.size());
  for (const auto& k : keys)
  {
    auto it = pairMap.find(k);
    if (it != pairMap.end()) {
    out.push_back(it->second);
    } else {
      out.push_back(aclco::test::DefaultEmptyValue<Value>());
    }
  }
  return out;
}

} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map find correctness",
  "[static_map][findCorrect]",
  ((typename K, typename V, int BucketSize, typename ProbeScheme), K, V, BucketSize, ProbeScheme),
  (uint32_t, uint32_t, 1, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::DoubleHashing<uint32_t>),
  (uint64_t, uint64_t, 1, aclco::test::map_factory::LinearProbing<uint64_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::LinearProbing<uint64_t>),
  (float, float, 1, aclco::test::map_factory::LinearProbing<float>),
  (float, float, 5, aclco::test::map_factory::LinearProbing<float>),
  (float, float, 5, aclco::test::map_factory::DoubleHashing<float>),
  (aclco::fp16_t, aclco::fp16_t, 1, aclco::test::map_factory::LinearProbing<aclco::fp16_t>),
  (aclco::fp16_t, aclco::fp16_t, 5, aclco::test::map_factory::DoubleHashing<aclco::fp16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;
  using Probe = ProbeScheme;
  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(157u, 100000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("find correctness", Key, Value, BS, capacity, n, params, Probe);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                        aclco::Extent<std::size_t>(hostPairs.size()),
                        stream);
  REQUIRE_PRINT(fail == 0);

  //全部都能找到
  SECTION("normal test find all") {
  PRINT_SECTION("normal test find all");
  
  std::size_t findN = static_cast<std::size_t>(0.5 * n);

  auto hostKeys = PickFirstKeys<Key, Value>(hostPairs, findN);
  auto expected  = ExpectedValuesForKeys<Key, Value>(hostPairs, hostKeys);

  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  aclco::test::DeviceBuffer<Value> dVals(hostKeys.size());
  dVals.MemsetZero(stream);

  map.Find(static_cast<void*>(dKeys.Data()),
           static_cast<void*>(dVals.Data()),
           aclco::Extent<std::size_t>(hostKeys.size()),
           stream);

  aclco::test::Sync(stream);

  auto got = dVals.CopyToHost(stream);
  REQUIRE_PRINT(got.size() == expected.size());

  for (std::size_t i = 0; i < got.size(); ++i) {
    CAPTURE(i, hostKeys[i]);
    REQUIRE_PRINT(got[i] == expected[i]);
  }
 }

  //全部都找不到
  SECTION("find none") {
    PRINT_SECTION("find none");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(static_cast<Key>(10000000000U + i));
    }
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }

  //能找到部分键
  SECTION("find some keys") {
    PRINT_SECTION("find some keys");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    findKeys.push_back(hostPairs[0].first);
    for (std::size_t i = 0; i < findN - 1; ++i) {
      findKeys.push_back(static_cast<Key>(10000000000U + i));
    }
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == findN);

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map find negative test",
  "[static_map][findNegative]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint64_t, uint64_t, 1),
  (uint64_t, uint64_t, 5),
  (float, float, 1),
  (float, float, 5),
  (aclco::fp16_t, aclco::fp16_t, 1),
  (aclco::fp16_t, aclco::fp16_t, 5))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = 128;

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto n = GENERATE(5u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_WITH_PARAMS("find negative test", Key, Value, BS, capacity, n, params);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                        aclco::Extent<std::size_t>(hostPairs.size()),
                        stream);
  REQUIRE_PRINT(fail == 0);

  //查找重复键
  SECTION("find duplicate Keys") {
    PRINT_SECTION("find duplicate Keys");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(hostPairs[0].first);
    }
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }

  //查找nullptr
  SECTION("find nullptr") {
    PRINT_SECTION("find nullptr");
    std::size_t findN = 1;
    std::vector<Key> findKeys;
    auto expected = ExpectedValuesForKeys<Key, Value>(hostPairs, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<Value> dVals(findKeys.size());
    dVals.MemsetZero(stream);

    map.Find(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dVals.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    aclco::test::Sync(stream);

    auto got = dVals.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }
}


