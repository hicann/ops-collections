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
#include <unordered_set>
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/set_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"
#include "tests/common/dump_table.h"
namespace
{
template <typename Key>
std::size_t ExpectedNum(const std::vector<Key>& hostKeys, std::vector<Key> keys)
{
  std::size_t count = 0;
  std::unordered_set<Key> keySet;
  keySet.reserve(hostKeys.size());
  for (const auto& key : hostKeys) {
    keySet.insert(key);
  }
  for (auto i = 0; i < keys.size(); ++i) {
    Key currentKey = keys[i];
    if (keySet.find(currentKey) != keySet.end()) {
      ++count;
    }
  }
  return count;
}

template <typename Key>
std::vector<Key> PickKeys(const std::vector<Key>& hostKeys, std::size_t startIdx, std::size_t countN)
{
  std::vector<Key> keys;
  keys.reserve(countN);
  for (std::size_t i = startIdx; i < hostKeys.size() && i < startIdx + countN; ++i)
  {
    keys.push_back(hostKeys[i]);
  }
  return keys;
}

} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_set count correctness",
  "[static_set][countCorrect]",
  ((typename K, int BucketSize, typename ProbeScheme), K, BucketSize, ProbeScheme),
  (uint32_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::DoubleHashing<uint64_t>),
  (uint64_t, 5, aclco::test::set_factory::DoubleHashing<uint64_t>),
  (float, 1, aclco::test::set_factory::DoubleHashing<float>),
  (float, 5, aclco::test::set_factory::DoubleHashing<float>),
  (float, 5, aclco::test::set_factory::LinearProbing<float>),
  (aclco::fp16_t, 1, aclco::test::set_factory::DoubleHashing<uint16_t>),
  (aclco::fp16_t, 5, aclco::test::set_factory::LinearProbing<uint16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  constexpr int BS = BucketSize;
  using Probe = ProbeScheme;
  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = GENERATE(164u, 100000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("count correctness", Key, BS, capacity, n, params, Probe);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
  dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  auto fail = set.Insert(static_cast<void*>(dKeys.Data()),
                        aclco::Extent<std::size_t>(hostKeys.size()),
                        stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("count keys in set") {
    PRINT_SECTION("count keys in set");

    std::size_t countN = static_cast<std::size_t>(0.5 * n);
    std::size_t startIdx = static_cast<std::size_t>(0.1 * n);

    auto countKeys = PickKeys<Key>(hostKeys, startIdx, countN);
    auto expected = ExpectedNum<Key>(hostKeys, countKeys);

    aclco::test::DeviceBuffer<Key> dKeys(countKeys.size());
    dKeys.CopyFromHostAsync(countKeys.data(), countKeys.size(), stream);

    auto result = set.Count(static_cast<void*>(dKeys.Data()),
                            aclco::Extent<std::size_t>(countKeys.size()),
                            stream);

    REQUIRE_PRINT(result == expected);
  }

  SECTION("count keys not in set") {
    PRINT_SECTION("count keys not in set");
    std::vector<Key> countKeys;
    std::size_t countN = static_cast<std::size_t>(0.5 * n);
    countKeys.reserve(countN);
    for (std::size_t i = 1; i < countN; ++i) {
      countKeys.push_back(static_cast<Key>(10000000000U + i));
    }

    aclco::test::DeviceBuffer<Key> dKeys(countKeys.size());
    dKeys.CopyFromHostAsync(countKeys.data(), countKeys.size(), stream);

    auto expected = ExpectedNum<Key>(hostKeys, countKeys);
    auto result = set.Count(static_cast<void*>(dKeys.Data()),
                            aclco::Extent<std::size_t>(countKeys.size()),
                            stream);

    REQUIRE_PRINT(result == expected);
  }
}
  TEMPLATE_TEST_CASE_SIG(
  "static_set count negative tests",
  "[static_set][countNegative]",
  ((typename K, int BucketSize), K, BucketSize),
  (uint32_t, 1),
  (uint32_t, 5),
  (int32_t, 1),
  (int32_t, 5),
  (uint64_t, 1),
  (uint64_t, 5),
  (float, 1),
  (float, 5),
  (aclco::fp16_t, 1),
  (aclco::fp16_t, 5))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = 128;

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto n = GENERATE(5u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("count negative tests", Key, BS, capacity, n, params);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dKeys(n);
  dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

  auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
  REQUIRE_PRINT(fail == 0);

  //重复计数同样的数据
  SECTION("count duplicate keys") {
    PRINT_SECTION("count duplicate keys");
    std::size_t countN = 3;

    std::vector<Key> countKeys;
    countKeys.reserve(countN);
    for (std::size_t i = 0; i < countN; ++i) {
      if(!hostKeys.empty()) {
        countKeys.push_back(hostKeys[0]);
      }
    }

    aclco::test::DeviceBuffer<Key> dKeys(countN);
    dKeys.CopyFromHostAsync(countKeys.data(), countN, stream);

    auto expected = ExpectedNum<Key>(hostKeys, countKeys);
    auto count = set.Count(static_cast<void*>(dKeys.Data()), 
                            aclco::Extent<std::size_t>(countN), stream);
    REQUIRE_PRINT(count == expected);
  }

  //传入nullptr
  SECTION("count nullptr") {
    PRINT_SECTION("count nullptr");
    std::size_t countN = 1;
    auto count = set.Count(nullptr, aclco::Extent<std::size_t>(countN), stream);
    REQUIRE_PRINT(count == 0);
  }
}