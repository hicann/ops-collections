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

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/set_factory.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h" 

namespace
{
template <typename Key>
std::vector<Key> PickKeys(const std::vector<Key>& hostKeys, std::size_t findN)
{
  std::vector<Key> keys;
  keys.reserve(findN);
  for (std::size_t i = 0; i < hostKeys.size() && keys.size() < findN; ++i)
  {
    keys.push_back(hostKeys[i]);
  }
  return keys;
}

template <typename Key>
std::vector<bool> ExpectedBooleansForKeys(const std::vector<Key>& hostKeys,
                                         const std::vector<Key>& keys)
{
  std::unordered_set<Key> keySet;
  keySet.reserve(hostKeys.size());
  for (const auto& key : hostKeys) {
    keySet.insert(key);
  }
  
  std::vector<bool> out;
  out.reserve(keys.size());
  for (const auto& k: keys) {
    out.push_back(keySet.find(k) != keySet.end());
  }
  return out;

}

} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_set contains correctness",
  "[static_set][containsCorrect]",
  ((typename K,  int BucketSize, typename ProbeScheme), K, BucketSize, ProbeScheme),
  (uint32_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (int32_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (int32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 5, aclco::test::set_factory::LinearProbing<uint32_t>))
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
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("contains correctness", Key, BS, capacity, n, params, Probe);

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

  //全部都包含
  SECTION("normal test contain all") {
    PRINT_SECTION("normal test contain all");

    std::size_t findN = static_cast<std::size_t>(0.5 * n);
    auto findKeys = PickKeys<Key>(hostKeys, findN);
    auto expected = ExpectedBooleansForKeys<Key>(hostKeys, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<unsigned char> dFindKeys(findKeys.size());
    dFindKeys.MemsetZero(stream);

    set.Contains(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dFindKeys.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    auto got = dFindKeys.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(expected[i] == (got[i] != 0));
    }
  }

  //全部都不包含
  SECTION("contain none") {
    PRINT_SECTION("contain none");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(static_cast<Key>(10000000000U + i));
    }
    auto expected = ExpectedBooleansForKeys<Key>(hostKeys, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<unsigned char> dFindKeys(findKeys.size());
    dFindKeys.MemsetZero(stream);

    set.Contains(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dFindKeys.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    auto got = dFindKeys.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(expected[i] == (got[i] != 0));
    }
  }

  //包含部分键
  SECTION("contain some keys") {
    PRINT_SECTION("contain some keys");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    findKeys.push_back(hostKeys[0]);
    for (std::size_t i = 0; i < findN - 1; ++i) {
      findKeys.push_back(static_cast<Key>(10000000000U + i));
    }
    auto expected = ExpectedBooleansForKeys<Key>(hostKeys, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<unsigned char> dFindKeys(findKeys.size());
    dFindKeys.MemsetZero(stream);

    set.Contains(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dFindKeys.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    auto got = dFindKeys.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == findN);

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(expected[i] == (got[i] != 0));
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_set contains negative test",
  "[static_set][containsNegative]",
  ((typename K, int BucketSize), K, BucketSize),
  (uint32_t, 1),
  (uint32_t, 5),
  (uint64_t, 1),
  (uint64_t, 5),
  (float, 1),
  (float, 5))
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

  auto seed = GENERATE(1u);
  auto n = GENERATE(5u);

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("contains negative test", Key, BS, capacity, n, params);

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

  //查找重复键
  SECTION("find duplicate Keys") {
    PRINT_SECTION("find duplicate Keys");
    std::size_t findN = 3;
    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(hostKeys[0]);
    }
    auto expected = ExpectedBooleansForKeys<Key>(hostKeys, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<unsigned char> dFindKeys(findKeys.size());
    dFindKeys.MemsetZero(stream);

    set.Contains(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dFindKeys.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    auto got = dFindKeys.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(expected[i] == (got[i] != 0));
    }
  }

  //查找nullptr
  SECTION("find nullptr") {
    PRINT_SECTION("find nullptr");
    std::size_t findN = 1;
    std::vector<Key> findKeys;
    auto expected = ExpectedBooleansForKeys<Key>(hostKeys, findKeys);

    aclco::test::DeviceBuffer<Key> dKeys(findKeys.size());
    dKeys.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

    aclco::test::DeviceBuffer<unsigned char> dFindKeys(findKeys.size());
    dFindKeys.MemsetZero(stream);

    set.Contains(static_cast<void*>(dKeys.Data()),
            static_cast<void*>(dFindKeys.Data()),
            aclco::Extent<std::size_t>(findKeys.size()),
            stream);

    auto got = dFindKeys.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(expected[i] == (got[i] != 0));
    }
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_set containsAsync",
  "[static_set][containsAsync]",
  ((typename K, int BucketSize), K, BucketSize),
  (uint32_t, 1),
  (uint32_t, 5),
  (int32_t, 1),
  (int32_t, 5),
  (uint64_t, 1),
  (uint64_t, 5),
  (float, 1),
  (float, 5))
{ 
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  constexpr int BS = BucketSize;
  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = GENERATE(164u, 100000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("containsAsync correctness", Key, BS, capacity, n, params);

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
  
  std::size_t findN = 0.5 * capacity;
  std::vector<Key> findKeys;
  findKeys.reserve(findN);
  findKeys.push_back(hostKeys[0]);
  for (std::size_t i = 0; i < findN - 1; ++i) {
    findKeys.push_back(static_cast<Key>(10000000000U + i));
  }
  auto expected = ExpectedBooleansForKeys<Key>(hostKeys, findKeys);

  aclco::test::DeviceBuffer<Key> dKeys2(findKeys.size());
  dKeys2.CopyFromHostAsync(findKeys.data(), findKeys.size(), stream);

  aclco::test::DeviceBuffer<unsigned char> dFindKeys(findKeys.size());
  dFindKeys.MemsetZero(stream);

  set.ContainsAsync(static_cast<void*>(dKeys2.Data()),
          static_cast<void*>(dFindKeys.Data()),
          aclco::Extent<std::size_t>(findKeys.size()),
          stream);
  aclco::test::Sync(stream);

  auto got = dFindKeys.CopyToHost(stream);
  REQUIRE_PRINT(got.size() == findN);

  for (std::size_t i = 0; i < got.size(); ++i) {
  CAPTURE(i, findKeys[i]);
  REQUIRE_PRINT(expected[i] == (got[i] != 0));
  }
}
