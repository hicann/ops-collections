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

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"
#include "tests/common/set_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/test_print.h"

#include <vector>
#include <unordered_set>

TEMPLATE_TEST_CASE_SIG(
  "static_set erase correctness",
  "[static_set][erase]",
  ((typename K,  int BucketSize, typename ProbeScheme), K, BucketSize, ProbeScheme),
  (uint32_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 1, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (float, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (aclco::fp16_t, 1, aclco::test::set_factory::DoubleHashing<uint16_t>),
  (aclco::fp16_t, 5, aclco::test::set_factory::LinearProbing<uint16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbeScheme;
  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, Probe>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
    SKIP("skip n == 0");
  }
  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("erase correctness", Key, BS, capacity, n, params, Probe);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dKeys(n);
  dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

  auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
  REQUIRE_PRINT(fail == 0);

  //删除下标为偶数的键
  SECTION("erase even index keys") {
    PRINT_SECTION("erase even index keys");
    std::vector<Key> eraseKeys;
    std::size_t eraseN;
    if(n % 2 == 0) {
      eraseN = n / 2;
    }
    else {
      eraseN = n / 2 + 1;
    }
    eraseKeys.reserve(eraseN);
    for (std::size_t i = 0; i < n; i += 2) {
      eraseKeys.push_back(hostKeys[i]);
    }

    aclco::test::DeviceBuffer<Key> dKeys(eraseN);
    dKeys.CopyFromHostAsync(eraseKeys.data(), eraseN, stream);

    auto eraseFail = set.Erase(static_cast<void*>(dKeys.Data()), 
                              aclco::Extent<std::size_t>(eraseN), stream);
    REQUIRE_PRINT(eraseFail == 0);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
    auto base     = aclco::test::GoldenInsert<Key>(hostKeys, sent);
    auto golden   = aclco::test::GoldenAfterErase<Key>(std::move(base), eraseKeys);

    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
 }

 //删除下标为奇数的键
  SECTION("erase odd index keys") {
    PRINT_SECTION("erase odd index keys");
    std::vector<Key> eraseKeys;
    std::size_t eraseN = (n / 2);
    eraseKeys.reserve(eraseN);
    for (std::size_t i = 1; i < n; i += 2) {
      eraseKeys.push_back(hostKeys[i]);
    }

    aclco::test::DeviceBuffer<Key> dKeys(eraseN);
    dKeys.CopyFromHostAsync(eraseKeys.data(), eraseN, stream);

    auto eraseFail = set.Erase(static_cast<void*>(dKeys.Data()), 
                              aclco::Extent<std::size_t>(eraseN), stream);
    REQUIRE_PRINT(eraseFail == 0);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
    auto base     = aclco::test::GoldenInsert<Key>(hostKeys, sent);
    auto golden   = aclco::test::GoldenAfterErase<Key>(std::move(base), eraseKeys);

    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
 }

 //改变eraseN
 SECTION("erase with different ratios") {
  PRINT_SECTION("erase with different ratios");
  auto ratio = GENERATE(0.1f, 0.8f);
  std::size_t eraseN = static_cast<std::size_t>(ratio * n);
  if (eraseN == 0) eraseN = 1;
  if (eraseN > n) eraseN = n;

  std::vector<Key> eraseKeys;
  eraseKeys.reserve(eraseN);
  for (std::size_t i = 0; i < eraseN; ++i) {
      eraseKeys.push_back(hostKeys[i]);
  }
  aclco::test::DeviceBuffer<Key> dKeys(eraseN);
  dKeys.CopyFromHostAsync(eraseKeys.data(), eraseN, stream);

  auto eraseFail = set.Erase(static_cast<void*>(dKeys.Data()), 
                            aclco::Extent<std::size_t>(eraseN), stream);
  REQUIRE_PRINT(eraseFail == 0);

  auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
  auto base     = aclco::test::GoldenInsert<Key>(hostKeys, sent);
  auto golden   = aclco::test::GoldenAfterErase<Key>(std::move(base), eraseKeys);

  std::string diff;
  bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
  INFO(diff);
  REQUIRE_PRINT(ok);
 }

 //多次删除
 SECTION("multiple erase") {
  PRINT_SECTION("multiple erase");
  std::size_t firstEraseN = n / 2;
  std::size_t secondEraseN = (n - firstEraseN) / 2;
  if (firstEraseN == 0 || secondEraseN == 0) {
    SKIP("skip firstEraseN == 0 || secondEraseN == 0");
  }
  std::vector<Key> firstEraseKeys;
  std::vector<Key> secondEraseKeys;
  for (std::size_t i = 0; i < firstEraseN; ++i) {
      firstEraseKeys.push_back(hostKeys[i]);
  }
  for (std::size_t i = 0; i < secondEraseN; ++i) {
      secondEraseKeys.push_back(hostKeys[firstEraseN + i]);
  }
  aclco::test::DeviceBuffer<Key> dFirstKeys(firstEraseN);
  dFirstKeys.CopyFromHostAsync(firstEraseKeys.data(), firstEraseN, stream);

  auto eraseFail = set.Erase(static_cast<void*>(dFirstKeys.Data()), 
                            aclco::Extent<std::size_t>(firstEraseN), stream);
  REQUIRE_PRINT(eraseFail == 0);

  aclco::test::DeviceBuffer<Key> dSecondKeys(secondEraseN);
  dSecondKeys.CopyFromHostAsync(secondEraseKeys.data(), secondEraseN, stream);

  auto eraseFail2 = set.Erase(static_cast<void*>(dSecondKeys.Data()), 
                            aclco::Extent<std::size_t>(secondEraseN), stream);
  REQUIRE_PRINT(eraseFail2 == 0);

  std::vector<Key> allEraseKeys = firstEraseKeys;
  allEraseKeys.insert(allEraseKeys.end(), secondEraseKeys.begin(), secondEraseKeys.end());

  auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
  auto base     = aclco::test::GoldenInsert<Key>(hostKeys, sent);
  auto golden   = aclco::test::GoldenAfterErase<Key>(std::move(base), allEraseKeys);

  std::string diff;
  bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
  INFO(diff);
  REQUIRE_PRINT(ok);
 }
}

  TEMPLATE_TEST_CASE_SIG(
  "static_set erase negative tests",
  "[static_set][erase]",
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
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("erase negative tests", Key, BS, capacity, n, params);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dKeys(n);
  dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

  auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
  REQUIRE_PRINT(fail == 0);

  //重复删除同样的数据
  SECTION("erase duplicate keys") {
    PRINT_SECTION("erase duplicate keys");
    std::size_t eraseN = 3;

    std::vector<Key> eraseKeys;
    eraseKeys.reserve(eraseN);
    for (std::size_t i = 0; i < eraseN; ++i) {
      if(!hostKeys.empty()) {
        eraseKeys.push_back(hostKeys[0]);
      }
    }

    aclco::test::DeviceBuffer<Key> dKeys(eraseN);
    dKeys.CopyFromHostAsync(eraseKeys.data(), eraseN, stream);

    auto eraseFail = set.Erase(static_cast<void*>(dKeys.Data()), 
                            aclco::Extent<std::size_t>(eraseN), stream);
    REQUIRE_PRINT(eraseFail == eraseN - 1);
  }

  //删除不存在的数据
  SECTION("erase non-existent keys") {
    PRINT_SECTION("erase non-existent keys");
    std::size_t eraseN = 3;

    std::unordered_set<Key> existingKeys;
    for (const auto& key : hostKeys) {
      existingKeys.insert(key);
    }

    std::vector<Key> eraseKeys;
    eraseKeys.reserve(eraseN);
    
    Key candidate = static_cast<Key>(1);
    while (eraseKeys.size() < eraseN) {
      if (existingKeys.find(candidate) != existingKeys.end()) {
        candidate++;
        if (candidate == 0) {
          SKIP("Cannot find enough non-existent keys");
        }
        continue;
      }
      eraseKeys.push_back(candidate);
      candidate++;
      if (candidate == 0 && eraseKeys.size() < eraseN) {
        SKIP("Cannot find enough non-existent keys");
      }
    }

    aclco::test::DeviceBuffer<Key> dKeys(eraseN);
    dKeys.CopyFromHostAsync(eraseKeys.data(), eraseN, stream);

    auto eraseFail = set.Erase(static_cast<void*>(dKeys.Data()), 
                            aclco::Extent<std::size_t>(eraseN), stream);
    REQUIRE_PRINT(eraseFail == eraseN);
    std::cout<< "end" << std::endl;
  }

  //删除nullptr
  SECTION("erase nullptr") {
    PRINT_SECTION("erase nullptr");
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Key> dKeys(n);
    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

    auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                            aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail != 0);
    std::size_t eraseN = 1;
    auto eraseFail = set.Erase(nullptr, aclco::Extent<std::size_t>(eraseN), stream);
    REQUIRE_PRINT(eraseFail == eraseN);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_set eraseAsync",
  "[static_set][eraseAsync]",
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

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  capacity = set.Capacity();    
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
    SKIP("skip n == 0");
  }
  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("eraseAsync correctness", Key, BS, capacity, n, params);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dKeys(n);
  dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

  auto fail = set.Insert(static_cast<void*>(dKeys.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
  REQUIRE_PRINT(fail == 0);

  std::size_t eraseN = n / 2;
  if (eraseN == 0) {
      eraseN = 1;
  }

  std::vector<Key> eraseKeys;
  eraseKeys.reserve(eraseN);
  for (std::size_t i = 0; i < eraseN; ++i) {
      eraseKeys.push_back(hostKeys[i]);
  }

  aclco::test::DeviceBuffer<Key> dKeys2(eraseN);
  dKeys2.CopyFromHostAsync(eraseKeys.data(), eraseN, stream);

  set.EraseAsync(static_cast<void*>(dKeys2.Data()), 
                aclco::Extent<std::size_t>(eraseN), stream);
  aclco::test::Sync(stream);

  auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
  auto base     = aclco::test::GoldenInsert<Key>(hostKeys, sent);
  auto golden   = aclco::test::GoldenAfterErase<Key>(std::move(base), eraseKeys);

  std::string diff;
  bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
  INFO(diff);
  REQUIRE_PRINT(ok); 
}