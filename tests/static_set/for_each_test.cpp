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
#include <unordered_set>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"
#include "tests/common/set_factory.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/test_print.h"
#include "hash_functions.h"

namespace
{
template <typename Key>
struct CountEvenKeys {
  __gm__ uint32_t *counter;

  CountEvenKeys() : counter{nullptr} {}
  COLLECTION_DEVICE CountEvenKeys(__gm__ uint8_t *state) : counter{reinterpret_cast<__gm__ uint32_t*>(state)} {}

  template <typename SlotKey>
  COLLECTION_DEVICE void operator()(SlotKey key) const noexcept
  {
    if (key % 2 == 0) {
      AscendC::Simt::AtomicAdd(counter, 1u);
    }
  }
};

template <typename Key>
struct CountAll {
  __gm__ uint32_t *counter;

  CountAll() : counter{nullptr} {}
  COLLECTION_DEVICE CountAll(__gm__ uint8_t *state) : counter{reinterpret_cast<__gm__ uint32_t*>(state)} {}

  template <typename SlotKey>
  COLLECTION_DEVICE void operator()(SlotKey) const noexcept
  {
    AscendC::Simt::AtomicAdd(counter, 1u);
  }
};
}

TEMPLATE_TEST_CASE_SIG(
  "static_set for_each count even keys",
  "[static_set][for_each]",
  ((typename K, int BucketSize, typename ProbingScheme), K, BucketSize, ProbingScheme),
  (uint32_t, 1, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::LinearProbing<uint64_t>),
  (uint64_t, 5, aclco::test::set_factory::LinearProbing<uint64_t>),
  (uint64_t, 5, aclco::test::set_factory::DoubleHashing<uint64_t>),
  (aclco::fp16_t, 1, aclco::test::set_factory::LinearProbing<uint16_t>),
  (aclco::fp16_t, 5, aclco::test::set_factory::DoubleHashing<uint16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
  using Key   = K;
  constexpr int BS = BucketSize;

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

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dInsertKeys(hostKeys.size());
  dInsertKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("for_each count even keys", Key, BS, capacity, n, params, Probe);

  auto fail = set.Insert(static_cast<void*>(dInsertKeys.Data()),
                         aclco::Extent<std::size_t>(hostKeys.size()),
                         stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("for_each counts even keys") {
    PRINT_SECTION("for_each counts even keys");

    std::vector<Key> probeKeys = hostKeys;

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    set.template ForEach<CountEvenKeys<Key>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    std::size_t expectedEven = 0;
    for (auto k : hostKeys) {
      if (k % 2 == 0) { ++expectedEven; }
    }
    REQUIRE_PRINT(result[0] == expectedEven);
  }

  SECTION("for_each counts all matching keys") {
    PRINT_SECTION("for_each counts all matching keys");

    std::vector<Key> probeKeys = hostKeys;

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    set.template ForEach<CountAll<Key>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    REQUIRE_PRINT(result[0] == hostKeys.size());
  }

  SECTION("for_each with non-existent keys") {
    PRINT_SECTION("for_each with non-existent keys");

    std::unordered_set<Key> keySet(hostKeys.begin(), hostKeys.end());
    std::size_t probeN = n;
    std::vector<Key> probeKeys;
    probeKeys.reserve(probeN);
    Key candidate = static_cast<Key>(0);
    while (probeKeys.size() < probeN) {
      if (keySet.find(candidate) == keySet.end() && candidate != sent) {
        probeKeys.push_back(candidate);
      }
      ++candidate;
    }

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    set.template ForEach<CountAll<Key>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    REQUIRE_PRINT(result[0] == 0);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_set for_each with mixed existent and non-existent keys",
  "[static_set][for_each]",
  ((typename K, int BucketSize, typename ProbingScheme), K, BucketSize, ProbingScheme),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 5, aclco::test::set_factory::LinearProbing<uint64_t>),
  (uint64_t, 5, aclco::test::set_factory::DoubleHashing<uint64_t>),
  (aclco::fp16_t, 5, aclco::test::set_factory::LinearProbing<uint16_t>),
  (aclco::fp16_t, 5, aclco::test::set_factory::DoubleHashing<uint16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = GENERATE(164u, 100000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.5f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  CAPTURE(seed, n, capacity, BS);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dInsertKeys(hostKeys.size());
  dInsertKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("for_each mixed keys", Key, BS, capacity, n, params, Probe);

  auto fail = set.Insert(static_cast<void*>(dInsertKeys.Data()),
                         aclco::Extent<std::size_t>(hostKeys.size()),
                         stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("for_each with 2x key range (half existent, half non-existent)") {
    PRINT_SECTION("for_each with 2x key range");

    std::unordered_set<Key> keySet(hostKeys.begin(), hostKeys.end());
    std::size_t probeN = n * 2;
    std::vector<Key> probeKeys;
    probeKeys.reserve(probeN);
    for (auto k : hostKeys) {
      probeKeys.push_back(k);
    }
    Key candidate = static_cast<Key>(0);
    while (probeKeys.size() < probeN) {
      if (keySet.find(candidate) == keySet.end() && candidate != sent) {
        probeKeys.push_back(candidate);
      }
      ++candidate;
    }

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    set.template ForEach<CountEvenKeys<Key>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    std::size_t expectedEven = 0;
    for (auto k : hostKeys) {
      if (k % 2 == 0) { ++expectedEven; }
    }
    REQUIRE_PRINT(result[0] == expectedEven);
  }
}
