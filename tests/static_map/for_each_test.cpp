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
#include "tests/common/map_factory.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/test_print.h"
#include "hash_functions.h"

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;

template <typename Key, typename Value>
struct CountEvenKeyWithValueOne {
  __gm__ uint32_t *counter;

  CountEvenKeyWithValueOne() : counter{nullptr} {}
  COLLECTION_DEVICE CountEvenKeyWithValueOne(__gm__ uint8_t *state) : counter{reinterpret_cast<__gm__ uint32_t*>(state)} {}

  template <typename SlotKey, typename SlotValue>
  COLLECTION_DEVICE void operator()(Pair<SlotKey, SlotValue> slot) const noexcept
  {
    if (slot.first % 2 == 0 && slot.second == 1) {
      AscendC::Simt::AtomicAdd(counter, 1u);
    }
  }
};

template <typename Key, typename Value>
struct CountAll {
  __gm__ uint32_t *counter;

  CountAll() : counter{nullptr} {}
  COLLECTION_DEVICE CountAll(__gm__ uint8_t *state) : counter{reinterpret_cast<__gm__ uint32_t*>(state)} {}

  template <typename SlotKey, typename SlotValue>
  COLLECTION_DEVICE void operator()(Pair<SlotKey, SlotValue> slot) const noexcept
  {
    AscendC::Simt::AtomicAdd(counter, 1u);
  }
};
}

TEMPLATE_TEST_CASE_SIG(
  "static_map for_each count even keys with value 1",
  "[static_map][for_each]",
  ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize, ProbingScheme),
  (uint32_t, uint32_t, 1, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::DoubleHashing<uint32_t>),
  (uint64_t, uint64_t, 1, aclco::test::map_factory::LinearProbing<uint64_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::LinearProbing<uint64_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::DoubleHashing<uint64_t>),
  (aclco::fp16_t, aclco::fp16_t, 1, aclco::test::map_factory::LinearProbing<aclco::fp16_t>),
  (aclco::fp16_t, aclco::fp16_t, 5, aclco::test::map_factory::DoubleHashing<aclco::fp16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(164u, 100000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  CAPTURE(seed, n, capacity, BS);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("for_each count even keys with value 1", Key, Value, BS, capacity, n, params, Probe);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                         aclco::Extent<std::size_t>(hostPairs.size()),
                         stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("for_each counts even keys with value 1") {
    PRINT_SECTION("for_each counts even keys with value 1");

    std::vector<Key> probeKeys;
    probeKeys.reserve(hostPairs.size());
    for (auto const& p : hostPairs) {
      probeKeys.push_back(p.first);
    }

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    map.template ForEach<CountEvenKeyWithValueOne<Key, Value>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    std::size_t expected = 0;
    for (auto const& p : hostPairs) {
      if (p.first % 2 == 0 && p.second == 1) { ++expected; }
    }
    REQUIRE_PRINT(result[0] == expected);
  }

  SECTION("for_each counts all matching slots") {
    PRINT_SECTION("for_each counts all matching slots");

    std::vector<Key> probeKeys;
    probeKeys.reserve(hostPairs.size());
    for (auto const& p : hostPairs) {
      probeKeys.push_back(p.first);
    }

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    map.template ForEach<CountAll<Key, Value>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    REQUIRE_PRINT(result[0] == hostPairs.size());
  }

  SECTION("for_each with non-existent keys") {
    PRINT_SECTION("for_each with non-existent keys");

    std::unordered_set<Key> keySet;
    keySet.reserve(hostPairs.size());
    for (auto const& p : hostPairs) {
      keySet.insert(p.first);
    }
    std::size_t probeN = n;
    std::vector<Key> probeKeys;
    probeKeys.reserve(probeN);
    Key candidate = static_cast<Key>(0);
    while (probeKeys.size() < probeN) {
      if (keySet.find(candidate) == keySet.end() && candidate != sent.emptyKey) {
        probeKeys.push_back(candidate);
      }
      ++candidate;
    }

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    map.template ForEach<CountAll<Key, Value>>(
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
  "static_map for_each with mixed existent and non-existent keys",
  "[static_map][for_each]",
  ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize, ProbingScheme),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::DoubleHashing<uint32_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::LinearProbing<uint64_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::DoubleHashing<uint64_t>),
  (aclco::fp16_t, aclco::fp16_t, 1, aclco::test::map_factory::LinearProbing<aclco::fp16_t>),
  (aclco::fp16_t, aclco::fp16_t, 5, aclco::test::map_factory::DoubleHashing<aclco::fp16_t>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(164u, 100000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.5f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  if(n == 0) {
      SKIP("skip n == 0");
  }
  CAPTURE(seed, n, capacity, BS);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
      SKIP("Can not create enough keys, when key number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("for_each mixed keys", Key, Value, BS, capacity, n, params, Probe);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                         aclco::Extent<std::size_t>(hostPairs.size()),
                         stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("for_each with 2x key range (half existent, half non-existent)") {
    PRINT_SECTION("for_each with 2x key range");

    std::unordered_set<Key> keySet;
    keySet.reserve(hostPairs.size());
    for (auto const& p : hostPairs) {
      keySet.insert(p.first);
    }
    std::size_t probeN = n * 2;
    std::vector<Key> probeKeys;
    probeKeys.reserve(probeN);
    for (auto const& p : hostPairs) {
      probeKeys.push_back(p.first);
    }
    Key candidate = static_cast<Key>(0);
    while (probeKeys.size() < probeN) {
      if (keySet.find(candidate) == keySet.end() && candidate != sent.emptyKey) {
        probeKeys.push_back(candidate);
      }
      ++candidate;
    }

    aclco::test::DeviceBuffer<Key> dKeys(probeKeys.size());
    dKeys.CopyFromHostAsync(probeKeys.data(), probeKeys.size(), stream);

    aclco::test::DeviceBuffer<uint32_t> dCounter(1);
    dCounter.MemsetZero(stream);

    map.template ForEach<CountEvenKeyWithValueOne<Key, Value>>(
      static_cast<void*>(dKeys.Data()),
      aclco::Extent<std::size_t>(probeKeys.size()),
      static_cast<void*>(dCounter.Data()),
      stream);

    auto result = dCounter.CopyToHost(stream);
    REQUIRE_PRINT(result.size() == 1);
    std::size_t expected = 0;
    for (auto const& p : hostPairs) {
      if (p.first % 2 == 0 && p.second == 1) { ++expected; }
    }
    REQUIRE_PRINT(result[0] == expected);
  }
}
