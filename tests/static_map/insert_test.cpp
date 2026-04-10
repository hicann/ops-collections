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
#include "tests/common/map_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/golden.h"
#include "tests/common/matchers.h"
#include "tests/common/test_print.h"
#include "hash_functions.h"
namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;
} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map insert correctness",
  "[static_map][insert]",
  ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize, ProbingScheme),
  (uint32_t, uint32_t, 1, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::LinearProbing<uint32_t>),
  (uint32_t, uint32_t, 5, aclco::test::map_factory::DoubleHashing<uint32_t>),
  (uint64_t, uint64_t, 1, aclco::test::map_factory::LinearProbing<uint64_t>),
  (uint64_t, uint64_t, 5, aclco::test::map_factory::LinearProbing<uint64_t>),
  (float, float, 1, aclco::test::map_factory::LinearProbing<float>),
  (float, float, 5, aclco::test::map_factory::LinearProbing<float>),
  (float, float, 5, aclco::test::map_factory::DoubleHashing<float>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, Probe>(
    capacity, sent, stream);

  capacity = map.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_WITH_PROBE("insert correctness", Key, Value, BS, capacity, n, params, Probe);

  SECTION("normal insert unique pairs") {
    PRINT_SECTION("normal insert unique pairs");
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }
    
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }

  SECTION("insert after clear") {
    PRINT_SECTION("insert after clear");
    CAPTURE(seed, n, capacity, BS);
    map.Clear(stream);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);
    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
}

TEMPLATE_TEST_CASE_SIG(
  "static_map insert negative tests",
  "[static_map][insert]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint32_t, uint64_t, 5),
  (uint64_t, uint32_t, 1),
  (float, float, 1),
  (uint8_t, uint32_t, 1),
  (int8_t, int32_t, 1),
  (uint64_t, float, 1)) 
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

  std::string params = "seed=" + std::to_string(seed);
  PRINT_BEFORE_EXEC_WITH_PARAMS("insert negative tests", Key, Value, BS, capacity, 0, params);

  SECTION("test n > capacity") {
    PRINT_SECTION("test n > capacity");
    std::size_t n = capacity + BS;
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n - map.Capacity());
  }

  SECTION("insert duplicate keys") {
    PRINT_SECTION("insert duplicate keys");
    std::size_t n = static_cast<std::size_t>(0.2 * capacity);
    std::size_t uniqueN = static_cast<std::size_t>(0.1 * capacity);
    auto hostPairs = aclco::test::MakeExamplesWithDuplicates<Key, Value>(seed, n, uniqueN, sent);
    if (hostPairs.empty()) {
      SKIP("Can not create enough pairs, when pair number is bigger than the max exact integer of Key type!");
    }

    CAPTURE(seed, n, capacity, BS);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n - uniqueN);
  }

  SECTION("insert nullptr") {
    PRINT_SECTION("insert nullptr");
    std::size_t n = static_cast<std::size_t>(0.1 * capacity);
    CAPTURE(seed, n, capacity, BS);
    auto fail = map.Insert(nullptr, 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == n);
  }

  SECTION("test n == 0") {
    PRINT_SECTION("test n == 0");
    std::size_t n = 0;
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);
    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);
  }
}