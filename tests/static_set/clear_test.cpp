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
#include <unordered_set>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/set_factory.h"
#include "tests/common/dump_table.h"
#include "tests/common/matchers.h"
#include "tests/common/generators.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE_SIG(
  "static_set clear correctness (sync Clear)",
  "[static_set][clear]",
  ((typename K,  int BucketSize), K, BucketSize),
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

  std::size_t capacity = 160000000;
  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("clear correctness", Key, BS, capacity, n, params);

  //不插入，直接清空
  SECTION("directly clear") {
    PRINT_SECTION("directly clear");
    set.Clear(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);

    std::unordered_set<Key> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
      // 插入后多次清空
  SECTION("multiple clear") {
    PRINT_SECTION("multiple clear");
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

    auto fail = set.Insert(static_cast<void*>(dKeys.Data()),
                          aclco::Extent<std::size_t>(hostKeys.size()),
                          stream);
    REQUIRE_PRINT(fail == 0);

    set.Clear(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);

    std::unordered_set<Key> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);

    set.Clear(stream);
    aclco::test::Sync(stream);

    bool ok2 = aclco::test::EqualAsSet<Key>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok2);
  }
}


TEMPLATE_TEST_CASE_SIG(
  "static_set clearAsync",
  "[static_set][clearAsync]",
  ((typename K,  int BucketSize), K, BucketSize),
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

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);
  CAPTURE(seed, n, capacity, BS);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PARAMS("clearAsync correctness", Key, BS, capacity, n, params);

  //不插入，直接清空
  SECTION("directly clear") {
    PRINT_SECTION("directly clear");
    set.ClearAsync(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);

    std::unordered_set<Key> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
    // 插入后多次清空
  SECTION("multiple clear") {
    PRINT_SECTION("multiple clear");
    auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
    if (hostKeys.empty()) {
      SKIP("Can not create enough keys, when pair number is bigger than the max exact integer of Key type!");
    }

    aclco::test::DeviceBuffer<Key> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

    auto fail = set.Insert(static_cast<void*>(dKeys.Data()),
                          aclco::Extent<std::size_t>(hostKeys.size()),
                          stream);
    REQUIRE_PRINT(fail == 0);

    set.ClearAsync(stream);
    aclco::test::Sync(stream);

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);

    std::unordered_set<Key> goldenEmpty;
    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);

    set.ClearAsync(stream);
    aclco::test::Sync(stream);

    bool ok2 = aclco::test::EqualAsSet<Key>(observed, goldenEmpty, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok2);
  }
}