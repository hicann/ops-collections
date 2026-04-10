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

namespace
{
template <typename K, typename V>
using Pair = aclco::Pair<K, V>;

template <typename T>
struct UnEqualTo
{
  COLLECTION_HOST_DEVICE constexpr bool operator()(T const& lhs, T const& rhs) const noexcept
  {
    return lhs != rhs;
  }
};
} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map default key equal",
  "[static_map][default_key_equal]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint64_t, uint64_t, 1),
  (uint64_t, uint64_t, 5),
  (float, float, 1),
  (float, float, 5)) 
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(10000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.5f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  //测试默认KeyEqual
  SECTION("default keyEqual insert") {
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE(fail == 0);

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key, Value>(hostPairs, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, golden, &diff);
    INFO(diff);
    REQUIRE(ok);
  }
}

  TEMPLATE_TEST_CASE_SIG(
  "static_map un equal key equal",
  "[static_map][un_equal_key_equal]",
  ((typename K, typename V, int BucketSize), K, V, BucketSize),
  (uint32_t, uint32_t, 1),
  (uint32_t, uint32_t, 5),
  (uint64_t, uint64_t, 1),
  (uint64_t, uint64_t, 5),
  (float, float, 1),
  (float, float, 5)) 
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  using Value = V;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key, Value>();

  std::size_t capacity = GENERATE(10000000u);

  auto map = aclco::test::map_factory::MakeStaticMap<Key, Value, BS, aclco::test::map_factory::LinearProbing<Key>, UnEqualTo<Key>>(
    capacity, sent, stream);

  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.5f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  //测试UnEqualTo KeyEqual
  SECTION("default unEqual keyEqual insert") {
    CAPTURE(seed, n, capacity, BS);
    auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);

    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    auto fail = map.Insert(static_cast<void*>(dPairs.Data()), 
                          aclco::Extent<std::size_t>(n), stream);
    REQUIRE(fail == n);

  }
}

