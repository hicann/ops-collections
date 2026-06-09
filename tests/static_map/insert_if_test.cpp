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

struct IsOdd {
  COLLECTION_SIMT_DEVICE bool operator()(uint32_t val) const noexcept
  {
    return val % 2 != 0;
  }
};
} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map insert_if odd stencil",
  "[static_map][insert_if]",
  ((typename K, typename V, int BucketSize, typename ProbingScheme), K, V, BucketSize, ProbingScheme),
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
  PRINT_BEFORE_EXEC_WITH_PROBE("insert_if odd stencil", Key, Value, BS, capacity, n, params, Probe);

  CAPTURE(seed, n, capacity, BS);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
    SKIP("Can not create enough pairs!");
  }

  SECTION("insert_if with stencil") {
    PRINT_SECTION("insert_if with stencil");

    std::vector<uint32_t> hostStencil(n);
    for (std::size_t i = 0; i < n; ++i) {
      hostStencil[i] = static_cast<uint32_t>(i);
    }

    aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(n);
    dPairs.CopyFromHostAsync(hostPairs.data(), n, stream);

    aclco::test::DeviceBuffer<uint32_t> dStencil(n);
    dStencil.CopyFromHostAsync(hostStencil.data(), n, stream);

    auto fail = map.template InsertIf<uint32_t, IsOdd>(
      static_cast<void*>(dPairs.Data()), dStencil.Data(),
      aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);

    std::vector<Pair<Key, Value>> filteredPairs;
    for (std::size_t i = 0; i < n; ++i) {
      if (hostStencil[i] % 2 != 0) {
        filteredPairs.push_back(hostPairs[i]);
      }
    }

    auto observed = aclco::test::DumpTable<Key, Value>(map, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key, Value>(filteredPairs, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsMap<Key, Value>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
}
