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
#include "hash_functions.h"

namespace
{
struct IsMod3Zero {
  COLLECTION_DEVICE bool operator()(uint32_t val) const noexcept
  {
    return val % 3 == 0;
  }
};
}

TEMPLATE_TEST_CASE_SIG(
  "static_set insert_if mod3 zero stencil",
  "[static_set][insert_if]",
  ((typename K, int BucketSize, typename ProbingScheme), K, BucketSize, ProbingScheme),
  (uint32_t, 1, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::LinearProbing<uint64_t>),
  (uint64_t, 5, aclco::test::set_factory::LinearProbing<uint64_t>),
  (float, 1, aclco::test::set_factory::LinearProbing<float>),
  (float, 5, aclco::test::set_factory::LinearProbing<float>),
  (float, 5, aclco::test::set_factory::DoubleHashing<float>))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;
  using Probe = ProbingScheme;
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

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("insert_if mod3 zero stencil", Key, BS, capacity, n, params, Probe);

  CAPTURE(seed, n, capacity, BS);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
    SKIP("Can not create enough keys, when number is bigger than the max exact integer of Key type!");
  }

  SECTION("insert_if with stencil") {
    PRINT_SECTION("insert_if with stencil");

    std::vector<uint32_t> hostStencil(n);
    for (std::size_t i = 0; i < n; ++i) {
      hostStencil[i] = static_cast<uint32_t>(i);
    }

    aclco::test::DeviceBuffer<Key> dKeys(n);
    dKeys.CopyFromHostAsync(hostKeys.data(), n, stream);

    aclco::test::DeviceBuffer<uint32_t> dStencil(n);
    dStencil.CopyFromHostAsync(hostStencil.data(), n, stream);

    auto fail = set.template InsertIf<uint32_t, IsMod3Zero>(
      static_cast<void*>(dKeys.Data()), dStencil.Data(),
      aclco::Extent<std::size_t>(n), stream);
    REQUIRE_PRINT(fail == 0);

    std::vector<Key> filteredKeys;
    for (std::size_t i = 0; i < n; ++i) {
      if (hostStencil[i] % 3 == 0) {
        filteredKeys.push_back(hostKeys[i]);
      }
    }

    auto observed = aclco::test::DumpTable<Key>(set, sent, stream);
    auto golden   = aclco::test::GoldenInsert<Key>(filteredKeys, sent);

    std::string diff;
    bool ok = aclco::test::EqualAsSet<Key>(observed, golden, &diff);
    INFO(diff);
    REQUIRE_PRINT(ok);
  }
}
