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
struct IsMod3One {
  COLLECTION_SIMT_DEVICE bool operator()(uint32_t val) const noexcept
  {
    return val % 3 == 1;
  }
};

template <typename Key>
std::vector<bool> ExpectedContainsIfKeys(const std::vector<Key>& hostKeys,
                                         const std::vector<Key>& findKeys,
                                         const std::vector<uint32_t>& stencil)
{
  std::unordered_set<Key> keySet;
  keySet.reserve(hostKeys.size());
  for (const auto& key : hostKeys) {
    keySet.insert(key);
  }

  std::vector<bool> out;
  out.reserve(findKeys.size());
  for (std::size_t i = 0; i < findKeys.size(); ++i) {
    if (stencil[i] % 3 != 1) {
      out.push_back(false);
    } else {
      out.push_back(keySet.find(findKeys[i]) != keySet.end());
    }
  }
  return out;
}
}

TEMPLATE_TEST_CASE_SIG(
  "static_set contains_if mod3 one stencil",
  "[static_set][contains_if]",
  ((typename K, int BucketSize, typename ProbingScheme), K, BucketSize, ProbingScheme),
  (uint32_t, 1, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::LinearProbing<uint32_t>),
  (uint32_t, 5, aclco::test::set_factory::DoubleHashing<uint32_t>),
  (uint64_t, 1, aclco::test::set_factory::LinearProbing<uint64_t>),
  (uint64_t, 5, aclco::test::set_factory::LinearProbing<uint64_t>),
  (float, 1, aclco::test::set_factory::LinearProbing<float>),
  (float, 5, aclco::test::set_factory::LinearProbing<float>),
  (float, 5, aclco::test::set_factory::DoubleHashing<float>),
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

  std::size_t capacity = GENERATE(100u, 1000000u);

  auto set = aclco::test::set_factory::MakeStaticSet<Key, BS, Probe>(
    capacity, sent, stream);

  capacity = set.Capacity();
  auto seed = GENERATE(1u);
  auto ratio = GENERATE(0.1f, 0.5f, 0.9f);
  auto n = static_cast<std::size_t>(ratio * capacity);

  std::string params = "seed=" + std::to_string(seed) + ", ratio=" + std::to_string(ratio);
  PRINT_BEFORE_EXEC_SET_WITH_PROBE("contains_if mod3 one stencil", Key, BS, capacity, n, params, Probe);

  CAPTURE(seed, n, capacity, BS);

  auto hostKeys = aclco::test::MakeExamples<Key>(seed, n, sent, "uniform", true);
  if (hostKeys.empty()) {
    SKIP("Can not create enough keys, when number is bigger than the max exact integer of Key type!");
  }

  aclco::test::DeviceBuffer<Key> dInsertKeys(hostKeys.size());
  dInsertKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

  auto fail = set.Insert(static_cast<void*>(dInsertKeys.Data()),
                         aclco::Extent<std::size_t>(hostKeys.size()),
                         stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("contains_if with stencil") {
    PRINT_SECTION("contains_if with stencil");
    std::size_t findN = hostKeys.size();

    std::vector<uint32_t> hostStencil(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      hostStencil[i] = static_cast<uint32_t>(i);
    }

    auto expected = ExpectedContainsIfKeys<Key>(hostKeys, hostKeys, hostStencil);

    aclco::test::DeviceBuffer<Key> dFindKeys(findN);
    dFindKeys.CopyFromHostAsync(hostKeys.data(), findN, stream);

    aclco::test::DeviceBuffer<uint32_t> dStencil(findN);
    dStencil.CopyFromHostAsync(hostStencil.data(), findN, stream);

    aclco::test::DeviceBuffer<unsigned char> dOutputBools(findN);
    dOutputBools.MemsetZero(stream);

    set.template ContainsIf<uint32_t, IsMod3One>(
      static_cast<void*>(dFindKeys.Data()), dStencil.Data(),
      static_cast<void*>(dOutputBools.Data()),
      aclco::Extent<std::size_t>(findN), stream);

    auto got = dOutputBools.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, hostKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }
}
