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

template <typename Key, typename Value>
std::vector<Value> ExpectedFindIfValues(const std::vector<aclco::Pair<Key, Value>>& hostPairs,
                                        const std::vector<Key>& findKeys,
                                        const std::vector<uint32_t>& stencil,
                                        Value emptyValue)
{
  std::unordered_map<Key, Value> pairMap;
  pairMap.reserve(hostPairs.size());
  for (const auto& pair : hostPairs) {
    pairMap[pair.first] = pair.second;
  }

  std::vector<Value> out;
  out.reserve(findKeys.size());
  for (std::size_t i = 0; i < findKeys.size(); ++i) {
    if (stencil[i] % 2 == 0) {
      out.push_back(emptyValue);
    } else {
      auto it = pairMap.find(findKeys[i]);
      if (it != pairMap.end()) {
        out.push_back(it->second);
      } else {
        out.push_back(emptyValue);
      }
    }
  }
  return out;
}
} // namespace

TEMPLATE_TEST_CASE_SIG(
  "static_map find_if odd stencil",
  "[static_map][find_if]",
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
  PRINT_BEFORE_EXEC_WITH_PROBE("find_if odd stencil", Key, Value, BS, capacity, n, params, Probe);

  CAPTURE(seed, n, capacity, BS);

  auto hostPairs = aclco::test::MakeExamples<Key, Value>(seed, n, sent, "uniform", true);
  if (hostPairs.empty()) {
    SKIP("Can not create enough pairs!");
  }

  aclco::test::DeviceBuffer<Pair<Key, Value>> dPairs(hostPairs.size());
  dPairs.CopyFromHostAsync(hostPairs.data(), hostPairs.size(), stream);

  auto fail = map.Insert(static_cast<void*>(dPairs.Data()),
                         aclco::Extent<std::size_t>(hostPairs.size()),
                         stream);
  REQUIRE_PRINT(fail == 0);

  SECTION("find_if with stencil") {
    PRINT_SECTION("find_if with stencil");
    std::size_t findN = hostPairs.size();

    std::vector<Key> findKeys;
    findKeys.reserve(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      findKeys.push_back(hostPairs[i].first);
    }

    std::vector<uint32_t> hostStencil(findN);
    for (std::size_t i = 0; i < findN; ++i) {
      hostStencil[i] = static_cast<uint32_t>(i);
    }

    auto expected = ExpectedFindIfValues<Key, Value>(hostPairs, findKeys, hostStencil, sent.emptyValue);

    aclco::test::DeviceBuffer<Key> dKeys(findN);
    dKeys.CopyFromHostAsync(findKeys.data(), findN, stream);

    aclco::test::DeviceBuffer<uint32_t> dStencil(findN);
    dStencil.CopyFromHostAsync(hostStencil.data(), findN, stream);

    aclco::test::DeviceBuffer<Value> dOutputValues(findN);
    dOutputValues.MemsetZero(stream);

    map.template FindIf<uint32_t, IsOdd>(
      static_cast<void*>(dKeys.Data()), dStencil.Data(),
      static_cast<void*>(dOutputValues.Data()),
      aclco::Extent<std::size_t>(findN), stream);

    auto got = dOutputValues.CopyToHost(stream);
    REQUIRE_PRINT(got.size() == expected.size());

    for (std::size_t i = 0; i < got.size(); ++i) {
      CAPTURE(i, findKeys[i]);
      REQUIRE_PRINT(got[i] == expected[i]);
    }
  }
}
