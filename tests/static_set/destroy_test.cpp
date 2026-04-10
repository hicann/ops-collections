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
  "static_set destroy test",
  "[static_set][destroy]",
  ((typename K, int BucketSize), K, BucketSize),
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
  auto setPtr = std::make_unique<aclco::test::set_factory::StaticSetT<Key, BS, aclco::test::set_factory::DoubleHashing<Key>, aclco::EqualTo<Key>>>(
      aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
          capacity, sent, stream));
  
  CAPTURE(setPtr->Capacity(), BS);
  PRINT_BEFORE_EXEC_SET("destroy test", Key, BS, setPtr->Capacity(), 0);   
  REQUIRE_PRINT(setPtr != nullptr);
  REQUIRE_PRINT(setPtr->Capacity() >= capacity);
  setPtr.reset();

  REQUIRE_PRINT(setPtr == nullptr);
}

TEMPLATE_TEST_CASE_SIG(
  "static_set extent storage destroy test",
  "[static_set][extentStorageDestroy]",
  ((typename K, int BucketSize), K, BucketSize),
  (uint32_t, 1),
  (int32_t, 5),
  (uint64_t, 1),
  (float, 5))
{
  aclco::test::AclGlobalGuard g_acl;
  aclco::test::AclStreamGuard sg;
  auto stream = sg.stream;

  using Key   = K;
  constexpr int BS = BucketSize;

  auto sent = aclco::test::MakeDefaultSentinels<Key>();

  std::size_t capacity = 159;

  auto setPtr = std::make_unique<aclco::test::set_factory::StaticSetT<Key, BS, aclco::test::set_factory::DoubleHashing<Key>, aclco::EqualTo<Key>>>(
      aclco::test::set_factory::MakeStaticSet<Key, BS, aclco::test::set_factory::DoubleHashing<Key>>(
          capacity, sent, stream));

  CAPTURE(setPtr->Capacity(), BS);
  PRINT_BEFORE_EXEC_SET("destroy test", Key, BS, setPtr->Capacity(), 0);   
  REQUIRE_PRINT(setPtr != nullptr);
  REQUIRE_PRINT(setPtr->Capacity() >= capacity);
  setPtr.reset();

  REQUIRE_PRINT(setPtr == nullptr);
}