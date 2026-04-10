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

#include <acl/acl.h>
#include <cstdint>
#include <vector>

#include "pair.h"
#include "bucket_storage.h"
#include "utility/allocator.h"

#include "../common/acl_env.h"

TEST_CASE("BucketStorage Initialize sets all values and capacity", "[storage]")
{
  aclco::test::AclGlobalGuard aclGlobal{};
  aclco::test::AclStreamGuard streamGuard{};

  using ValueType = aclco::Pair<uint32_t, uint32_t>;
  constexpr int32_t bucketSize = 5;
  constexpr std::uint32_t wholeSize = 20; // 4 buckets, 5 elements per bucket
  constexpr ValueType initValue = {3, 3};
  ValueType *initValuePtr;
  auto ret = aclrtMalloc((void**)&initValuePtr, sizeof(ValueType), ACL_MEM_MALLOC_HUGE_FIRST);
  REQUIRE(ret == ACL_SUCCESS);
  ret = aclrtMemcpy((void*)initValuePtr, sizeof(ValueType), (void*)&initValue, sizeof(ValueType), ACL_MEMCPY_HOST_TO_DEVICE);
  REQUIRE(ret == ACL_SUCCESS);

  aclco::DefaultAllocator<ValueType> allocator;

  auto bucketStorage =
    std::make_shared<aclco::BucketStorage<ValueType, bucketSize>>(wholeSize, allocator);

  bucketStorage->Initialize(initValuePtr, streamGuard.stream);

  auto const capacity = bucketStorage->Capacity();
  REQUIRE(capacity == wholeSize);

  std::vector<ValueType> host(capacity, {0, 0});
  ret = aclrtMemcpy(host.data(), sizeof(ValueType) * capacity,
                         bucketStorage->Data(), sizeof(ValueType) * capacity,
                         ACL_MEMCPY_DEVICE_TO_HOST);
  REQUIRE(ret == ACL_SUCCESS);

  for (std::uint32_t i = 0; i < capacity; ++i) {
    REQUIRE(host[i] == initValue);
  }
}