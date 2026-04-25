/*
 * Copyright (c) 2022-2025， NVIDIA CPRPORATION.
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "macros.h"

namespace aclco {
template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
COLLECTION_HOST_DEVICE constexpr StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::StaticMapRef(ValueType emptyValue,
                            KeyEqual const& predicate,
                            ProbingScheme const& probingScheme,
                            StorageRef storageRef) noexcept
  : impl_{emptyValue, predicate, probingScheme, storageRef}
{
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename Value>
COLLECTION_DEVICE bool StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::Insert(Value value) noexcept
{
  return impl_.Insert(value);
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename Value>
COLLECTION_DEVICE bool StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::InsertOrAssign(Value value) noexcept
{
  return impl_.InsertOrAssign(value);
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename ProbeKey>
COLLECTION_DEVICE bool StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::Erase(ProbeKey key) noexcept
{
  return impl_.Erase(key);
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename ProbeKey>
COLLECTION_DEVICE typename StaticMapRef<Key, KeyEqual, ProbingScheme, StorageRef>::PayloadType StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::Find(ProbeKey key) noexcept
{
  return impl_.Find(key);
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename ProbeKey>
COLLECTION_DEVICE bool StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::Contains(ProbeKey key) noexcept
{
  return impl_.Contains(key);
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename ProbeKey, typename CallbackOp>
COLLECTION_DEVICE void StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::ForEach(ProbeKey key, CallbackOp& callback_op) noexcept
{
  impl_.ForEach(key, callback_op);
}

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
template <typename ProbeKey>
COLLECTION_DEVICE typename StaticMapRef<Key, KeyEqual, ProbingScheme, StorageRef>::SizeType StaticMapRef<
  Key,
  KeyEqual,
  ProbingScheme,
  StorageRef>::Count(ProbeKey key) noexcept
{
  return impl_.Count(key);
}
}