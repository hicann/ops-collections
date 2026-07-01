/*
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
#include "probing_scheme.h"
#include "detail/open_addressing/open_addressing_ref_impl.h"

namespace aclco {
template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>

class StaticSetRef {
  using ImplType = OpenAddressingRefImpl<Key, KeyEqual, ProbingScheme, StorageRef>;

 public:
  using KeyType = Key;
  using ValueType = typename ImplType::ValueType;
  using PayloadType = ValueType;
  using SizeType = typename ImplType::SizeType;
  COLLECTION_HOST_DEVICE explicit constexpr StaticSetRef(Key emptyKey,
                                                         KeyEqual const& predicate,
                                                         ProbingScheme const& probingScheme,
                                                         StorageRef storageRef) noexcept;

  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE bool Insert(ProbeKey key) noexcept;

  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE Pair<PayloadType, bool> InsertAndFind(ProbeKey key) noexcept;  

  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE bool Erase(ProbeKey key) noexcept;

  /** @brief 墓碑删除重载：删除时写入 erasedSlotValue 而非空值。 */
  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE bool Erase(ProbeKey key, ValueType erasedSlotValue) noexcept;

  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE PayloadType Find(ProbeKey key) noexcept;

  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE bool Contains(ProbeKey key) noexcept;

  template <typename ProbeKey, typename CallbackOp>
  COLLECTION_SIMT_DEVICE void ForEach(ProbeKey key, CallbackOp& callback_op) noexcept;

  template <typename ProbeKey>
  COLLECTION_SIMT_DEVICE SizeType Count(ProbeKey key) noexcept;

 private:
  ImplType impl_;
};
}

#include "detail/static_set/static_set_ref.inl"