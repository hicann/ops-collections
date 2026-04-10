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

#include <memory>

#include <acl/acl.h>

#include "detail/open_addressing/open_addressing_impl.h"
#include "extent.h"
#include "hash_functions.h"

namespace aclco {
static constexpr size_t defaultSetBucketSize = 5;

template <class Key,
          class Extent = Extent<size_t>,
          class KeyEqual = aclco::EqualTo<Key>,
          class ProbingScheme = aclco::DoubleHashing<aclco::xxhash_32<Key>>,
          class Storage = Storage<defaultSetBucketSize>>
class StaticSet {
  static_assert(sizeof(Key) <= 8, "Container dose not support key type size bigger than 8 bytes.");

  using ImplType = OpenAddressingImpl<Key,
                                      Key,
                                      Extent,
                                      KeyEqual,
                                      ProbingScheme,
                                      Storage>;

 public:
  static constexpr auto bucketSize = ImplType::bucketSize;

  using SizeType = typename ImplType::SizeType;
  using KeyType = typename ImplType::KeyType;
  using ValueType = typename ImplType::ValueType;

  StaticSet(StaticSet const&) = delete;
  StaticSet& operator=(StaticSet const&) = delete;

  StaticSet(StaticSet&&) = default;
  StaticSet& operator=(StaticSet &&) = default;
  ~StaticSet() = default;


  constexpr StaticSet(Extent capacity,
                      Key emptyKey,
                      KeyEqual const& pred = {},
                      ProbingScheme const& probingScheme = {},
                      Storage storage = {},
                      aclrtStream stream = nullptr);

  SizeType Insert(void *keys, Extent keyNum, aclrtStream stream);

  void InsertAsync(void *keys, Extent keyNum, aclrtStream stream);

  SizeType Erase(void *keys, Extent keyNum, aclrtStream stream);

  void EraseAsync(void *keys, Extent keyNum, aclrtStream stream);

  void Clear(aclrtStream stream);

  void ClearAsync(aclrtStream stream) noexcept;

  void Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);

  void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);
  
  void Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // outputValues中的元素为bool类型

  void ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // outputValues中的元素为bool类型
  
  constexpr auto Capacity() const noexcept;

  ValueType* Data() const;

 private:
  std::unique_ptr<ImplType> impl_;
  Key emptyKey_;
};
}


#include "detail/static_set/static_set.inl"