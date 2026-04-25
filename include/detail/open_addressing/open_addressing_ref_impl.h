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

#include "kernel_operator.h"
#include "macros.h"
#include "detail/storages/bucket_storage_ref.h"
#include "probing_scheme.h"
#include "utility/equal_wrapper.h"
#include "utility/is_same.h"
#include "utility/conditional.h"

namespace aclco {
enum class InsertResult : int32_t {FAILED, SUCCESS, DUPLICATE};

template <typename Key,
          typename KeyEqual,
          typename ProbingScheme,
          typename StorageRef>
class OpenAddressingRefImpl {
 public:
  using KeyType = Key;
  using ProbingSchemeType = ProbingScheme;
  using Hasher = typename ProbingSchemeType::Hasher;
  using StorageRefType = StorageRef;
  using ValueType = typename StorageRefType::ValueType;
  using SizeType = typename StorageRefType::SizeType;
  using Iterator = typename StorageRefType::Iterator;
  using ConstIterator = typename StorageRefType::ConstIterator;

  static constexpr auto bucketSize = StorageRefType::bucketSize;
  static constexpr bool hasPayload = isPairV<ValueType>;

  COLLECTION_HOST_DEVICE explicit constexpr OpenAddressingRefImpl(
    ValueType emptySlotValue,
    KeyEqual const& predicate,
    ProbingSchemeType const& probingScheme,
    StorageRefType storageRef) noexcept
    : emptySlotValue_{emptySlotValue},
      predicate_{predicate},
      probingScheme_{probingScheme},
      storageRef_{storageRef}
    {
    }

  template <typename Value>
  COLLECTION_HOST_DEVICE constexpr auto ExtractKey(Value value) const noexcept
  {
    if constexpr (hasPayload) {
      return value.first;
    } else {
      return value;
    }
  }

  template <typename Value>
  COLLECTION_DEVICE constexpr auto ExtractValue(Value value) const noexcept
  {
    if constexpr (hasPayload) {
      return value.second;
    } else {
      return value;
    }
  }

  template <typename Value>
  COLLECTION_HOST_DEVICE constexpr auto ExtractPayload(Value value) const noexcept
  {
    return value.second;
  }

  template <typename Value>
  COLLECTION_DEVICE InsertResult BackToBackCas(__gm__ Value *address,
                                                     Value expected,
                                                     Value desire) noexcept
  {
    using KeyType = decltype(expected.first);
    using MappedType = decltype(expected.second);

    auto expectedKey = expected.first;
    auto expectedPayload = expected.second;

    auto oldKey = AscendC::Simt::AtomicCas((__gm__ KeyType*)&(address->first), expectedKey, desire.first);

    if (this->predicate_.EqualTo(oldKey, expectedKey) == EqualResult::EQUAL) {
      auto oldValue = AscendC::Simt::AtomicCas((__gm__ MappedType*)&(address->second), expectedPayload, desire.second);
      return InsertResult::SUCCESS;
    } else if (this->predicate_.EqualTo(oldKey, desire.first) == EqualResult::EQUAL) {
      return InsertResult::DUPLICATE;
    }

    return InsertResult::FAILED;
  }

  template <typename Value>
  COLLECTION_DEVICE InsertResult PackCas(__gm__ Value *address,
                                               Value expected,
                                               Value desired) noexcept
  {
    using PackedType = typename Conditional<sizeof(Value) == 4, uint32_t, uint64_t>::Type;

    __gm__ auto *slotPtr = reinterpret_cast<__gm__ PackedType*>(address);
    auto *expectedPtr = reinterpret_cast<PackedType*>(&expected);
    auto *desirePtr = reinterpret_cast<PackedType*>(&desired);

    auto oldSlot = AscendC::Simt::AtomicCas(slotPtr, *expectedPtr, *desirePtr);
    auto *oldSlotPtr = reinterpret_cast<Value *>(&oldSlot);

    if (this->predicate_.EqualTo(this->ExtractKey(expected), this->ExtractKey(*oldSlotPtr)) == EqualResult::EQUAL) {
      return InsertResult::SUCCESS;
    } else {
      if (this->predicate_.EqualTo(this->ExtractKey(desired), this->ExtractKey(*oldSlotPtr)) == EqualResult::EQUAL) {
        return InsertResult::DUPLICATE;
      }
      return InsertResult::FAILED;
    }
  }

  template <typename Value>
  COLLECTION_DEVICE InsertResult CasDependentWrite(__gm__ Value *address,
                                                         Value expected,
                                                         Value desired) noexcept
  {
    using MappedType = std::decay_t<decltype(this->ExtractPayload(emptySlotValue_))>;

    __gm__ auto *keyAddr = reinterpret_cast<__gm__ KeyType*>(&(address->first));
    auto expectedKey = expected.first;
    auto desiredKey = desired.first;
    auto oldKey = AscendC::Simt::AtomicCas(keyAddr, expectedKey, desiredKey);

    if (oldKey == expectedKey) {
      auto const desiredValue = desired.second;
      address->second = desiredValue;
      return InsertResult::SUCCESS;
    }

    if (this->predicate_.EqualTo(desiredKey, expectedKey) == EqualResult::EQUAL) {
      return InsertResult::DUPLICATE;
    }

    return InsertResult::FAILED;
  }

  template <typename Value>
  COLLECTION_DEVICE InsertResult AttemptInsert(__gm__ Value *address,
                                           Value expected,
                                           Value desired) noexcept
  {
    if constexpr (sizeof(Value) <= 8) {
      return PackCas(address, expected, desired);
    } else {
      return BackToBackCas(address, expected, desired);
    }
  }

  template <typename Value>
  COLLECTION_DEVICE InsertResult AttemptInsertStable(__gm__ Value *address,
                                                 Value expected,
                                                 Value desired) noexcept
  {
    if constexpr (sizeof(Value) <= 8) {
      return PackCas(address, expected, desired);
    } else {
      return CasDependentWrite(address, expected, desired);
    }
  }

  template <typename Value>
  COLLECTION_DEVICE InsertResult AttemptInsertOrAssign(__gm__ Value *address,
                                                            Value expected,
                                                            Value desired) noexcept
  {
    __gm__ auto *keyAddr = reinterpret_cast<__gm__ KeyType*>(&(address->first));
    auto expectedKey = expected.first;
    auto desiredKey = desired.first;
    auto oldKey = AscendC::Simt::AtomicCas(keyAddr, expectedKey, desiredKey);

    if (oldKey == expectedKey) {
      address->second = desired.second;
      return InsertResult::SUCCESS;
    }

    if (this->predicate_.EqualTo(oldKey, desiredKey) == EqualResult::EQUAL) {
      address->second = desired.second;
      return InsertResult::SUCCESS;
    }

    return InsertResult::FAILED;
  }

  template <typename Value>
  COLLECTION_DEVICE bool Insert(Value value)
  {
    __gm__ Value *tableHandle = storageRef_.Data();
    SizeType tableSize = storageRef_.Capacity();
    auto const key = this->ExtractKey(value);

    auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, tableSize);
    auto const initIdx = *probingIter;

    while (true) {
      __gm__ Value *bucketSlotsAddr = tableHandle + *probingIter;

      for (size_t i = 0; i < bucketSize; i++) {
        auto const slotContent = *(bucketSlotsAddr + i);
        EqualResult insertFlag = predicate_.template operator()<IsInsert::YES>(
          key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_));
        if (insertFlag == EqualResult::EQUAL) {
          return false;
        }
        if (insertFlag == EqualResult::AVAILABLE) {
          InsertResult result = AttemptInsert(bucketSlotsAddr + i, emptySlotValue_, value); // expectedValue是否要为empty?
          if (result == InsertResult::SUCCESS) {
            return true;
          } else if (result == InsertResult::DUPLICATE) {
            return false;
          }
          continue;
        }
      }
      ++probingIter;
      if (*probingIter == initIdx) { return false; }
    }
  }

  template <typename Value>
  COLLECTION_DEVICE bool InsertOrAssign(Value value)
  {
    __gm__ Value *tableHandle = storageRef_.Data();
    SizeType tableSize = storageRef_.Capacity();
    auto const key = this->ExtractKey(value);

    auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, tableSize);
    auto const initIdx = *probingIter;

    while (true) {
      __gm__ Value *bucketSlotsAddr = tableHandle + *probingIter;

      for (size_t i = 0; i < bucketSize; i++) {
        auto const slotContent = *(bucketSlotsAddr + i);
        EqualResult insertFlag = predicate_.template operator()<IsInsert::YES>(
          key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_));
        if (insertFlag == EqualResult::EQUAL) {
          (bucketSlotsAddr + i)->second = value.second;
          return true;
        }
        if (insertFlag == EqualResult::AVAILABLE) {
          InsertResult result = AttemptInsertOrAssign(bucketSlotsAddr + i, emptySlotValue_, value);
          if (result == InsertResult::SUCCESS) {
            return true;
          }
          continue;
        }
      }
      ++probingIter;
      if (*probingIter == initIdx) { return false; }
    }
  }

  template <typename ProbeKey>
  COLLECTION_DEVICE bool Erase(ProbeKey key) noexcept
  {
    __gm__ auto *tableHandle = storageRef_.Data();

    auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
    auto const initIdx = *probingIter;

    while (true) {
      __gm__ auto *bucketSlotsAddr = tableHandle + *probingIter;

      for (size_t i = 0; i < bucketSize; i++) {
        auto const slotContent = *(bucketSlotsAddr + i);
        EqualResult eraseFlag = predicate_.template operator()<IsInsert::NO>(
          key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_));

        if (eraseFlag == EqualResult::EQUAL) {
          InsertResult result = AttemptInsertStable(bucketSlotsAddr + i, slotContent, emptySlotValue_);
          switch(result) {
            case InsertResult::SUCCESS: return true;
            case InsertResult::DUPLICATE: return false;
            default :continue;
          }
        }
      }
      ++probingIter;
      if (*probingIter == initIdx) { return false; }
    }
  }

  template <typename ProbeKey>
  COLLECTION_DEVICE auto Find(ProbeKey key) noexcept
  {
    __gm__ auto *tableHandle = storageRef_.Data();

    auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
    auto const initIdx = *probingIter;

    while (true) {
      __gm__ auto *bucketSlotsAddr = tableHandle + *probingIter;
      // find 函数
      for (auto i = 0; i < bucketSize; i++) {
        auto const slotContent = *(bucketSlotsAddr + i);
        switch (EqualResult findFlag = predicate_.template operator()<IsInsert::NO>(
          key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_))) {
            case EqualResult::EMPTY: {
              // 找不到返回0
              return ExtractValue(emptySlotValue_);
            }
            case EqualResult::EQUAL: {
              // 找到了返回对应的value
              // 需要填写
              return ExtractValue(slotContent);
            }
            default: continue; // 找不到则继续
          }
      }
      ++probingIter;
      // 没找到返回0
      if (*probingIter == initIdx) { 
        return ExtractValue(emptySlotValue_);
      }
    }
  }

  template <typename ProbeKey>
  COLLECTION_DEVICE bool Contains(ProbeKey key) noexcept
  {
    __gm__ auto *tableHandle = storageRef_.Data();

    auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
    auto const initIdx = *probingIter;

    while (true) {
      __gm__ auto *bucketSlotsAddr = tableHandle + *probingIter;
      // contains 函数
      for (auto i = 0; i < bucketSize; i++) {
        auto const slotContent = *(bucketSlotsAddr + i);
        switch (EqualResult containsFlag = predicate_.template operator()<IsInsert::NO>(
          key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_))) {
            case EqualResult::EMPTY: {
              return false;
            }
            case EqualResult::EQUAL: {
              return true;
            }
            default: continue; // 找不到则继续
          }
      }
      ++probingIter;
      if (*probingIter == initIdx) { return false; }
    }
  }

  template <typename ProbeKey, typename CallbackOp>
  COLLECTION_DEVICE void ForEach(ProbeKey key, CallbackOp& callback_op) noexcept
  {
    __gm__ auto *tableHandle = storageRef_.Data();

    auto probingIter = probingScheme_.template MakeIterator<bucketSize>(key, storageRef_.Capacity());
    auto const initIdx = *probingIter;

    while (true) {
      __gm__ auto *bucketSlotsAddr = tableHandle + *probingIter;

      for (size_t i = 0; i < bucketSize; i++) {
        auto const slotContent = *(bucketSlotsAddr + i);
        switch (EqualResult forEachFlag = predicate_.template operator()<IsInsert::NO>(
          key, this->ExtractKey(slotContent), this->ExtractKey(emptySlotValue_))) {
            case EqualResult::EMPTY: {
              return;
            }
            case EqualResult::EQUAL: {
              callback_op(slotContent);
              break;
            }
            default: continue;
          }
      }
      ++probingIter;
      if (*probingIter == initIdx) { return; }
    }
  }

  template <typename ProbeKey>
  COLLECTION_DEVICE SizeType Count(ProbeKey key) noexcept
  {
    return static_cast<SizeType>(this->Contains(key));
  }

  ValueType emptySlotValue_;
  EqualWrapper<KeyType, KeyEqual> predicate_;
  ProbingSchemeType probingScheme_;
  StorageRefType storageRef_;
};
}