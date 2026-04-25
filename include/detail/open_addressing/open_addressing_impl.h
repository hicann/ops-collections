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

#include "bucket_storage.h"
#include "detail/storages/arg_storage.h"
#include "kernels.h"
#include "probing_scheme.h"
#include "storage.h"
#include "tiling/platform/platform_ascendc.h"
#include "utility/allocator.h"
#include "utility/kernel_launch_utils.h"
#include "utility/is_same.h"

namespace aclco {
struct AlwaysTrue {
  template <typename T>
  COLLECTION_DEVICE bool operator()(T) const noexcept
  {
    return true;
  }
};

template <class Key,
          class Value,
          class Extent,
          class KeyEqual,
          class ProbingScheme,
          class Storage>
class OpenAddressingImpl {
 public:
  static constexpr auto bucketSize = Storage::bucketSize;
  using KeyType = Key;
  using ValueType = Value;
  using SizeType = typename Extent::ValueType;
  using ProbingSchemeType = ProbingScheme;
  using Hasher = typename ProbingSchemeType::Hasher;
  using StorageType = BucketStorage<ValueType, bucketSize, SizeType>;

  constexpr OpenAddressingImpl(Extent capacity,
                              Value emptyValue,
                              KeyEqual const& pred,
                              ProbingScheme const& probingScheme,
                              aclrtStream stream)
    : probingScheme_{probingScheme},
      storage_{MakeValidExtent<ProbingScheme, Storage>(capacity), DefaultAllocator<ValueType>()},
      emptyValue_{emptyValue},
      predicate_{pred}
  {
    emptyValueStorage_.Initialize(emptyValue, stream);
    this->Clear(stream);
  }

  SizeType Insert(void *values, Extent valueNum, aclrtStream stream)
  {
    return InsertIf<ValueType, AlwaysTrue>(values, values, valueNum, stream);
  }


  void InsertAsync(void *values, Extent valueNum, aclrtStream stream)
  {
    InsertIfAsync<ValueType, AlwaysTrue>(values, values, valueNum, stream);
  }

  SizeType InsertOrAssign(void *values, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return 0; }
    if (values == nullptr) { return valueNum; }
    argStorage_.Initialize(0, stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertOrAssign<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum,
      (uint8_t*)argStorage_.Data());

    uint32_t ret = aclrtSynchronizeStream(stream);
    CheckRet(ret, "StaticMap::InsertOrAssign::aclrtSynchronizeStream");
    return argStorage_.LoadToHost(stream);
  }

  void InsertOrAssignAsync(void *values, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertOrAssignAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum);
  }

  template <typename StencilT, typename Predicate>
  SizeType InsertIf(void *values, void *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return 0; }
    if (values == nullptr) { return valueNum; }
    if (stencil == nullptr) { return valueNum; }
    argStorage_.Initialize(0, stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum,
      (uint8_t*)argStorage_.Data());

    uint32_t ret = aclrtSynchronizeStream(stream);
    if (isPairV<ValueType>) {
      CheckRet(ret, "StaticMap::InsertIf::aclrtSynchronizeStream");
    }
    else {
      CheckRet(ret, "StaticSet::InsertIf::aclrtSynchronizeStream");
    }
    return argStorage_.LoadToHost(stream);
  }

  template <typename StencilT, typename Predicate>
  void InsertIfAsync(void *values, void *stencil, Extent valueNum, aclrtStream stream)
  {
    if (valueNum == 0) { return; }
    if (values == nullptr) { return; }
    if (stencil == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::InsertIfAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)values,
      (uint8_t*)stencil,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      valueNum);
  }

  SizeType Erase(void *keys, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return 0; }
    if (keys == nullptr) { return keyNum; }
    argStorage_.Initialize(0, stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::Erase<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum,	 
      (uint8_t*)argStorage_.Data());

    uint32_t ret = aclrtSynchronizeStream(stream); 
    if (isPairV<ValueType>) {
      CheckRet(ret, "StaticMap::Erase::aclrtSynchronizeStream"); 
    }
    else {
      CheckRet(ret, "StaticSet::Erase::aclrtSynchronizeStream");     
    }
    return argStorage_.LoadToHost(stream);
  }

  void EraseAsync(void *keys, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::EraseAsync<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
  }

  template <typename StencilT, typename Predicate>
  void FindIfAsync(void *keys, void *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    if (stencil == nullptr) { return; }
    if (outputValues == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::FindIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)stencil,
      (uint8_t*)outputValues,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
  }

  void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    FindIfAsync<KeyType, AlwaysTrue>(keys, keys, outputValues, keyNum, stream);
  }

  template <typename StencilT, typename Predicate>
  void ContainsIfAsync(void *keys, void *stencil, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    if (stencil == nullptr) { return; }
    if (outputValues == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::ContainsIf<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, StencilT, Predicate><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)stencil,
      (uint8_t*)outputValues,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
  }

  void ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream)
  {
    ContainsIfAsync<KeyType, AlwaysTrue>(keys, keys, outputValues, keyNum, stream);
  }

  template <typename CallbackOp>
  void ForEachAsync(void *keys, Extent keyNum, void *callbackArgs, aclrtStream stream)
  {
    if (keyNum == 0) { return; }
    if (keys == nullptr) { return; }
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::ForEach<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual, CallbackOp><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum,
      (uint8_t*)callbackArgs);
  }

  template <typename KeyType>
  void ClearSimd(uint8_t *table, KeyType emptyValue, uint64_t tableCapacity, aclrtStream stream)
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    uint64_t ubSize = 0;
    platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    auto totalDataNum = tableCapacity;
    uint64_t tmp = ubSize / BLOCK_SIZE / BUFFER_NUM;
    uint32_t tileBlockNum = 1U;
    if (tmp > 0) {
      uint64_t tb = tmp / sizeof(KeyType);
      tileBlockNum = (tb == 0) ? 1U : tb;
    }

    // 每个 tile 包含的元素数（至少 1）
    uint32_t tileDataNum = tileBlockNum * BLOCK_SIZE;
    if (tileDataNum == 0U) tileDataNum = BLOCK_SIZE;

    // 总 block 数（向上取整）
    uint32_t totalBlockNum = (totalDataNum + BLOCK_SIZE - 1ULL) / BLOCK_SIZE;

    uint32_t actualCoreNum = static_cast<uint32_t>(aivCoreNum);
    if (actualCoreNum > totalBlockNum) actualCoreNum = totalBlockNum;
    if (actualCoreNum == 0ULL) actualCoreNum = 1ULL; // 最少 1 core

    uint32_t eachCoreBlockNum = totalBlockNum / actualCoreNum; // 每一个核分到的基本块数
    uint32_t coresWithExtra = static_cast<uint32_t>(totalBlockNum % aivCoreNum); // 前 coresWithExtra 个核是 remain-core
    
    // common-core 数量（元素）
    uint32_t commonCoreDataNum = static_cast<uint32_t>(eachCoreBlockNum * BLOCK_SIZE);
    uint32_t commonTileNum = static_cast<uint32_t>(eachCoreBlockNum / tileBlockNum);
    uint32_t commonRemainBlock = eachCoreBlockNum % tileBlockNum;
    uint32_t commonAlignTileNum = commonTileNum + (commonRemainBlock > 0 ? 1 : 0);
    uint32_t commonTailDataNum = (commonRemainBlock > 0) ? (commonRemainBlock * BLOCK_SIZE) : 0;

    // remain-core 数量（每个多一个block）
    uint32_t earchRemainCoreBlockNum = eachCoreBlockNum + 1ULL;
    uint32_t remainCoreDataNum = static_cast<uint32_t>(earchRemainCoreBlockNum * BLOCK_SIZE);
    uint32_t remainTileNum = static_cast<uint32_t>(earchRemainCoreBlockNum / static_cast<uint32_t>(tileBlockNum));
    uint32_t remainRemainBlock = earchRemainCoreBlockNum % tileBlockNum;
    uint32_t remainAlignTileNum = remainTileNum + (remainRemainBlock > 0 ? 1 : 0);
    uint32_t remainTailDataNum = (remainRemainBlock > 0) ? (remainRemainBlock * BLOCK_SIZE) : 0;

    aclco::ClearSIMD<KeyType><<<aivCoreNum, 0, stream>>>(
      table,
      emptyValue,
      commonCoreDataNum, commonAlignTileNum, commonTailDataNum,
      remainCoreDataNum, remainAlignTileNum, remainTailDataNum,
      tileDataNum, coresWithExtra);
  }

  void Clear(aclrtStream stream)
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    if (sizeof(ValueType) <= 8 && storage_.Capacity() > aivCoreNum * BLOCK_SIZE * sizeof(ValueType)) {
      if constexpr (isPairV<ValueType>) {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_.first, storage_.Capacity() * 2, stream);
        uint32_t ret = aclrtSynchronizeStream(stream); 
        CheckRet(ret, "StaticMap::Clear::aclrtSynchronizeStream");
      } else {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_, storage_.Capacity(), stream);
        uint32_t ret = aclrtSynchronizeStream(stream); 
        CheckRet(ret, "StaticSet::Clear::aclrtSynchronizeStream");
      }
    } else {
      storage_.Initialize(emptyValueStorage_.Data(), stream);
    }
  }

  void ClearAsync(aclrtStream stream) noexcept
  {
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
     if (sizeof(ValueType) <= 8 && storage_.Capacity() > aivCoreNum * BLOCK_SIZE * sizeof(ValueType)) {
      if constexpr (isPairV<ValueType>) {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_.first, storage_.Capacity() * 2, stream);
      } else {
        ClearSimd((uint8_t*)storage_.Data(), emptyValue_, storage_.Capacity(), stream);
      }
    } else {
      storage_.InitializeAsync(emptyValueStorage_.Data(), stream);
    }
  }

  SizeType Count(void *keys, Extent keyNum, aclrtStream stream)
  {
    if (keyNum == 0) { return 0; }
    if (keys == nullptr) {return 0; }
    argStorage_.Initialize(0, stream);
    auto aivCoreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAiv();
    aclco::Count<KeyType, ValueType, bucketSize, ProbingScheme, KeyEqual><<<aivCoreNum, 0, stream>>>(
      (uint8_t*)storage_.Data(),
      (uint8_t*)keys,
      (uint32_t*)argStorage_.Data(),
      (uint8_t*)emptyValueStorage_.Data(),
      storage_.Capacity(),
      keyNum);
    
    uint32_t ret = aclrtSynchronizeStream(stream); 
    if (isPairV<ValueType>) {
      CheckRet(ret, "StaticMap::Count::aclrtSynchronizeStream"); 
    }
    else {
      CheckRet(ret, "StaticSet::Count::aclrtSynchronizeStream");     
    }
    return argStorage_.LoadToHost(stream);
  }

  constexpr auto Capacity() const noexcept
  {
    return storage_.Capacity();
  }

  ValueType* Data() const
  {
    return storage_.Data();
  }

  /**
   * @brief Gets the probing scheme.
   *
   * @return The probing scheme used for the container
   */
  [[nodiscard]] constexpr ProbingSchemeType const& GetProbingScheme() const noexcept
  {
    return probingScheme_;
  }

  /**
   * @brief Gets the key comparator.
   *
   * @return The comparator used to compare keys
   */
  [[nodiscard]] constexpr KeyEqual GetKeyEqual() const noexcept
  {
    return predicate_;
  }

  /**
   * @brief Gets the function(s) used to hash keys
   *
   * @return The function(s) used to hash keys
   */
  [[nodiscard]] constexpr Hasher HashFunction() const noexcept
  {
    return this->GetProbingScheme().HashFunction();
  }

 protected:
  ValueType emptyValue_;
  ArgStorage<SizeType, SizeType> argStorage_;
  ArgStorage<Value> emptyValueStorage_;
  ProbingSchemeType probingScheme_;
  StorageType storage_;
  KeyEqual predicate_;
};
}