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
#include "static_map_ref.h"
#include "static_set_ref.h"
#include "detail/storages/bucket_storage_ref.h"
#include "probing_scheme.h"
#include "hash_functions.h"
#include "pair.h"
#include "utility/is_same.h"

namespace aclco {
constexpr uint32_t MAX_THREAD_NUM = 2048;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
constexpr uint32_t DEFAULT_THREAD_NUM = 512;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename Value>
__simt_callee__ __aicore__ inline void CopyEmptyValue(__gm__ Value* value, __gm__ Value* emptyValue)
{
  if constexpr (isPairV<Value>) {
    value->first = emptyValue->first;
    value->second = emptyValue->second;
  }
  else {
    *value = *emptyValue;
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertIfSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum, __gm__ uint32_t *insertFailedNum)
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);
  
  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    if (!pred(stencilValue)) { continue; }
    Value insertValue = *((__gm__ Value*)(values) + i);
    if (!ref.Insert(insertValue)) { 
      AscendC::Simt::AtomicAdd(insertFailedNum, addVal); 
    }
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertIfSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    if (!pred(stencilValue)) { continue; }
    Value insertValue = *((__gm__ Value*)(values) + i);
    ref.Insert(insertValue);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertOrAssignSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum, __gm__ uint32_t *insertFailedNum)
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    Value insertValue = *((__gm__ Value*)(values) + i);
    if (!ref.InsertOrAssign(insertValue)) {
      AscendC::Simt::AtomicAdd(insertFailedNum, addVal);
    }
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertOrAssignSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    Value insertValue = *((__gm__ Value*)(values) + i);
    ref.InsertOrAssign(insertValue);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void InsertAndFindSimtAsync(
  __gm__ uint8_t *table, __gm__ uint8_t *values, __gm__ uint8_t *outputFind, __gm__ uint8_t *outputInsert,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t valueNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < valueNum; i = i + totalThreadNum) {
    Value value = *((__gm__ Value*)(values) + i);
    auto const [found, inserted] = ref.InsertAndFind(value);

    if constexpr (isPairV<Value>) {
      *((__gm__ typename Value::SecondType*)(outputFind) + i) = found;
    } else {
      *((__gm__ Value*)(outputFind) + i) = found;
    }
    *((__gm__ bool*)(outputInsert) + i) = inserted;
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void EraseSimt( // 如果传入的对象或结构体只支持传数据，则意味着上层软件设计必须考虑到编译器不支持的问题，设计上将数据和处理函数分开
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t keyNum, __gm__ uint32_t *eraseFailedNum) // 这些参数待后续编译器支持结构体传参后整合成结构体
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);
  
  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key eraseKey = *((__gm__ Key*)(keys) + i);
    if (!ref.Erase(eraseKey)) { 
       AscendC::Simt::AtomicAdd(eraseFailedNum, addVal); 
     }
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void EraseSimtAsync( // 如果传入的对象或结构体只支持传数据，则意味着上层软件设计必须考虑到编译器不支持的问题，设计上将数据和处理函数分开
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t keyNum) // 这些参数待后续编译器支持结构体传参后整合成结构体
{
  uint32_t addVal = 1;
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);
  
  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key eraseKey = *((__gm__ Key*)(keys) + i);
    ref.Erase(eraseKey);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void FindIfSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t keyNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    Key findKey = *((__gm__ Key*)(keys) + i);
    if constexpr (isPairV<Value>) {
      *((__gm__ typename Value::SecondType*)(outputValues) + i) = pred(stencilValue)
        ? ref.Find(findKey) : ((__gm__ Value*)emptyValue)->second;
    } else {
      *((__gm__ Value*)(outputValues) + i) = pred(stencilValue)
        ? ref.Find(findKey) : *((__gm__ Value*)emptyValue);
    }
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void ContainsIfSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t keyNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  Predicate pred = {};

  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    StencilT stencilValue = *((__gm__ StencilT*)(stencil) + i);
    Key findKey = *((__gm__ Key*)(keys) + i);
    *((__gm__ bool*)(outputValues) + i) = pred(stencilValue) ? ref.Contains(findKey) : false;
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename CallbackOp>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void ForEachSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint8_t *emptyValue,
  uint32_t tableSize, uint32_t keyNum, __gm__ uint8_t *callbackArgs)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
    StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
    StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual keyEqual = {};
  CallbackOp callback = CallbackOp(callbackArgs);

  RefType ref(*((__gm__ Value*)emptyValue), keyEqual, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key probeKey = *((__gm__ Key*)(keys) + i);
    ref.ForEach(probeKey, callback);
  }
}

template <typename Value>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREAD_NUM) inline void ClearSimt(
  __gm__ uint8_t *table, uint32_t tableSize, __gm__ uint8_t *emptyValue) 
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using ValueType = Value;

  __gm__ ValueType* data = (__gm__ ValueType*)table;
  __gm__ ValueType* emptyVal = (__gm__ ValueType*)emptyValue;

  for (uint32_t i = globalThreadIdx; i < tableSize; i = i + totalThreadNum) {
    CopyEmptyValue(&data[i], emptyVal);
  }
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_LAUNCH_BOUND) inline void CountSimt(
  __gm__ uint8_t *table, __gm__ uint8_t *keys, __gm__ uint32_t *outputNum,
  __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t keyNum)
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using StorageRefType = aclco::BucketStorageRef<Value, BucketSize>;
  using ProbingSchemeType = ProbingScheme;
  using RefType = typename std::conditional<!isPairV<Value>,
  StaticSetRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>,
  StaticMapRef<Key, KeyEqual, ProbingSchemeType, StorageRefType>>::type;

  StorageRefType tableRef = StorageRefType(tableSize, (__gm__ Value*)table);
  ProbingSchemeType probingScheme = {};
  KeyEqual predicate = {};
  RefType ref(*((__gm__ Value*)emptyValue), predicate, probingScheme, tableRef);

  for (uint32_t i = globalThreadIdx; i < keyNum; i = i + totalThreadNum) {
    Key targetKey = *((__gm__ Key*)(keys) + i);
    AscendC::Simt::AtomicAdd(outputNum, ref.Count(targetKey)); 
  }
}

template <typename Key>
__vector__ __global__ __aicore__ void ClearSIMD(__gm__ uint8_t *table, Key emptyKey,
                                                uint32_t commonCoreDataNum, uint32_t commonCoreAlignTileNum, uint32_t commonCoreTailDataNum,
                                                uint32_t remainCoreDataNum, uint32_t remainCoreAlignTileNum, uint32_t remainCoreTailDataNum,
                                                uint32_t tileDataNum, uint32_t remainBlockNum)
{
  // 只能为32bit, 因为当前Duplicate接口仅支持到32bit
  using KeyType = Key;

  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> tableQueue;
  AscendC::GlobalTensor<KeyType> tableGM;

  uint32_t coreId = AscendC::GetBlockIdx();
  uint32_t globalBufferIdx;

  uint32_t coreDataNum;
  uint32_t tileNum;
  uint32_t tailDataNum;
  uint32_t processDataNum;

  if (coreId < remainBlockNum) {
    coreDataNum = remainCoreDataNum;
    tileNum = remainCoreAlignTileNum;
    tailDataNum = remainCoreTailDataNum;
    globalBufferIdx = remainCoreDataNum * coreId;
  } else {
    coreDataNum = commonCoreDataNum;
    tileNum = commonCoreAlignTileNum;
    tailDataNum = commonCoreTailDataNum;
    globalBufferIdx = remainCoreDataNum * remainBlockNum + (coreId - remainBlockNum) * commonCoreDataNum;
  }

  tableGM.SetGlobalBuffer((__gm__ KeyType*)table + globalBufferIdx, coreDataNum);
  pipe.InitBuffer(tableQueue, BUFFER_NUM, tileDataNum * sizeof(KeyType));

  uint32_t loopCount = tileNum;
  processDataNum = tileDataNum;
  AscendC::LocalTensor<KeyType> tableLocal = tableQueue.AllocTensor<KeyType>();
  AscendC::Duplicate<KeyType>(tableLocal, emptyKey, processDataNum);
  tableQueue.EnQue(tableLocal);
  tableLocal = tableQueue.DeQue<KeyType>();
  for (uint32_t i = 0; i < loopCount; i++) {
    if (i == tileNum - 1) {
      processDataNum = tailDataNum;
    }
    AscendC::DataCopy(tableGM[i * tileDataNum], tableLocal, processDataNum);
  }
  tableQueue.FreeTensor(tableLocal);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__attribute__((aiv)) __global__ __aicore__ void InsertIf(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                         __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
                                                         uint32_t tableSize, uint32_t valueNum,
                                                         __gm__ uint8_t *insertFailedNum)
{
  AscendC::Simt::VF_CALL<InsertIfSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{1024},
    table, values, stencil, emptyValue, tableSize, valueNum, (__gm__ uint32_t*)insertFailedNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__attribute__((aiv)) __global__ __aicore__ void InsertIfAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                               __gm__ uint8_t *stencil, __gm__ uint8_t *emptyValue,
                                                               uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<InsertIfSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, stencil, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void InsertOrAssign(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                                __gm__ uint8_t *emptyValue,
                                                                uint32_t tableSize, uint32_t valueNum,
                                                                __gm__ uint8_t *insertFailedNum)
{
  AscendC::Simt::VF_CALL<InsertOrAssignSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{1024},
    table, values, emptyValue, tableSize, valueNum, (__gm__ uint32_t*)insertFailedNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void InsertOrAssignAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                                      __gm__ uint8_t *emptyValue,
                                                                      uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<InsertOrAssignSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void InsertAndFindAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                         __gm__ uint8_t *outputFind, __gm__ uint8_t *outputInsert,
                                                         __gm__ uint8_t *emptyValue, uint32_t tableSize,
                                                         uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<InsertAndFindSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{1024},
    table, values, outputFind, outputInsert, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void Erase(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                      __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t valueNum,	 
                                                      __gm__ uint8_t *eraseFailedNum)
{
  AscendC::Simt::VF_CALL<EraseSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, tableSize, valueNum, (__gm__ uint32_t*)eraseFailedNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void EraseAsync(__gm__ uint8_t *table, __gm__ uint8_t *values,
                                                      __gm__ uint8_t *emptyValue, uint32_t tableSize, uint32_t valueNum)
{
  AscendC::Simt::VF_CALL<EraseSimtAsync<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, values, emptyValue, tableSize, valueNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__attribute__((aiv)) __global__ __aicore__ void FindIf(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                       __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
                                                       __gm__ uint8_t *emptyValue,
                                                       uint32_t tableSize, uint32_t keyNum)
{
  AscendC::Simt::VF_CALL<FindIfSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, stencil, outputValues, emptyValue, tableSize, keyNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename StencilT, typename Predicate>
__attribute__((aiv)) __global__ __aicore__ void ContainsIf(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                           __gm__ uint8_t *stencil, __gm__ uint8_t *outputValues,
                                                           __gm__ uint8_t *emptyValue,
                                                           uint32_t tableSize, uint32_t keyNum)
{
  AscendC::Simt::VF_CALL<ContainsIfSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, StencilT, Predicate>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, stencil, outputValues, emptyValue, tableSize, keyNum);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual, typename CallbackOp>
__attribute__((aiv)) __global__ __aicore__ void ForEach(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                        __gm__ uint8_t *emptyValue,
                                                        uint32_t tableSize, uint32_t keyNum,
                                                        __gm__ uint8_t *callbackArgs)
{
  AscendC::Simt::VF_CALL<ForEachSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual, CallbackOp>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, emptyValue, tableSize, keyNum, callbackArgs);
}

template <typename Value>
__attribute__((aiv)) __global__ __aicore__ void Clear(__gm__ uint8_t *table, uint32_t tableSize,
                                                      __gm__ uint8_t *emptyValue)
{
  AscendC::Simt::VF_CALL<ClearSimt<Value>>(AscendC::Simt::Dim3{MAX_THREAD_NUM},
    table, tableSize, emptyValue);
}

template <typename Key, typename Value, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void Count(__gm__ uint8_t *table, __gm__ uint8_t *keys,
                                                       __gm__ uint32_t *outputNum, __gm__ uint8_t *emptyValue,
                                                       uint32_t tableSize, uint32_t keyNum)
{
  AscendC::Simt::VF_CALL<CountSimt<Key, Value, BucketSize, ProbingScheme, KeyEqual>>(AscendC::Simt::Dim3{DEFAULT_THREAD_NUM},
    table, keys, outputNum, emptyValue, tableSize, keyNum);
}
}