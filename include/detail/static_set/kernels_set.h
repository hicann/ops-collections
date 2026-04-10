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
#include "static_set_ref.h"
#include "detail/storages/bucket_storage_ref.h"
#include "probing_scheme.h"
#include "hash_functions.h"

namespace aclco {

template <typename Key>
__simt_vf__ __aicore__ LAUNCH_BOUND(MAX_THREAD_NUM) inline void ClearSetSimt( 
  __gm__ uint8_t *table, uint32_t tableSize, __gm__ uint8_t *emptyValue) 
{
  uint32_t blockIndex = AscendC::Simt::GetBlockIdx();
  uint32_t blockNumber = AscendC::Simt::GetBlockNum();
  uint32_t globalThreadIdx = blockIndex * AscendC::Simt::GetThreadNum() + AscendC::Simt::GetThreadIdx();
  uint32_t totalThreadNum = blockNumber * AscendC::Simt::GetThreadNum();

  using KeyType = Key;

  __gm__ KeyType* data = (__gm__ KeyType*)table;
  __gm__ KeyType* emptyVal = (__gm__ KeyType*)emptyValue;

  for (uint32_t i = globalThreadIdx; i < tableSize; i = i + totalThreadNum) {
    data[i]= *emptyVal;
  }
}


template <typename Key, uint32_t BucketSize, typename ProbingScheme, typename KeyEqual>
__attribute__((aiv)) __global__ __aicore__ void ClearSet(__gm__ uint8_t *table, uint32_t tableSize,
                                                      __gm__ uint8_t *emptyValue)
{
  AscendC::Simt::VF_CALL<ClearSetSimt<Key>>(AscendC::Simt::Dim3{MAX_THREAD_NUM},
    table, tableSize, emptyValue);
}
}