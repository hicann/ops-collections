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

#include <cstdint>

#include "kernel_operator.h"
#include "macros.h"
#include "roaring_bitmap_ref.h"

namespace aclco::detail {

constexpr uint32_t kRoaringThreadsPerBlock = 1024;

template <bool Aligned16, bool Aligned32>
COLLECTION_SIMT_VF LAUNCH_BOUND(kRoaringThreadsPerBlock) inline void RoaringBitmapContains32Simt(
    __gm__ uint8_t const* bitmap, __gm__ RoaringBitmapMetadata32 const* metadata, __gm__ uint32_t const* keys,
    __gm__ bool* output, uint64_t keyNum)
{
    uint64_t globalThread = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
                            AscendC::Simt::GetThreadIdx();
    uint64_t totalThreads = static_cast<uint64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    for (uint64_t i = globalThread; i < keyNum; i += totalThreads) {
        output[i] = RoaringContains32<Aligned16, Aligned32>(bitmap, metadata, keys[i]);
    }
}

COLLECTION_SIMT_VF LAUNCH_BOUND(kRoaringThreadsPerBlock) inline void RoaringBitmapContains64Simt(
    __gm__ uint8_t const* bitmap, __gm__ RoaringBitmapMetadata64 const* metadata,
    __gm__ RoaringBitmapBucket const* buckets, __gm__ uint64_t const* keys, __gm__ bool* output, uint64_t keyNum)
{
    uint64_t globalThread = static_cast<uint64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
                            AscendC::Simt::GetThreadIdx();
    uint64_t totalThreads = static_cast<uint64_t>(AscendC::Simt::GetBlockNum()) * AscendC::Simt::GetThreadNum();
    RoaringBitmapRef<uint64_t> ref(bitmap, metadata, buckets, metadata->numKeys, metadata->sizeBytes);
    for (uint64_t i = globalThread; i < keyNum; i += totalThreads) {
        output[i] = ref.Contains(keys[i]);
    }
}

} // namespace aclco::detail

namespace aclco {

extern "C" COLLECTION_AIV_GLOBAL void RoaringBitmapContains32Kernel(
    __gm__ uint8_t const* bitmap, __gm__ detail::RoaringBitmapMetadata32 const* metadata, __gm__ uint32_t const* keys,
    __gm__ bool* output, uint64_t keyNum)
{
    AscendC::Simt::VF_CALL<detail::RoaringBitmapContains32Simt<false, false>>(
        AscendC::Simt::Dim3{detail::kRoaringThreadsPerBlock}, bitmap, metadata, keys, output, keyNum);
}

extern "C" COLLECTION_AIV_GLOBAL void RoaringBitmapContains64Kernel(
    __gm__ uint8_t const* bitmap, __gm__ detail::RoaringBitmapMetadata64 const* metadata,
    __gm__ detail::RoaringBitmapBucket const* buckets, __gm__ uint64_t const* keys, __gm__ bool* output,
    uint64_t keyNum)
{
    AscendC::Simt::VF_CALL<detail::RoaringBitmapContains64Simt>(AscendC::Simt::Dim3{detail::kRoaringThreadsPerBlock},
                                                                bitmap, metadata, buckets, keys, output, keyNum);
}

extern "C" COLLECTION_AIV_GLOBAL void RoaringBitmapContains32Aligned16Kernel(
    __gm__ uint8_t const* bitmap, __gm__ detail::RoaringBitmapMetadata32 const* metadata, __gm__ uint32_t const* keys,
    __gm__ bool* output, uint64_t keyNum)
{
    AscendC::Simt::VF_CALL<detail::RoaringBitmapContains32Simt<true, false>>(
        AscendC::Simt::Dim3{detail::kRoaringThreadsPerBlock}, bitmap, metadata, keys, output, keyNum);
}

extern "C" COLLECTION_AIV_GLOBAL void RoaringBitmapContains32AlignedKernel(
    __gm__ uint8_t const* bitmap, __gm__ detail::RoaringBitmapMetadata32 const* metadata, __gm__ uint32_t const* keys,
    __gm__ bool* output, uint64_t keyNum)
{
    AscendC::Simt::VF_CALL<detail::RoaringBitmapContains32Simt<true, true>>(
        AscendC::Simt::Dim3{detail::kRoaringThreadsPerBlock}, bitmap, metadata, keys, output, keyNum);
}

} // namespace aclco
