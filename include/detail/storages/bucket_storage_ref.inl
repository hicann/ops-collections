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

#include <cassert>
#include <acl/acl.h>
#include "utility/kernel_launch_utils.h"

namespace aclco {
template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr BucketStorageRef<T, BucketSize, Extent>::BucketStorageRef(Extent size, __gm__ ValueType* slots) noexcept
: capacity_{size}, slots_{slots}
{
}

template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr typename BucketStorageRef<T, BucketSize, Extent>::Iterator
BucketStorageRef<T, BucketSize, Extent>::End() noexcept
{
    return Iterator{reinterpret_cast<__gm__ ValueType*>(this->Data() + this->Capacity())};
}

template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr typename BucketStorageRef<T, BucketSize, Extent>::Iterator
BucketStorageRef<T, BucketSize, Extent>::End() const noexcept
{
    return Iterator{reinterpret_cast<__gm__ ValueType*>(this->Data() + this->Capacity())};
}

template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr __gm__ typename BucketStorageRef<T, BucketSize, Extent>::ValueType*
BucketStorageRef<T, BucketSize, Extent>::Data() noexcept
{
    return slots_;
}

template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr __gm__ typename BucketStorageRef<T, BucketSize, Extent>::ValueType*
BucketStorageRef<T, BucketSize, Extent>::Data() const noexcept
{
    return slots_;
}

template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr typename BucketStorageRef<T, BucketSize, Extent>::SizeType
BucketStorageRef<T, BucketSize, Extent>::NumBuckets() const noexcept
{
    return capacity_ / bucketSize;
}

template<typename T, int32_t BucketSize, typename Extent>
COLLECTION_DEVICE constexpr typename BucketStorageRef<T, BucketSize, Extent>::SizeType
BucketStorageRef<T, BucketSize, Extent>::Capacity() const noexcept
{
    return capacity_;
}
}// namespace aclco