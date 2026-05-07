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

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "macros.h"
#include "utility/math_utils.h"
#include "utility/traits.h"

namespace aclco {
namespace detail {
COLLECTION_HOST_DEVICE int Memcmp(void const* lhs, void const* rhs, size_t count)
{
  auto lhsC = reinterpret_cast<unsigned char const*>(lhs);
  auto rhsC = reinterpret_cast<unsigned char const*>(rhs);
  while (count--) {
    auto const lhsV = *lhsC++;
    auto const rhsV = *rhsC++;
    if (lhsV < rhsV) { return -1; }
    if (lhsV > rhsV) { return 1; }
  }
  return 0;
}

template <std::size_t TypeSize>
struct BitWiseCompareImpl {
  COLLECTION_HOST_DEVICE static bool compare(char const* lhs, char const* rhs)
  {
    return Memcmp(lhs, rhs, TypeSize) == 0;
  }
};

template <>
struct BitWiseCompareImpl<4> {
  COLLECTION_HOST_DEVICE static bool compare(char const* lhs, char const* rhs)
  {
    return *reinterpret_cast<uint32_t const*>(lhs) == *reinterpret_cast<uint32_t const*>(rhs);
  }
};

template <>
struct BitWiseCompareImpl<8> {
  COLLECTION_HOST_DEVICE static bool compare(char const* lhs, char const* rhs)
  {
    return *reinterpret_cast<uint64_t const*>(lhs) == *reinterpret_cast<uint64_t const*>(rhs);
  }
};

/**
 * @brief 计算类型的对齐值，取类型大小向上取整到2的幂与16之间的较小值
 */
template <typename T>
COLLECTION_HOST_DEVICE constexpr std::size_t alignment()
{
  constexpr std::size_t alignment = BitCeil(sizeof(T));
  return std::size_t{16} < alignment ? std::size_t{16} : alignment;
}

/**
 * @brief 对两个对象执行按位相等比较
 *
 * @tparam T 具有唯一对象表示的类型
 * @param lhs 第一个对象
 * @param rhs 第二个对象
 * @return lhs 和 rhs 的对象表示中的位是否相同
 */
template <typename T>
COLLECTION_HOST_DEVICE constexpr bool BitWiseCompare(T lhs, T rhs)
{
  static_assert(
    aclco::isBitwiseComparableV<T>,
    "按位比较的对象必须具有唯一的对象表示，或者通过 aclco::isBitwiseComparableV 的特化"
    "显式声明为可安全进行按位比较。");

  alignas(detail::alignment<T>()) T lhsAligned{lhs};
  alignas(detail::alignment<T>()) T rhsAligned{rhs};
  return detail::BitWiseCompareImpl<sizeof(T)>::compare(reinterpret_cast<char const*>(&lhsAligned),
                                                          reinterpret_cast<char const*>(&rhsAligned));
}

}  // namespace detail
}  // namespace aclco
