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

#include <acl/acl.h>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include "macros.h"

namespace aclco {

 /**
  * @brief 计算 pair 结构的对齐大小
  * 
  * @tparam First 第一个元素的类型
  * @tparam Second 第二个元素的类型
  * 
  * @return 返回 pair 结构的对齐大小
  */
template <typename First, typename Second>
COLLECTION_HOST_DEVICE constexpr std::size_t PairAlignment();

/**
 * @brief 自定义的 pair 类型
 *
 * @tparam First pair 中第一个值的类型
 * @tparam Second pair 中第二个值的类型
 * 
 * @warning First 和 Second 类型大小不能超过8字节
 */
template<typename First, typename Second>
struct alignas(PairAlignment<First, Second>()) Pair {
    using FirstType  = First;
    using SecondType = Second;
    Pair() = default;
    ~Pair() = default;
    Pair(Pair const&) = default;
    Pair(Pair&&) = default;
    Pair& operator=(Pair const&) = default;
    Pair& operator=(Pair&&) = default;

    /**
   * @brief 构造函数
   *
   * @param f 第一个元素的值
   * @param s 第二个元素的值
   */
    COLLECTION_HOST_DEVICE constexpr Pair(First const& f, Second const& s);

    /**
   * @brief 从给定的 Pair `p` 拷贝构造一个 Pair
   *
   * @tparam F 第一个元素的类型
   * @tparam S 第二个元素的类型
   *
   * @param p 要从中拷贝的 Pair
   */
    template <typename F, typename S>
    COLLECTION_HOST_DEVICE constexpr Pair(__gm__ Pair<F, S> const& p);

    // template <typename F, typename S>
    // COLLECTION_DEVICE constexpr Pair(__ubuf__ Pair<F, S> const& p);

    First first;
    Second second;
};

/**
 * @brief 创建 Pair 的便捷函数
 *
 * @tparam F 第一个元素的类型
 * @tparam S 第二个元素的类型
 *
 * @param f 第一个元素的值
 * @param s 第二个元素的值
 *
 * @return 返回一个包含给定值的 Pair
 */
template <typename F, typename S>
COLLECTION_HOST_DEVICE constexpr Pair<F, S> MakePair(F&& f, S&& s) noexcept;

/**
 * @brief 测试两个 Pair 的两个元素是否都相等
 *
 * @tparam T1 左侧 Pair 第一个元素的类型
 * @tparam T2 左侧 Pair 第二个元素的类型
 * @tparam U1 右侧 Pair 第一个元素的类型
 * @tparam U2 右侧 Pair 第二个元素的类型
 *
 * @param lhs 左侧的 Pair
 * @param rhs 右侧的 Pair
 *
 * @return 如果两个 Pair 相等则返回 true，否则返回 false
 */
template <class T1, class T2, class U1, class U2>
COLLECTION_HOST_DEVICE constexpr bool operator==(Pair<T1, T2> const& lhs,
                                                 Pair<U1, U2> const& rhs) noexcept;

} // namespace aclco
#include "detail/pair/pair.inl"