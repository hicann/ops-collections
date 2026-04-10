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
#include "utility/math_utils.h"

namespace aclco {
template <typename First, typename Second>
COLLECTION_HOST_DEVICE constexpr std::size_t PairAlignment()
{
    constexpr std::size_t alignment = BitCeil(sizeof(First) + sizeof(Second));
    return std::min(std::size_t{16}, alignment);
}

template <typename First, typename Second>
COLLECTION_HOST_DEVICE constexpr Pair<First, Second>::Pair(First const& f, Second const& s)
  : first{f}, second{s}
{
}

template <typename First, typename Second>
template <typename F, typename S>
COLLECTION_HOST_DEVICE constexpr Pair<First, Second>::Pair(__gm__ Pair<F, S> const& p)
  : first{p.first}, second{p.second}
{
}

// template <typename First, typename Second>
// template <typename F, typename S>
// COLLECTION_DEVICE constexpr Pair<First, Second>::Pair(__ubuf__ Pair<F, S> const& p)
//   : first{p.first}, second{p.second}
// {
// }

template <typename F, typename S>
COLLECTION_HOST_DEVICE constexpr Pair<F, S> MakePair(F&& f, S&& s) noexcept
{
    return Pair<F, S>(f, s);
}

template <class T1, class T2, class U1, class U2>
COLLECTION_HOST_DEVICE constexpr bool operator==(Pair<T1, T2> const& lhs,
                                                 Pair<U1, U2> const& rhs) noexcept
{
    return lhs.first == rhs.first && lhs.second == rhs.second;
}

} // namespace aclco