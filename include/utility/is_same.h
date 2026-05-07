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
#include "pair.h"

namespace aclco {
struct FalseType {
  static constexpr bool value = false;
};

struct TrueType {
  static constexpr bool value = true;
};

template <class T, class U>
struct IsSame : FalseType {};

template <class T>
struct IsSame<T, T> : TrueType {};

template <class T, class U>
inline constexpr bool isSameV = IsSame<T, U>::value;

template<typename T>
struct IsPair : FalseType {};

template<typename T, typename U>
struct IsPair<aclco::Pair<T, U>> : TrueType {};

template <class T>
inline constexpr bool isPairV = IsPair<T>::value;

template<typename T, bool IsPair = isPairV<T>>
struct PayloadTypeOf {
  using Type = T;
};

template<typename T, typename U>
struct PayloadTypeOf<aclco::Pair<T, U>, true> {
  using Type = U;
};

template<typename T>
using PayloadTypeT = typename PayloadTypeOf<T>::Type;
}