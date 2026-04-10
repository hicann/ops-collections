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
#include <acl/acl.h>
#include <cstdlib>

namespace aclco {
template <typename T>
class DefaultAllocator {
 public:
  using ValueType = T;
  using value_type = ValueType; ///< Allocator's value type

  DefaultAllocator() = default;

  template <class U>
  DefaultAllocator(DefaultAllocator<U> const&) noexcept
  {
  }

  ValueType* Allocate(std::size_t n) const
  {
    ValueType* p;
    aclrtMalloc((void**)&p, sizeof(ValueType) * n, ACL_MEM_MALLOC_HUGE_FIRST);
    return p;
  }

  void Deallocate(ValueType* p)
  {
    aclrtFree((void*)p);
  }
};

// template <typename T, typename U>
// bool operator==(DefaultAllocator<T> const&, DefaultAllocator<U> const&) noexcept
// {
//   return true;
// }

// template <typename T, typename U>
// bool operator!=(DefaultAllocator<T> const& lhs, DefaultAllocator<U> const& rhs) noexcept
// {
//   return not(lhs == rhs);
// }
}