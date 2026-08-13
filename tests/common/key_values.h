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

#include <limits>
#include <type_traits>
#include <vector>

namespace aclco::test {

template <typename Key>
void SetSignedBoundaryValues(std::vector<Key>& values)
{
    static_assert(std::is_integral_v<Key> && std::is_signed_v<Key>, "Key must be a signed integral type");
    if (values.size() > 3) {
        values[0] = Key{-1};
        values[1] = std::numeric_limits<Key>::min();
        values[2] = std::numeric_limits<Key>::max();
    }
}

} // namespace aclco::test
