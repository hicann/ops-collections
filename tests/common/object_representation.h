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

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace aclco::test {

template <typename Destination, typename Source>
Destination ObjectRepresentationCast(Source const& source) noexcept
{
    static_assert(sizeof(Destination) == sizeof(Source), "Object representations must have the same size");
    static_assert(std::is_trivially_copyable_v<Destination>, "Destination must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<Source>, "Source must be trivially copyable");

    // Callers must provide an object representation that is valid for
    // Destination. This helper only performs the byte-wise C++17 copy.
    Destination destination{};
    auto* destinationBytes = reinterpret_cast<unsigned char*>(&destination);
    auto const* sourceBytes = reinterpret_cast<unsigned char const*>(&source);
    std::copy_n(sourceBytes, sizeof(Destination), destinationBytes);
    return destination;
}

} // namespace aclco::test
