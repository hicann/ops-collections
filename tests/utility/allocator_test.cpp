/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>

#include <acl/acl.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

#include "bucket_storage.h"
#include "utility/allocator.h"

#include "../common/acl_env.h"

namespace {

template <typename T>
struct HostAllocator {
    using ValueType = T;
    using value_type = T;

    template <typename U>
    struct rebind {
        using other = HostAllocator<U>;
    };

    explicit HostAllocator(bool fail = false) : fail_{fail} {}

    T* Allocate(std::size_t count) const noexcept
    {
        if (fail_ || count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
            return nullptr;
        }
        return new (std::nothrow) T[count];
    }

    void Deallocate(T* pointer) const noexcept { delete[] pointer; }

    bool fail_;
};

} // namespace

TEST_CASE("DefaultAllocator and BucketStorage reject invalid allocation", "[utility][allocator]")
{
    aclco::test::AclGlobalGuard aclGlobal{};

    using Value = std::uint64_t;
    aclco::DefaultAllocator<Value> allocator;

    constexpr std::size_t controlCount = 3;
    auto* controlAllocation = allocator.Allocate(controlCount);
    REQUIRE(controlAllocation != nullptr);
    allocator.Deallocate(controlAllocation);

    constexpr std::size_t overflowingCount = std::numeric_limits<std::size_t>::max() / sizeof(Value) + 2;
    auto* overflowing = allocator.Allocate(overflowingCount);
    if (overflowing != nullptr) {
        allocator.Deallocate(overflowing);
    }
    REQUIRE(overflowing == nullptr);
    REQUIRE_THROWS_AS((aclco::BucketStorage<Value, 1>{aclco::Extent<std::size_t>{overflowingCount}}), std::bad_alloc);

    HostAllocator<Value> hostAllocator;
    REQUIRE(hostAllocator.Allocate(overflowingCount) == nullptr);
    aclco::BucketStorage<Value, 1, aclco::Extent<std::size_t>, HostAllocator<Value>> controlStorage{
        aclco::Extent<std::size_t>{controlCount}, hostAllocator};
    REQUIRE(controlStorage.Data() != nullptr);
    REQUIRE(static_cast<std::size_t>(controlStorage.Capacity()) == controlCount);

    HostAllocator<Value> failingAllocator{true};
    REQUIRE_THROWS_AS((aclco::BucketStorage<Value, 1, aclco::Extent<std::size_t>, HostAllocator<Value>>{
                          aclco::Extent<std::size_t>{controlCount}, failingAllocator}),
                      std::bad_alloc);

    aclco::BucketStorage<Value, 1, aclco::Extent<std::size_t>, HostAllocator<Value>> empty{
        aclco::Extent<std::size_t>{0}, failingAllocator};
    REQUIRE(empty.Data() == nullptr);
    REQUIRE(static_cast<std::size_t>(empty.Capacity()) == 0);
}
