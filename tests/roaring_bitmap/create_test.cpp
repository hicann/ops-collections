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
#include <catch2/catch_template_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "roaring_bitmap.h"
#include "tests/common/acl_env.h"
#include "tests/common/roaring_bitmap_factory.h"
#include "tests/common/test_print.h"

TEMPLATE_TEST_CASE("roaring_bitmap create test", "[roaring_bitmap][create]", uint32_t, uint64_t)
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    using Key = TestType;
    auto keys = aclco::test::roaring_bitmap_factory::AcceptanceKeys<Key>();
    auto serialized = aclco::test::roaring_bitmap_factory::Serialize(keys);

    PRINT_BEFORE_EXEC("create test", Key, Key, 1, keys.size(), 0);
    aclco::RoaringBitmap<Key> bitmap(serialized.data(), serialized.size(), {}, stream);
    REQUIRE_PRINT(bitmap.Size() == keys.size());
    REQUIRE_PRINT(!bitmap.Empty());
    REQUIRE_PRINT(bitmap.Data() != nullptr);
    REQUIRE_PRINT(bitmap.SizeBytes() == serialized.size());
}
