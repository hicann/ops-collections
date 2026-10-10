/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "common/static_multimap_test_common.h"

using namespace aclco::test::static_multimap;

TEMPLATE_TEST_CASE_SIG("static_multimap create correctness", "[static_multimap][create]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto requested = GENERATE(1u, 5u, 128u, 1027u, 8192u, 1000000u);
    CAPTURE(requested);

    auto map = MakeMultimap<Key, Value>(requested, stream);
    REQUIRE(map.Capacity() >= requested);
    REQUIRE(map.Size(stream) == 0u);

    std::vector<Key> out_keys(map.Capacity());
    std::vector<Value> out_values(map.Capacity());
    aclco::test::DeviceBuffer<Key> d_keys(out_keys.size());
    aclco::test::DeviceBuffer<Value> d_values(out_values.size());
    auto retrieved = map.RetrieveAll(d_keys.Data(), d_values.Data(), aclco::Extent<std::size_t>(out_keys.size()),
                                     stream);
    REQUIRE(retrieved == 0u);
}

TEMPLATE_TEST_CASE_SIG("static_multimap create invalid capacity", "[static_multimap][create][negative]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    REQUIRE_THROWS(MakeMultimap<Key, Value>(0u, sg.stream));
}
