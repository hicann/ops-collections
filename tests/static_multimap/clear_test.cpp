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

TEMPLATE_TEST_CASE_SIG("static_multimap clear correctness", "[static_multimap][clear]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(5u, 128u, 1027u, 1000000u);
    auto occupancy = GENERATE(0.0, 0.1, 0.5, 0.9);
    CAPTURE(capacity, occupancy);

    auto map = MakeMultimap<Key, Value>(capacity, stream);
    auto pair_num = static_cast<std::size_t>(capacity * occupancy);
    auto pairs = MakePairs<Key, Value>(pair_num, 1u);
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(pairs.size());
    if (!pairs.empty()) {
        d_pairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    }
    map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(pairs.size()), stream);

    map.Clear(stream);
    REQUIRE(map.Size(stream) == 0u);
    map.Clear(stream);
    REQUIRE(map.Size(stream) == 0u);

    SECTION("insert after clear")
    {
        auto replacement = MakePairs<Key, Value>(std::max<std::size_t>(1u, capacity / 4u), 2u);
        aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_replacement(replacement.size());
        d_replacement.CopyFromHostAsync(replacement.data(), replacement.size(), stream);
        REQUIRE(map.Insert(d_replacement.Data(), aclco::Extent<std::size_t>(replacement.size()), stream) == 0u);
        REQUIRE(map.Size(stream) == replacement.size());
    }
}
