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

TEMPLATE_TEST_CASE_SIG("static_multimap retrieve all correctness", "[static_multimap][retrieveAll]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(5u, 128u, 1027u, 8192u, 100000u);
    auto occupancy = GENERATE(0.0, 0.1, 0.5, 0.9, 1.0);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto pair_num = static_cast<std::size_t>(capacity * occupancy);
    auto distinct = pair_num == 0u ? 0u : std::max<std::size_t>(1u, pair_num / multiplicity);
    CAPTURE(capacity, occupancy, multiplicity, pair_num);

    auto map = MakeMultimap<Key, Value>(capacity, stream);
    auto pairs = MakePairs<Key, Value>(distinct, multiplicity);
    if (pairs.size() > pair_num) {
        pairs.resize(pair_num);
    }
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(pairs.size());
    if (!pairs.empty()) {
        d_pairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    }
    map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(pairs.size()), stream);

    aclco::test::DeviceBuffer<Key> d_keys(pairs.size());
    aclco::test::DeviceBuffer<Value> d_values(pairs.size());
    auto retrieved = map.RetrieveAll(d_keys.Data(), d_values.Data(), aclco::Extent<std::size_t>(pairs.size()), stream);
    REQUIRE(retrieved == pairs.size());
    auto actual = NormalizePairs(d_keys.CopyToHost(stream), d_values.CopyToHost(stream), retrieved);
    REQUIRE(actual == NormalizePairs(pairs));
}

TEMPLATE_TEST_CASE_SIG("static_multimap retrieve_all boundary test", "[static_multimap][retrieveAll][boundary]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto map = MakeMultimap<Key, Value>(128u, sg.stream);
    REQUIRE(map.RetrieveAll(nullptr, nullptr, aclco::Extent<std::size_t>(0u), sg.stream) == 0u);
}
