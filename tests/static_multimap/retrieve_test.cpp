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

TEMPLATE_TEST_CASE_SIG("static_multimap retrieve correctness", "[static_multimap][retrieve]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    TestStream context;
    auto stream = context.stream;
    auto capacity = GENERATE(128u, 1027u, 8192u);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto matching_rate = GENERATE(0.0, 0.1, 0.5, 1.0);
    auto distinct = std::max<std::size_t>(1u, capacity / (2u * multiplicity));
    CAPTURE(capacity, multiplicity, matching_rate);

    auto map = MakeMultimap<Key, Value>(capacity, stream);
    auto pairs = MakePairs<Key, Value>(distinct, multiplicity);
    auto oracle = MapOracle(pairs);
    auto d_pairs = InsertPairs(map, pairs, stream);

    auto queries = MakeQueries<Key>(distinct, std::min<std::size_t>(1024u, capacity), matching_rate);
    auto expected_count = ExpectedCount(oracle, queries);
    aclco::test::DeviceBuffer<Key> d_queries(queries.size());
    aclco::test::DeviceBuffer<Key> d_probe_out(expected_count);
    aclco::test::DeviceBuffer<Value> d_match_out(expected_count);
    d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);

    auto retrieved = map.Retrieve(d_queries.Data(), aclco::Extent<std::size_t>(queries.size()), d_probe_out.Data(),
                                  d_match_out.Data(), aclco::Extent<std::size_t>(expected_count), stream);
    REQUIRE(retrieved == expected_count);

    auto actual = NormalizePairs(d_probe_out.CopyToHost(stream), d_match_out.CopyToHost(stream), retrieved);
    std::vector<std::pair<Key, Value>> expected;
    for (auto query : queries) {
        auto range = oracle.equal_range(query);
        for (auto it = range.first; it != range.second; ++it) {
            expected.emplace_back(query, it->second);
        }
    }
    std::sort(expected.begin(), expected.end());
    REQUIRE(actual == expected);
}

TEMPLATE_TEST_CASE_SIG("static_multimap retrieve boundary test", "[static_multimap][retrieve][boundary]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto map = MakeMultimap<Key, Value>(128u, sg.stream);
    REQUIRE(map.Retrieve(nullptr, aclco::Extent<std::size_t>(0u), nullptr, nullptr, aclco::Extent<std::size_t>(0u),
                         sg.stream) == 0u);
}
