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

TEMPLATE_TEST_CASE_SIG("static_multimap contains correctness", "[static_multimap][contains]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    TestStream context;
    auto stream = context.stream;
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto matching_rate = GENERATE(0.0, 0.1, 0.5, 1.0);
    auto distinct = std::max<std::size_t>(1u, capacity / (2u * multiplicity));
    CAPTURE(capacity, multiplicity, matching_rate, distinct);

    auto map = MakeMultimap<Key, Value>(capacity, stream);
    auto pairs = MakePairs<Key, Value>(distinct, multiplicity);
    auto oracle = MapOracle(pairs);
    auto d_pairs = InsertPairs(map, pairs, stream);

    auto queries = MakeQueries<Key>(distinct, std::min<std::size_t>(4096u, capacity), matching_rate);
    aclco::test::DeviceBuffer<Key> d_queries(queries.size());
    aclco::test::DeviceBuffer<unsigned char> d_output(queries.size());
    d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
    d_output.MemsetZero(stream);
    map.Contains(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);

    auto actual = d_output.CopyToHost(stream);
    REQUIRE(actual.size() == queries.size());
    for (std::size_t i = 0; i < queries.size(); ++i) {
        CAPTURE(i, queries[i]);
        REQUIRE((actual[i] != 0) == (oracle.count(queries[i]) != 0));
    }
}

TEMPLATE_TEST_CASE_SIG("static_multimap contains negative test", "[static_multimap][contains][negative]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    TestStream context;
    auto stream = context.stream;
    auto map = MakeMultimap<Key, Value>(128u, stream);
    auto pairs = MakePairs<Key, Value>(5u, 2u);
    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(pairs.size());
    d_pairs.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
    REQUIRE(map.Insert(d_pairs.Data(), aclco::Extent<std::size_t>(pairs.size()), stream) == 0u);

    SECTION("duplicate queries retain output positions")
    {
        std::vector<Key> queries(7u, static_cast<Key>(1));
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        aclco::test::DeviceBuffer<unsigned char> d_output(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        map.Contains(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);
        for (auto flag : d_output.CopyToHost(stream)) {
            REQUIRE(flag != 0u);
        }
    }

    SECTION("zero extent accepts null buffers")
    {
        map.Contains(nullptr, nullptr, aclco::Extent<std::size_t>(0u), stream);
        REQUIRE(map.Size(stream) == pairs.size());
    }
}
