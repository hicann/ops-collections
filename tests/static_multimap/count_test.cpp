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

TEMPLATE_TEST_CASE_SIG("static_multimap count correctness", "[static_multimap][count]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto multiplicity = GENERATE(1u, 2u, 4u, 8u);
    auto matching_rate = GENERATE(0.0, 0.1, 0.5, 1.0);
    CAPTURE(capacity, multiplicity, matching_rate);

    QueryFixture<Key, Value> fixture(capacity, multiplicity, matching_rate);
    auto actual = fixture.map.Count(fixture.deviceQueries.Data(), aclco::Extent<std::size_t>(fixture.queries.size()),
                                    fixture.context.stream);
    REQUIRE(actual == ExpectedCount(fixture.oracle, fixture.queries));
}

TEMPLATE_TEST_CASE_SIG("static_multimap count negative test", "[static_multimap][count][negative]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    TestStream context;
    auto stream = context.stream;
    constexpr std::size_t multiplicity = 4u;
    auto map = MakeMultimap<Key, Value>(128u, stream);
    auto pairs = MakePairs<Key, Value>(5u, multiplicity);
    auto d_pairs = InsertPairs(map, pairs, stream);

    SECTION("duplicate queries preserve multiplicity")
    {
        std::vector<Key> duplicate_queries(7u, static_cast<Key>(1));
        aclco::test::DeviceBuffer<Key> d_duplicate(duplicate_queries.size());
        d_duplicate.CopyFromHostAsync(duplicate_queries.data(), duplicate_queries.size(), stream);
        REQUIRE(map.Count(d_duplicate.Data(), aclco::Extent<std::size_t>(duplicate_queries.size()), stream) ==
                7u * multiplicity);
    }

    SECTION("zero extent accepts null input")
    {
        REQUIRE(map.Count(nullptr, aclco::Extent<std::size_t>(0u), stream) == 0u);
    }
}
