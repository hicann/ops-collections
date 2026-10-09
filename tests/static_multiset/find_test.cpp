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
#include "common/static_multiset_test_common.h"

using namespace aclco::test::static_multiset;

TEMPLATE_TEST_CASE_SIG("static_multiset find correctness", "[static_multiset][find]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    TestEnvironment environment;
    auto stream = environment.Stream();
    auto [capacity, multiplicity, matching_rate, distinct] = GenerateLookupParameters();
    CAPTURE(capacity, multiplicity, matching_rate);

    auto [set, keys, oracle, d_keys, queries] = QueryInput<Key>(
        capacity, distinct, multiplicity, std::min<std::size_t>(4096u, capacity), matching_rate, stream);
    aclco::test::DeviceBuffer<Key> d_queries(queries.size());
    aclco::test::DeviceBuffer<Key> d_output(queries.size());
    d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
    set.Find(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);

    auto actual = d_output.CopyToHost(stream);
    for (std::size_t i = 0; i < queries.size(); ++i) {
        auto expected = oracle.count(queries[i]) == 0u ? EmptyKey<Key>() : queries[i];
        CAPTURE(i, queries[i]);
        REQUIRE(actual[i] == expected);
    }
}

TEMPLATE_TEST_CASE_SIG("static_multiset find negative test", "[static_multiset][find][negative]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto set = MakeMultiset<Key>(128u, stream);
    auto keys = MakeRepeatedKeys<Key>(5u, 2u);
    auto d_keys = InsertKeys(set, keys, stream);

    SECTION("duplicate queries return the queried key")
    {
        std::vector<Key> queries(7u, static_cast<Key>(1));
        aclco::test::DeviceBuffer<Key> d_queries(queries.size());
        aclco::test::DeviceBuffer<Key> d_output(queries.size());
        d_queries.CopyFromHostAsync(queries.data(), queries.size(), stream);
        set.Find(d_queries.Data(), d_output.Data(), aclco::Extent<std::size_t>(queries.size()), stream);
        for (auto key : d_output.CopyToHost(stream)) {
            REQUIRE(key == static_cast<Key>(1));
        }
    }

    SECTION("zero extent accepts null buffers")
    {
        set.Find(nullptr, nullptr, aclco::Extent<std::size_t>(0u), stream);
        REQUIRE(set.Size(stream) == keys.size());
    }
}
