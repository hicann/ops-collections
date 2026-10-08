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
#include "common/hll_test_common.h"

using namespace aclco::test::hll_cases;

TEMPLATE_TEST_CASE_SIG("hyperloglog merge correctness", "[hyperloglog][merge]", ((typename Key, int Dummy), Key, Dummy),
                       (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    auto key_num = GENERATE(1u, 128u, 4096u, 65537u);
    CAPTURE(sketch_kb, key_num);

    auto merge_and_check = [&](std::vector<Key> const& left, std::vector<Key> const& right) {
        auto lhs = MakeBySketchSize<Key>(sketch_kb, stream);
        auto rhs = MakeBySketchSize<Key>(sketch_kb, stream);
        AddAndEstimate(lhs, left, stream);
        AddAndEstimate(rhs, right, stream);
        lhs.Merge(rhs, stream);

        std::unordered_set<Key> exact(left.begin(), left.end());
        exact.insert(right.begin(), right.end());
        REQUIRE(EstimateWithinAcceptance(lhs.Estimate(stream), exact.size(), sketch_kb));
    };

    SECTION("empty with empty and populated")
    {
        merge_and_check({}, {});
        merge_and_check({}, MakeKeys<Key>(key_num, key_num));
        merge_and_check(MakeKeys<Key>(key_num, key_num), {});
    }

    SECTION("same input is idempotent")
    {
        auto keys = MakeKeys<Key>(key_num, key_num);
        merge_and_check(keys, keys);
    }

    SECTION("overlapping and disjoint sketches")
    {
        auto left = MakeKeys<Key>(key_num, key_num, 1u);
        auto overlap = left;
        for (std::size_t i = overlap.size() / 2u; i < overlap.size(); ++i) {
            overlap[i] = static_cast<Key>(overlap[i] + static_cast<Key>(100000));
        }
        merge_and_check(left, overlap);

        auto disjoint = left;
        for (auto& key : disjoint) {
            key = static_cast<Key>(key + static_cast<Key>(200000));
        }
        merge_and_check(left, disjoint);
    }

    SECTION("incompatible sketches are rejected")
    {
        auto lhs = MakeBySketchSize<Key>(sketch_kb, stream);
        auto incompatible = MakeBySketchSize<Key>(sketch_kb == 8u ? 16u : 8u, stream);
        REQUIRE_THROWS(lhs.Merge(incompatible, stream));
    }
}
