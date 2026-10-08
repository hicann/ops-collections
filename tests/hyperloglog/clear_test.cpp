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

TEMPLATE_TEST_CASE_SIG("hyperloglog clear correctness", "[hyperloglog][clear]", ((typename Key, int Dummy), Key, Dummy),
                       (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    auto key_num = GENERATE(1u, 3u, 128u, 4096u, 65537u);
    CAPTURE(sketch_kb, key_num);

    auto hll = MakeBySketchSize<Key>(sketch_kb, stream);

    SECTION("clear empty is idempotent")
    {
        hll.Clear(stream);
        hll.Clear(stream);
        REQUIRE(hll.Estimate(stream) == 0u);
    }

    SECTION("clear populated sketch")
    {
        auto keys = MakeKeys<Key>(key_num, std::max<std::size_t>(1u, key_num / 2u));
        AddAndEstimate(hll, keys, stream);
        hll.Clear(stream);
        REQUIRE(hll.Estimate(stream) == 0u);
    }

    SECTION("clear then reuse with disjoint input")
    {
        auto first = MakeKeys<Key>(key_num, std::max<std::size_t>(1u, key_num / 2u), 1u);
        AddAndEstimate(hll, first, stream);
        hll.Clear(stream);

        auto second = MakeKeys<Key>(key_num, std::max<std::size_t>(1u, key_num), 17u);
        for (auto& key : second) {
            key = static_cast<Key>(key + static_cast<Key>(100000));
        }
        auto estimate = AddAndEstimate(hll, second, stream);
        REQUIRE(EstimateWithinAcceptance(estimate, ExactCardinality(second), sketch_kb));
    }
}
