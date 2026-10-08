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

TEMPLATE_TEST_CASE_SIG("hyperloglog estimate accuracy and repeatability", "[hyperloglog][estimate]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    auto cardinality = GENERATE(0u, 1u, 2u, 3u, 7u, 128u, 1024u, 4096u, 65537u, 1048576u);
    CAPTURE(sketch_kb, cardinality);

    auto keys = MakeKeys<Key>(cardinality * 2ULL, cardinality, 2026u);
    auto first = MakeBySketchSize<Key>(sketch_kb, stream);
    auto second = MakeBySketchSize<Key>(sketch_kb, stream);
    auto estimate_a = AddAndEstimate(first, keys, stream);
    auto estimate_b = AddAndEstimate(second, keys, stream);

    REQUIRE(estimate_a == estimate_b);
    REQUIRE(EstimateWithinAcceptance(estimate_a, cardinality, sketch_kb));

    SECTION("repeated estimate does not mutate state")
    {
        REQUIRE(first.Estimate(stream) == estimate_a);
        REQUIRE(first.Estimate(stream) == estimate_a);
    }
}
