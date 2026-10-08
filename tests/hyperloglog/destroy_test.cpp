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

TEMPLATE_TEST_CASE_SIG("hyperloglog destroy and move lifecycle", "[hyperloglog][destroy]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    CAPTURE(sketch_kb);

    SECTION("repeated create add destroy")
    {
        for (int round = 0; round < 8; ++round) {
            auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
            auto keys = MakeKeys<Key>(4096u, 2048u, static_cast<uint64_t>(round + 1));
            auto estimate = AddAndEstimate(hll, keys, stream);
            REQUIRE(EstimateWithinAcceptance(estimate, ExactCardinality(keys), sketch_kb));
        }
    }

    SECTION("move construction transfers ownership")
    {
        auto source = MakeBySketchSize<Key>(sketch_kb, stream);
        auto keys = MakeKeys<Key>(1024u, 1024u);
        AddAndEstimate(source, keys, stream);
        auto moved = std::move(source);
        REQUIRE(EstimateWithinAcceptance(moved.Estimate(stream), ExactCardinality(keys), sketch_kb));
    }
}
