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

TEMPLATE_TEST_CASE_SIG("hyperloglog create correctness", "[hyperloglog][create]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;

    SECTION("create by every supported sketch size")
    {
        auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
        CAPTURE(sketch_kb);
        auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
        REQUIRE(hll.SketchSizeKB() == sketch_kb);
        REQUIRE(hll.Estimate(stream) == 0u);
    }

    SECTION("create by standard deviation")
    {
        auto deviation = GENERATE(0.0115, 0.0082, 0.0058, 0.0041, 0.0029, 0.0021);
        CAPTURE(deviation);
        auto hll = MakeByStandardDeviation<Key>(deviation, stream);
        REQUIRE(hll.Estimate(stream) == 0u);
    }

    SECTION("create by precision")
    {
        auto precision = GENERATE(13u, 14u, 15u, 16u, 17u, 18u);
        CAPTURE(precision);
        auto hll = MakeByPrecision<Key>(precision, stream);
        REQUIRE(hll.Estimate(stream) == 0u);
    }

    SECTION("reject unsupported configuration")
    {
        REQUIRE_THROWS(MakeBySketchSize<Key>(0u, stream));
        REQUIRE_THROWS(MakeBySketchSize<Key>(7u, stream));
        REQUIRE_THROWS(MakeBySketchSize<Key>(512u, stream));
        REQUIRE_THROWS(MakeByPrecision<Key>(0u, stream));
    }
}
