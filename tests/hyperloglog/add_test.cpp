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

TEMPLATE_TEST_CASE_SIG("hyperloglog add correctness", "[hyperloglog][add]", ((typename Key, int Dummy), Key, Dummy),
                       (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    auto key_num = GENERATE(0u, 1u, 2u, 3u, 7u, 31u, 128u, 1024u, 4096u, 8195u, 32768u, 131072u);
    CAPTURE(sketch_kb, key_num);

    SECTION("unique and duplicate distributions")
    {
        auto distinct = GENERATE_COPY(std::max<std::size_t>(key_num, 1u), std::max<std::size_t>(key_num / 2u, 1u),
                                      std::max<std::size_t>(key_num / 8u, 1u));
        auto keys = MakeKeys<Key>(key_num, distinct, 1u);
        auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
        auto estimate = AddAndEstimate(hll, keys, stream);
        REQUIRE(EstimateWithinAcceptance(estimate, ExactCardinality(keys), sketch_kb));
    }

    SECTION("batched add equals one-shot estimate")
    {
        auto keys = MakeKeys<Key>(key_num, std::max<std::size_t>(key_num / 2u, 1u), 17u);
        auto one_shot = MakeBySketchSize<Key>(sketch_kb, stream);
        auto batched = MakeBySketchSize<Key>(sketch_kb, stream);
        auto expected = AddAndEstimate(one_shot, keys, stream);

        auto split = keys.size() / 2u;
        std::vector<Key> left(keys.begin(), keys.begin() + split);
        std::vector<Key> right(keys.begin() + split, keys.end());
        AddAndEstimate(batched, left, stream);
        auto actual = AddAndEstimate(batched, right, stream);
        REQUIRE(actual == expected);
    }

    SECTION("null pointer with nonzero count is rejected")
    {
        auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
        REQUIRE_THROWS(hll.Add(nullptr, aclco::Extent<std::size_t>(1u), stream));
    }
}

TEMPLATE_TEST_CASE_SIG("hyperloglog add large generalization", "[hyperloglog][add][large]",
                       ((typename Key, int Dummy), Key, Dummy), (int32_t, 0), (int64_t, 0))
{
    (void)aclco::test::hll_cases::GlobalAcl();
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto sketch_kb = GENERATE(8u, 16u, 32u, 64u, 128u, 256u);
    constexpr std::size_t key_num = 100000000ULL;
    auto keys = MakeKeys<Key>(key_num, key_num, 2026u);
    auto hll = MakeBySketchSize<Key>(sketch_kb, stream);
    auto estimate = AddAndEstimate(hll, keys, stream);
    REQUIRE(EstimateWithinAcceptance(estimate, key_num, sketch_kb));

    auto batched = MakeBySketchSize<Key>(sketch_kb, stream);
    aclco::test::DeviceBuffer<Key> input(keys.size());
    input.CopyFromHostAsync(keys.data(), keys.size(), stream);
    auto split = keys.size() / 2u;
    batched.Add(input.Data(), aclco::Extent<std::size_t>(split), stream);
    batched.Add(input.Data() + split, aclco::Extent<std::size_t>(keys.size() - split), stream);
    REQUIRE(batched.Estimate(stream) == estimate);
}
