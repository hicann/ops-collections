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

namespace {
struct IsOdd {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const noexcept { return value % 2u != 0u; }
};
} // namespace

TEMPLATE_TEST_CASE_SIG("static_multimap find_if correctness", "[static_multimap][find_if]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto matching_rate = GENERATE(0.0, 0.5, 1.0);
    auto stencil_mode = GENERATE(0u, 1u, 2u);
    auto count = std::min<std::size_t>(capacity / 2u, 4096u);
    StencilFixture<Key, Value> fixture(capacity, count, matching_rate, stencil_mode);
    aclco::test::DeviceBuffer<Value> d_output(count);
    d_output.MemsetZero(fixture.context.stream);
    fixture.map.template FindIf<uint32_t, IsOdd>(static_cast<void*>(fixture.deviceKeys.Data()),
                                                 fixture.deviceStencil.Data(), static_cast<void*>(d_output.Data()),
                                                 aclco::Extent<std::size_t>(count), fixture.context.stream);
    auto actual = d_output.CopyToHost(fixture.context.stream);
    for (std::size_t i = 0; i < count; ++i) {
        const bool active_hit = fixture.stencil[i] % 2u != 0u && fixture.keys[i] <= Key(count);
        if (!active_hit) {
            REQUIRE(actual[i] == EmptyValue<Value>());
        } else {
            REQUIRE(actual[i] == Value(fixture.keys[i] * Key(1024) + Key(1)));
        }
    }
}

TEMPLATE_TEST_CASE_SIG("static_multimap find_if boundary test", "[static_multimap][find_if][boundary]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto map = MakeMultimap<Key, Value>(128u, sg.stream);
    map.template FindIf<uint32_t, IsOdd>(nullptr, nullptr, nullptr, aclco::Extent<std::size_t>(0u), sg.stream);
    REQUIRE(map.Size(sg.stream) == 0u);
}
