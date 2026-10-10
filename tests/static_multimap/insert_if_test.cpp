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

TEMPLATE_TEST_CASE_SIG("static_multimap insert_if correctness", "[static_multimap][insert_if]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto capacity = GENERATE(128u, 1027u, 100000u);
    auto stencil_mode = GENERATE(0u, 1u, 2u);
    auto count = std::min<std::size_t>(capacity / 2u, 4096u);
    auto map = MakeMultimap<Key, Value>(capacity, stream);
    auto pairs = MakePairs<Key, Value>(count, 1u);
    std::vector<uint32_t> stencil(count);
    for (std::size_t i = 0; i < count; ++i) {
        stencil[i] = stencil_mode == 0u ? 0u : (stencil_mode == 1u ? 1u : static_cast<uint32_t>(i));
    }

    aclco::test::DeviceBuffer<aclco::Pair<Key, Value>> d_pairs(count);
    aclco::test::DeviceBuffer<uint32_t> d_stencil(count);
    d_pairs.CopyFromHostAsync(pairs.data(), count, stream);
    d_stencil.CopyFromHostAsync(stencil.data(), count, stream);
    auto failed = map.template InsertIf<uint32_t, IsOdd>(static_cast<void*>(d_pairs.Data()), d_stencil.Data(),
                                                         aclco::Extent<std::size_t>(count), stream);
    REQUIRE(failed == 0u);

    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i)
        keys[i] = pairs[i].first;
    aclco::test::DeviceBuffer<Key> d_keys(count);
    aclco::test::DeviceBuffer<unsigned char> d_output(count);
    d_keys.CopyFromHostAsync(keys.data(), count, stream);
    d_output.MemsetZero(stream);
    map.Contains(d_keys.Data(), d_output.Data(), aclco::Extent<std::size_t>(count), stream);
    auto actual = d_output.CopyToHost(stream);
    for (std::size_t i = 0; i < count; ++i) {
        CAPTURE(i, capacity);
        REQUIRE((actual[i] != 0) == (stencil[i] % 2u != 0u));
    }
}

TEMPLATE_TEST_CASE_SIG("static_multimap insert_if boundary test", "[static_multimap][insert_if][boundary]",
                       ((typename Key, typename Value, int Dummy), Key, Value, Dummy), (int32_t, int32_t, 0),
                       (int64_t, int64_t, 0))
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto map = MakeMultimap<Key, Value>(128u, sg.stream);
    REQUIRE(map.template InsertIf<uint32_t, IsOdd>(nullptr, nullptr, aclco::Extent<std::size_t>(0u), sg.stream) == 0u);
    REQUIRE(map.Size(sg.stream) == 0u);
}
