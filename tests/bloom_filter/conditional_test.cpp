/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"
#include "tests/common/key_values.h"

namespace {

template <typename Key>
std::vector<Key> MakeConditionalKeys(std::size_t count)
{
    std::vector<Key> keys;
    keys.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        keys.push_back(static_cast<Key>(i * 2654435761ULL + 17ULL));
    }
    if constexpr (std::is_integral_v<Key> && std::is_signed_v<Key>) {
        aclco::test::SetSignedBoundaryValues(keys);
    }
    return keys;
}

template <typename Word>
std::vector<Word> CopyFilterWords(Word const* device, std::size_t count, aclrtStream stream)
{
    std::vector<Word> host(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(host.data(), host.size() * sizeof(Word), device, count * sizeof(Word),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter conditional D2H");
    aclco::test::Sync(stream);
    return host;
}

inline std::vector<std::uint8_t> MakeStencil(std::size_t count)
{
    std::vector<std::uint8_t> stencil(count);
    for (std::size_t i = 0; i < stencil.size(); ++i) {
        stencil[i] = static_cast<std::uint8_t>((i % 3) != 1 ? 1 : 0);
    }
    return stencil;
}

template <typename Key>
std::vector<Key> SelectKeys(std::vector<Key> const& keys, std::vector<std::uint8_t> const& stencil)
{
    std::vector<Key> selected;
    selected.reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        if (stencil[i] != 0) {
            selected.push_back(keys[i]);
        }
    }
    return selected;
}

template <typename Key, typename Filter>
std::vector<std::uint8_t> RequireSynchronousConditional(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceKeys,
                                                        aclco::test::DeviceBuffer<std::uint8_t>& deviceStencil,
                                                        aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput,
                                                        std::vector<Key> const& keys,
                                                        std::vector<std::uint8_t> const& stencil,
                                                        std::vector<Key> const& selected, std::size_t numBlocks,
                                                        aclrtStream stream)
{
    using Extent = typename Filter::ExtentType;
    filter.AddIf(deviceKeys.Data(), deviceStencil.Data(), Extent{keys.size()}, stream);

    std::vector<std::uint32_t> const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, selected);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            expectedWords);

    std::vector<std::uint8_t> outputInitial(keys.size(), 0xff);
    deviceOutput.CopyFromHostAsync(outputInitial.data(), outputInitial.size(), stream);
    std::as_const(filter).ContainsIf(deviceKeys.Data(), deviceStencil.Data(), deviceOutput.Data(), Extent{keys.size()},
                                     stream);
    std::vector<std::uint8_t> const output = deviceOutput.CopyToHost(stream);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        CAPTURE(i);
        if (stencil[i] == 0) {
            REQUIRE(output[i] == 0);
        } else {
            REQUIRE((output[i] != 0) == aclco::test::BloomGoldenContains(expectedWords, numBlocks, keys[i]));
        }
    }
    return outputInitial;
}

template <typename Key, typename Filter>
void RequireAsynchronousConditional(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceKeys,
                                    aclco::test::DeviceBuffer<std::uint8_t>& deviceStencil,
                                    aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput,
                                    std::vector<std::uint8_t> const& stencil,
                                    std::vector<std::uint8_t> const& outputInitial, aclrtStream stream)
{
    using Extent = typename Filter::ExtentType;
    filter.Clear(stream);
    filter.AddIfAsync(deviceKeys.Data(), deviceStencil.Data(), Extent{stencil.size()}, stream);
    deviceOutput.CopyFromHostAsync(outputInitial.data(), outputInitial.size(), stream);
    std::as_const(filter).ContainsIfAsync(deviceKeys.Data(), deviceStencil.Data(), deviceOutput.Data(),
                                          Extent{stencil.size()}, stream);
    auto const output = deviceOutput.CopyToHost(stream);
    for (std::size_t i = 0; i < stencil.size(); ++i) {
        CAPTURE(i);
        REQUIRE(output[i] == (stencil[i] != 0 ? 1 : 0));
    }
}

template <typename Key, typename Filter>
void RequireZeroStencil(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceKeys,
                        aclco::test::DeviceBuffer<std::uint8_t>& deviceStencil,
                        aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput,
                        std::vector<std::uint8_t> const& outputInitial, aclrtStream stream)
{
    using Extent = typename Filter::ExtentType;
    using Word = typename Filter::WordType;
    std::vector<std::uint8_t> zeroStencil(outputInitial.size(), 0);
    deviceStencil.CopyFromHostAsync(zeroStencil.data(), zeroStencil.size(), stream);
    filter.Clear(stream);
    filter.AddIf(deviceKeys.Data(), deviceStencil.Data(), Extent{zeroStencil.size()}, stream);
    std::vector<Word> const emptyWords = CopyFilterWords(std::as_const(filter).Data(),
                                                         static_cast<std::size_t>(filter.NumWords()), stream);
    REQUIRE(std::all_of(emptyWords.begin(), emptyWords.end(), [](Word word) { return word == Word{0}; }));

    deviceOutput.CopyFromHostAsync(outputInitial.data(), outputInitial.size(), stream);
    std::as_const(filter).ContainsIf(deviceKeys.Data(), deviceStencil.Data(), deviceOutput.Data(),
                                     Extent{zeroStencil.size()}, stream);
    auto const output = deviceOutput.CopyToHost(stream);
    REQUIRE(std::all_of(output.begin(), output.end(), [](std::uint8_t value) { return value == 0; }));
}

template <typename Key, typename Filter>
void RequireConditionalArguments(Filter& filter, aclco::test::DeviceBuffer<Key>& deviceKeys,
                                 aclco::test::DeviceBuffer<std::uint8_t>& deviceStencil,
                                 aclco::test::DeviceBuffer<std::uint8_t>& deviceOutput, aclrtStream stream)
{
    using Extent = typename Filter::ExtentType;
    filter.AddIf(nullptr, nullptr, Extent{0}, stream);
    filter.AddIfAsync(nullptr, nullptr, Extent{0}, stream);
    std::as_const(filter).ContainsIf(nullptr, nullptr, nullptr, Extent{0}, stream);
    std::as_const(filter).ContainsIfAsync(nullptr, nullptr, nullptr, Extent{0}, stream);

    REQUIRE_THROWS_AS(filter.AddIf(nullptr, deviceStencil.Data(), Extent{1}, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(filter.AddIf(deviceKeys.Data(), nullptr, Extent{1}, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(filter.AddIfAsync(nullptr, deviceStencil.Data(), Extent{1}, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(filter.AddIfAsync(deviceKeys.Data(), nullptr, Extent{1}, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).ContainsIf(nullptr, deviceStencil.Data(), deviceOutput.Data(), Extent{1}, stream),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).ContainsIf(deviceKeys.Data(), nullptr, deviceOutput.Data(), Extent{1}, stream),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        std::as_const(filter).ContainsIf(deviceKeys.Data(), deviceStencil.Data(), nullptr, Extent{1}, stream),
        std::invalid_argument);
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter AddIf and ContainsIf honor byte stencils", "[bloom_filter][conditional]", std::int32_t,
                   std::uint32_t, std::int64_t, std::uint64_t, float)
{
    using Key = TestType;
    using Filter = aclco::BloomFilter<Key>;
    using Extent = typename Filter::ExtentType;
    using Policy = typename Filter::PolicyType;
    using Allocator = typename Filter::AllocatorType;

    static_assert(std::is_constructible_v<Filter, Extent, Policy const&, aclrtStream>);
    static_assert(std::is_constructible_v<Filter, Extent, Policy const&, Allocator const&, aclrtStream>);

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 257;
    constexpr std::size_t keyCount = 1025;
    std::vector<Key> const keys = MakeConditionalKeys<Key>(keyCount);
    std::vector<std::uint8_t> const stencil = MakeStencil(keyCount);
    std::vector<Key> const selected = SelectKeys(keys, stencil);

    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    aclco::test::DeviceBuffer<std::uint8_t> deviceStencil(stencil.size());
    aclco::test::DeviceBuffer<std::uint8_t> deviceOutput(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    deviceStencil.CopyFromHostAsync(stencil.data(), stencil.size(), stream);

    Filter filter(Extent{numBlocks}, Policy{}, stream);
    auto const outputInitial = RequireSynchronousConditional(filter, deviceKeys, deviceStencil, deviceOutput, keys,
                                                             stencil, selected, numBlocks, stream);
    RequireAsynchronousConditional(filter, deviceKeys, deviceStencil, deviceOutput, stencil, outputInitial, stream);
    RequireZeroStencil(filter, deviceKeys, deviceStencil, deviceOutput, outputInitial, stream);
    RequireConditionalArguments(filter, deviceKeys, deviceStencil, deviceOutput, stream);
}

TEMPLATE_TEST_CASE("signed BloomFilter routed Add remains bit-exact", "[bloom_filter][signed][routed]", std::int32_t,
                   std::int64_t)
{
    using Key = TestType;
    using Filter = aclco::BloomFilter<Key>;
    using Extent = typename Filter::ExtentType;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    constexpr std::size_t numBlocks = 257;
    constexpr std::size_t keyCount = 20000;
    std::vector<Key> const keys = MakeConditionalKeys<Key>(keyCount);
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    Filter filter(Extent{numBlocks}, typename Filter::PolicyType{}, stream);
    filter.Add(deviceKeys.Data(), Extent{keys.size()}, stream);

    std::vector<std::uint32_t> const expectedWords = aclco::test::MakeBloomGolden<Key>(numBlocks, keys);
    REQUIRE(CopyFilterWords(std::as_const(filter).Data(), static_cast<std::size_t>(filter.NumWords()), stream) ==
            expectedWords);
}
