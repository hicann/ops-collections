/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/object_representation.h"

#include "pair.h"
#include "detail/open_addressing/kernels.h"
#include "utility/kernel_launch_utils.h"

// Use fixed-width types because plain char has target-dependent signedness.
using SignedBytePair = aclco::Pair<std::int8_t, std::uint8_t>;
static_assert(sizeof(SignedBytePair) == sizeof(std::uint16_t));

extern "C" COLLECTION_AIV_GLOBAL void CreatePair(__gm__ std::uint16_t* pairAddr)
{
    SignedBytePair devicePair = aclco::MakePair<std::int8_t, std::uint8_t>(static_cast<std::int8_t>(-48),
                                                                           static_cast<std::uint8_t>(48));
    auto* packed = reinterpret_cast<std::uint16_t*>(&devicePair);
    AscendC::WriteGmByPassDCache<std::uint16_t>(pairAddr, *packed);
}

struct EmptyType {
    // Empty type used by the alignment test below.
};

TEST_CASE("Pair construction and equality (host)", "[utility][pair][host]")
{
    using First = std::int8_t;
    using Second = std::uint8_t;

    aclco::Pair<First, Second> p1 = aclco::MakePair<First, Second>(static_cast<First>(-48), static_cast<Second>(48));
    aclco::Pair<First, Second> p2 = aclco::MakePair<First, Second>(static_cast<First>(-48), static_cast<Second>(48));

    REQUIRE(static_cast<int>(p1.first) == -48);
    REQUIRE(static_cast<int>(p1.second) == 48);
    REQUIRE(p1 == p2);
}

TEST_CASE("Pair construction on device and D2H copy (device)", "[utility][pair][device]")
{
    aclco::test::AclGlobalGuard g_acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;

    using First = std::int8_t;
    using Second = std::uint8_t;
    using P = aclco::Pair<First, Second>;

    const int expectedFirst = -48;
    const int expectedSecond = 48;

    constexpr std::size_t wordCount = 1;
    aclco::test::DeviceBuffer<std::uint16_t> dOut(wordCount);
    dOut.MemsetZero(stream);

    CreatePair<<<1, nullptr, stream>>>(dOut.Data());
    aclco::test::Sync(stream);

    auto hostWords = dOut.CopyToHost(stream);
    REQUIRE(hostWords.size() == wordCount);

    P const hostPair = aclco::test::ObjectRepresentationCast<P>(hostWords[0]);

    REQUIRE(static_cast<int>(hostPair.first) == expectedFirst);
    REQUIRE(static_cast<int>(hostPair.second) == expectedSecond);
}

TEST_CASE("PairAlignment returns expected alignment", "[utility][pair][alignment]")
{
    // char + unsigned char
    {
        using T1 = char;
        using T2 = unsigned char;
        constexpr std::size_t expected = 2;
        const std::size_t actual = aclco::PairAlignment<T1, T2>();
        CAPTURE(expected, actual);
        REQUIRE(actual == expected);
    }

    // int8_t + int16_t
    {
        using T3 = int8_t;
        using T4 = int16_t;
        constexpr std::size_t expected = 4;
        const std::size_t actual = aclco::PairAlignment<T3, T4>();
        CAPTURE(expected, actual);
        REQUIRE(actual == expected);
    }

    // double + int32_t
    {
        using T5 = double;
        using T6 = int32_t;
        constexpr std::size_t expected = 16;
        const std::size_t actual = aclco::PairAlignment<T5, T6>();
        CAPTURE(expected, actual);
        REQUIRE(actual == expected);
    }

    // EmptyType + EmptyType
    {
        constexpr std::size_t expected = 2;
        const std::size_t actual = aclco::PairAlignment<EmptyType, EmptyType>();
        CAPTURE(expected, actual);
        REQUIRE(actual == expected);
    }
}
