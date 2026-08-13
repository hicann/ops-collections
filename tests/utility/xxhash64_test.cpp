/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <acl/acl.h>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

#include "hash_functions.h"
#include "kernel_operator.h"
#include "macros.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"
#include "tests/common/object_representation.h"

namespace aclco::test::xxhash64_device_detail {

constexpr std::uint32_t THREAD_NUM = 1024;

template <typename Key>
COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM) inline void HashSimt(__gm__ std::uint8_t* keys, __gm__ std::uint8_t* output,
                                                                 std::uint64_t count, std::uint64_t seed)
{
    __gm__ Key* input = reinterpret_cast<__gm__ Key*>(keys);
    __gm__ std::uint64_t* hashes = reinterpret_cast<__gm__ std::uint64_t*>(output);
    aclco::xxhash_64<Key> const hash{seed};

    std::uint64_t index = static_cast<std::uint64_t>(AscendC::Simt::GetBlockIdx()) * AscendC::Simt::GetThreadNum() +
                          AscendC::Simt::GetThreadIdx();
    std::uint64_t const stride = static_cast<std::uint64_t>(AscendC::Simt::GetBlockNum()) *
                                 AscendC::Simt::GetThreadNum();
    for (; index < count; index += stride) {
        Key const key = input[index];
        hashes[index] = hash(key);
    }
}

template <typename Key>
COLLECTION_AIV_GLOBAL void XXHash64Device(__gm__ std::uint8_t* keys, __gm__ std::uint8_t* output, std::uint64_t count,
                                          std::uint64_t seed)
{
    AscendC::Simt::VF_CALL<HashSimt<Key>>(AscendC::Simt::Dim3{THREAD_NUM}, keys, output, count, seed);
}

} // namespace aclco::test::xxhash64_device_detail

namespace {

template <typename Key>
std::vector<Key> DeviceHashValues()
{
    if constexpr (std::is_same_v<Key, std::int32_t>) {
        return {
            0, 1, -1, -123456789, std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()};
    } else if constexpr (std::is_same_v<Key, std::uint32_t>) {
        return {0u, 1u, 42u, 123456789u, 0x80000000u, 0xffffffffu};
    } else if constexpr (std::is_same_v<Key, std::int64_t>) {
        return {
            0, 1, -1, -123456789, std::numeric_limits<std::int64_t>::min(), std::numeric_limits<std::int64_t>::max()};
    } else if constexpr (std::is_same_v<Key, std::uint64_t>) {
        return {0ull, 1ull, 42ull, 123456789ull, 0x0123456789abcdefull, 0x8000000000000000ull, 0xffffffffffffffffull};
    } else {
        static_assert(std::is_same_v<Key, float>);
        std::vector<std::uint32_t> const bits = {0x00000000u, 0x80000000u, 0x3f800000u, 0xbf800000u,
                                                 0x7f800000u, 0xff800000u, 0x7fc00001u, 0x7fc01234u};
        std::vector<float> values(bits.size());
        for (std::size_t i = 0; i < bits.size(); ++i) {
            values[i] = aclco::test::ObjectRepresentationCast<float>(bits[i]);
        }
        return values;
    }
}

} // namespace

TEST_CASE("xxhash_64 matches canonical cuCollections vectors", "[utility][hash][xxhash64]")
{
    REQUIRE(aclco::xxhash_64<std::uint32_t>{0}(0u) == 4246796580750024372ull);
    REQUIRE(aclco::xxhash_64<std::uint32_t>{42}(0u) == 3614696996920510707ull);
    REQUIRE(aclco::xxhash_64<std::uint32_t>{0}(42u) == 15516826743637085169ull);
    REQUIRE(aclco::xxhash_64<std::uint32_t>{0}(123456789u) == 9462334144942111946ull);

    REQUIRE(aclco::xxhash_64<std::uint64_t>{0}(0ull) == 3803688792395291579ull);
    REQUIRE(aclco::xxhash_64<std::uint64_t>{42}(0ull) == 13194218611613725804ull);
    REQUIRE(aclco::xxhash_64<std::uint64_t>{0}(42ull) == 13066772586158965587ull);
    REQUIRE(aclco::xxhash_64<std::uint64_t>{0}(123456789ull) == 14662639848940634189ull);

    REQUIRE(aclco::test::bloom_golden_detail::XXHash64(std::uint32_t{0}, 0) == 4246796580750024372ull);
    REQUIRE(aclco::test::bloom_golden_detail::XXHash64(std::uint32_t{0}, 42) == 3614696996920510707ull);
    REQUIRE(aclco::test::bloom_golden_detail::XXHash64(std::uint64_t{0}, 0) == 3803688792395291579ull);
    REQUIRE(aclco::test::bloom_golden_detail::XXHash64(std::uint64_t{0}, 42) == 13194218611613725804ull);
}

TEMPLATE_TEST_CASE("xxhash_64 signed fast path preserves object representation", "[utility][hash][xxhash64][signed]",
                   std::int32_t, std::int64_t)
{
    using Key = TestType;
    for (Key const value : DeviceHashValues<Key>()) {
        for (std::uint64_t const seed : {std::uint64_t{0}, std::uint64_t{42}}) {
            CAPTURE(value, seed);
            REQUIRE(aclco::xxhash_64<Key>{seed}(value) == aclco::test::bloom_golden_detail::XXHash64(value, seed));
        }
    }
}

TEST_CASE("xxhash_64 preserves float object representation", "[utility][hash][xxhash64]")
{
    std::uint32_t const nanBits1 = 0x7fc00001u;
    std::uint32_t const nanBits2 = 0x7fc01234u;
    float const nan1 = aclco::test::ObjectRepresentationCast<float>(nanBits1);
    float const nan2 = aclco::test::ObjectRepresentationCast<float>(nanBits2);
    std::vector<float> const values = {
        0.0f, -0.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(), nan1, nan2};

    for (float value : values) {
        REQUIRE(aclco::xxhash_64<float>{0}(value) == aclco::test::bloom_golden_detail::XXHash64(value, 0));
        REQUIRE(aclco::xxhash_64<float>{42}(value) == aclco::test::bloom_golden_detail::XXHash64(value, 42));
    }
}

TEMPLATE_TEST_CASE("xxhash_64 device matches independent golden", "[utility][hash][xxhash64][device]", std::int32_t,
                   std::uint32_t, std::int64_t, std::uint64_t, float)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream const stream = streamGuard.stream;

    std::vector<Key> const keys = DeviceHashValues<Key>();
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    aclco::test::DeviceBuffer<std::uint64_t> deviceHashes(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);

    for (std::uint64_t const seed : {std::uint64_t{0}, std::uint64_t{42}}) {
        aclco::test::xxhash64_device_detail::XXHash64Device<Key><<<1, 0, stream>>>(
            reinterpret_cast<std::uint8_t*>(deviceKeys.Data()), reinterpret_cast<std::uint8_t*>(deviceHashes.Data()),
            static_cast<std::uint64_t>(keys.size()), seed);
        aclco::test::Sync(stream);
        std::vector<std::uint64_t> const observed = deviceHashes.CopyToHost(stream);

        REQUIRE(observed.size() == keys.size());
        for (std::size_t i = 0; i < keys.size(); ++i) {
            CAPTURE(seed, i);
            REQUIRE(observed[i] == aclco::test::bloom_golden_detail::XXHash64(keys[i], seed));
        }
    }
}
