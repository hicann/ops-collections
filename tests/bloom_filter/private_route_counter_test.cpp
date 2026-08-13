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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "detail/bloom_filter/kernels.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::private_route_counter_detail {

constexpr std::uint32_t THREAD_NUM = 1024;
constexpr std::uint32_t CAPACITY = 80;
constexpr std::uint32_t RESULT_WORDS = THREAD_NUM + 1;

COLLECTION_SIMT_VF LAUNCH_BOUND(THREAD_NUM) inline void CounterSimt(__gm__ std::uint32_t* output,
                                                                    __ubuf__ std::uint32_t* counters,
                                                                    std::uint32_t activeThreads,
                                                                    std::uint32_t firstBucket,
                                                                    std::uint32_t spreadAcrossPackedWord)
{
    std::uint32_t const thread = AscendC::Simt::GetThreadIdx();
    std::uint32_t const threadCount = AscendC::Simt::GetThreadNum();
    for (std::uint32_t word = thread; word < detail::bloom_filter::privateRoutePackedCounterWords;
         word += threadCount) {
        counters[word] = 0U;
    }
    asc_syncthreads();

    std::uint32_t ticket = std::numeric_limits<std::uint32_t>::max();
    if (thread < activeThreads) {
        std::uint32_t const bucket = firstBucket + (spreadAcrossPackedWord != 0 ? thread & 3U : 0U);
        ticket = detail::bloom_filter::PrivatePackedRouteTicket<CAPACITY>(counters, bucket);
    }
    output[thread] = ticket;
    asc_syncthreads();

    if (thread == 0) {
        output[THREAD_NUM] = counters[firstBucket >> 2];
    }
}

COLLECTION_AIV_GLOBAL void PrivateRouteCounter(__gm__ std::uint32_t* output, std::uint32_t activeThreads,
                                               std::uint32_t firstBucket, std::uint32_t spreadAcrossPackedWord)
{
    __ubuf__ std::uint32_t counters[detail::bloom_filter::privateRoutePackedCounterWords];
    AscendC::Simt::VF_CALL<CounterSimt>(AscendC::Simt::Dim3{THREAD_NUM}, output, counters, activeThreads, firstBucket,
                                        spreadAcrossPackedWord);
}

} // namespace aclco::test::private_route_counter_detail

namespace {

std::vector<std::uint32_t> RunPrivateCounter(aclco::test::DeviceBuffer<std::uint32_t>& deviceOutput, aclrtStream stream,
                                             std::uint32_t activeThreads, std::uint32_t firstBucket,
                                             bool spreadAcrossPackedWord)
{
    aclco::test::private_route_counter_detail::PrivateRouteCounter<<<1, 0, stream>>>(
        deviceOutput.Data(), activeThreads, firstBucket, static_cast<std::uint32_t>(spreadAcrossPackedWord));
    return deviceOutput.CopyToHost(stream);
}

void RequireTickets(std::vector<std::uint32_t> const& output, std::uint32_t activeThreads, std::uint32_t firstLane,
                    std::uint32_t laneStride, std::uint32_t expectedOverflow)
{
    std::vector<std::uint32_t> retained;
    std::uint32_t overflow = 0;
    for (std::uint32_t thread = firstLane; thread < activeThreads; thread += laneStride) {
        if (output[thread] == std::numeric_limits<std::uint32_t>::max()) {
            ++overflow;
        } else {
            retained.push_back(output[thread]);
        }
    }
    std::sort(retained.begin(), retained.end());
    REQUIRE(retained.size() == aclco::test::private_route_counter_detail::CAPACITY);
    for (std::uint32_t ticket = 0; ticket < retained.size(); ++ticket) {
        CAPTURE(firstLane, ticket);
        REQUIRE(retained[ticket] == ticket);
    }
    REQUIRE(overflow == expectedOverflow);
}

} // namespace

TEST_CASE("BloomFilter private packed counters issue unique bounded tickets",
          "[bloom_filter][correctness][routed_add][private_counter]")
{
    using namespace aclco::test::private_route_counter_detail;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream const stream = streamGuard.stream;
    aclco::test::DeviceBuffer<std::uint32_t> output(RESULT_WORDS);

    SECTION("four counters sharing one U32 do not lose CAS updates")
    {
        constexpr std::uint32_t firstBucket = aclco::detail::bloom_filter::finePrivateRouteBucketCount - 4U;
        auto const observed = RunPrivateCounter(output, stream, 324, firstBucket, true);
        REQUIRE(observed[THREAD_NUM] == 0xd0d0d0d0U);
        for (std::uint32_t lane = 0; lane < 4; ++lane) {
            RequireTickets(observed, 324, lane, 4, 1);
        }
    }

    SECTION("ticket 79 fills the final bucket without setting overflow")
    {
        constexpr std::uint32_t lastBucket = aclco::detail::bloom_filter::finePrivateRouteBucketCount - 1U;
        auto const observed = RunPrivateCounter(output, stream, CAPACITY, lastBucket, false);
        REQUIRE(observed[THREAD_NUM] == 0x50000000U);
        RequireTickets(observed, CAPACITY, 0, 1, 0);
    }

    SECTION("capacity plus one sets overflow and a later launch clears UB")
    {
        constexpr std::uint32_t bucket = 1;
        auto observed = RunPrivateCounter(output, stream, CAPACITY + 1U, bucket, false);
        REQUIRE(observed[THREAD_NUM] == 0x0000d000U);
        RequireTickets(observed, CAPACITY + 1U, 0, 1, 1);

        observed = RunPrivateCounter(output, stream, CAPACITY, bucket, false);
        REQUIRE(observed[THREAD_NUM] == 0x00005000U);
        RequireTickets(observed, CAPACITY, 0, 1, 0);
    }
}
