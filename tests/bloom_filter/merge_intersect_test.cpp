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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bloom_filter.h"
#include "tests/common/acl_env.h"
#include "tests/common/bloom_filter_golden.h"
#include "tests/common/device_buffer.h"

namespace {

template <typename Key>
std::vector<Key> Sequence(std::uint64_t begin, std::uint64_t count)
{
    std::vector<Key> values;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        values.push_back(static_cast<Key>(begin + i));
    }
    return values;
}

template <typename Word>
std::vector<Word> ReadWords(Word const* data, std::size_t count, aclrtStream stream)
{
    std::vector<Word> result(count);
    aclco::test::CheckAcl(aclrtMemcpyAsync(result.data(), result.size() * sizeof(Word), data, count * sizeof(Word),
                                           ACL_MEMCPY_DEVICE_TO_HOST, stream),
                          "aclrtMemcpyAsync BloomFilter combine D2H");
    aclco::test::Sync(stream);
    return result;
}

template <typename Filter, typename Key>
void AddKeys(Filter& filter, std::vector<Key> const& keys, aclrtStream stream)
{
    aclco::test::DeviceBuffer<Key> device(keys.size());
    device.CopyFromHostAsync(keys.data(), keys.size(), stream);
    filter.Add(device.Data(), aclco::Extent<std::size_t>{keys.size()}, stream);
}

enum class CombineOperation { Merge, Intersect };

template <CombineOperation Operation, typename Word>
std::vector<Word> CombineGolden(std::vector<Word> const& first, std::vector<Word> const& second)
{
    std::vector<Word> result(first.size());
    for (std::size_t i = 0; i < result.size(); ++i) {
        if constexpr (Operation == CombineOperation::Merge) {
            result[i] = first[i] | second[i];
        } else {
            result[i] = first[i] & second[i];
        }
    }
    return result;
}

template <typename Key>
struct CombineFilters {
    using Filter = aclco::BloomFilter<Key>;
    using Extent = typename Filter::ExtentType;
    using Policy = typename Filter::PolicyType;
    using Allocator = typename Filter::AllocatorType;

    CombineFilters(std::size_t numBlocks, aclrtStream stream)
        : destination(Extent{numBlocks}, Policy{}, Allocator{}, stream),
          source(Extent{numBlocks}, Policy{}, Allocator{}, stream)
    {}

    Filter destination;
    Filter source;
};

template <CombineOperation Operation, bool Async, typename Key, typename Word>
void VerifyCombine(CombineFilters<Key>& filters, std::vector<Key> const& destinationKeys,
                   std::vector<Key> const& sourceKeys, std::vector<Word> const& expected,
                   std::vector<Word> const& sourceExpected, aclrtStream stream)
{
    AddKeys(filters.destination, destinationKeys, stream);
    AddKeys(filters.source, sourceKeys, stream);
    if constexpr (Operation == CombineOperation::Merge) {
        if constexpr (Async) {
            filters.destination.MergeAsync(filters.source, stream);
        } else {
            filters.destination.Merge(filters.source, stream);
        }
    } else if constexpr (Async) {
        filters.destination.IntersectAsync(filters.source, stream);
    } else {
        filters.destination.Intersect(filters.source, stream);
    }
    if constexpr (Async) {
        aclco::test::Sync(stream);
    }
    REQUIRE(ReadWords(filters.destination.Data(), filters.destination.NumWords(), stream) == expected);
    REQUIRE(ReadWords(filters.source.Data(), filters.source.NumWords(), stream) == sourceExpected);
    if constexpr (!Async) {
        if constexpr (Operation == CombineOperation::Merge) {
            filters.destination.Merge(filters.destination, stream);
        } else {
            filters.destination.Intersect(filters.destination, stream);
        }
        REQUIRE(ReadWords(filters.destination.Data(), filters.destination.NumWords(), stream) == expected);
    }
}

} // namespace

TEMPLATE_TEST_CASE("BloomFilter Merge and Intersect match word-wise golden", "[bloom_filter][combine]", std::uint32_t,
                   std::uint64_t, float)
{
    using Key = TestType;
    using Filter = aclco::BloomFilter<Key>;
    using Word = typename Filter::WordType;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;
    constexpr std::size_t numBlocks = 509;

    auto keysA = Sequence<Key>(0, 2048);
    auto keysB = Sequence<Key>(1024, 2048);
    auto goldenA = aclco::test::MakeBloomGolden<Key>(numBlocks, keysA);
    auto goldenB = aclco::test::MakeBloomGolden<Key>(numBlocks, keysB);
    auto expectedMerge = CombineGolden<CombineOperation::Merge>(goldenA, goldenB);
    auto expectedIntersect = CombineGolden<CombineOperation::Intersect>(goldenA, goldenB);

    CombineFilters<Key> mergeFilters(numBlocks, stream);
    VerifyCombine<CombineOperation::Merge, false>(mergeFilters, keysA, keysB, expectedMerge, goldenB, stream);
    CombineFilters<Key> intersectFilters(numBlocks, stream);
    VerifyCombine<CombineOperation::Intersect, false>(intersectFilters, keysA, keysB, expectedIntersect, goldenB,
                                                      stream);
    CombineFilters<Key> asyncMergeFilters(numBlocks, stream);
    VerifyCombine<CombineOperation::Merge, true>(asyncMergeFilters, keysA, keysB, expectedMerge, goldenB, stream);
    CombineFilters<Key> asyncIntersectFilters(numBlocks, stream);
    VerifyCombine<CombineOperation::Intersect, true>(asyncIntersectFilters, keysA, keysB, expectedIntersect, goldenB,
                                                     stream);

    Filter incompatible(aclco::Extent<std::size_t>{numBlocks + 1}, typename Filter::PolicyType{},
                        typename Filter::AllocatorType{}, stream);
    REQUIRE_THROWS_AS(mergeFilters.destination.Merge(incompatible, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(mergeFilters.destination.MergeAsync(incompatible, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(intersectFilters.destination.Intersect(incompatible, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(intersectFilters.destination.IntersectAsync(incompatible, stream), std::invalid_argument);
}

TEST_CASE("BloomFilter combine rejects a different hash seed", "[bloom_filter][combine][validation]")
{
    using Key = std::uint32_t;
    using Policy = aclco::BloomFilterPolicy<Key>;
    using Filter = aclco::BloomFilter<Key, aclco::Extent<std::size_t>, Policy>;

    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    aclrtStream stream = streamGuard.stream;

    Policy seedOne(Policy::Hasher{1});
    Policy seedTwo(Policy::Hasher{2});
    Filter first(aclco::Extent<std::size_t>{16}, seedOne, Filter::AllocatorType{}, stream);
    Filter second(aclco::Extent<std::size_t>{16}, seedTwo, Filter::AllocatorType{}, stream);
    REQUIRE_THROWS_AS(first.Merge(second, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(first.MergeAsync(second, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(first.Intersect(second, stream), std::invalid_argument);
    REQUIRE_THROWS_AS(first.IntersectAsync(second, stream), std::invalid_argument);
}
