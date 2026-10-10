/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#include "common/static_multimap_test_common.h"
using namespace aclco::test::static_multimap;
namespace {
struct Selected {
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const { return value != 0; }
};
template <class T>
using Device = aclco::test::DeviceBuffer<T>;
using Extent = aclco::Extent<std::size_t>;
template <class T>
void CheckLookup(aclco::StaticMultimap<T, T>& map, std::unordered_multimap<T, T> const& oracle,
                 std::vector<T> const& queries, std::vector<uint32_t> const& stencil, Device<T>& keys,
                 Device<uint32_t>& mask, Device<unsigned char>& contains, Device<T>& found, aclrtStream stream)
{
    for (bool conditional : {false, true}) {
        if (conditional) {
            map.template ContainsIf<uint32_t, Selected>(keys.Data(), mask.Data(), contains.Data(),
                                                        Extent(queries.size()), stream);
            map.template FindIf<uint32_t, Selected>(keys.Data(), mask.Data(), found.Data(), Extent(queries.size()),
                                                    stream);
        } else {
            map.Contains(keys.Data(), contains.Data(), Extent(queries.size()), stream);
            map.Find(keys.Data(), found.Data(), Extent(queries.size()), stream);
        }
        auto actualContains = contains.CopyToHost(stream);
        auto actualFound = found.CopyToHost(stream);
        std::vector<unsigned char> expectedContains(queries.size());
        std::vector<T> expectedFound(queries.size(), EmptyValue<T>());
        for (std::size_t i = 0; i < queries.size(); ++i) {
            if (conditional && !stencil[i]) {
                continue;
            }
            auto range = oracle.equal_range(queries[i]);
            bool any = false;
            for (auto it = range.first; it != range.second; ++it) {
                if (!any || it->second < expectedFound[i]) {
                    expectedFound[i] = it->second;
                }
                any = true;
            }
            expectedContains[i] = any;
        }
        REQUIRE(actualContains == expectedContains);
        REQUIRE(actualFound == expectedFound);
    }
}

template <class T>
void CheckRetrieve(aclco::StaticMultimap<T, T>& map, std::unordered_multimap<T, T> const& oracle,
                   std::vector<T> const& queries, Device<T>& keys, aclrtStream stream)
{
    auto total = ExpectedCount(oracle, queries);
    Device<T> outKeys(total), outValues(total);
    REQUIRE(map.Retrieve(keys.Data(), Extent(queries.size()), outKeys.Data(), outValues.Data(), Extent(total),
                         stream) == total);
    std::vector<T> expectedKeys, expectedValues;
    for (T query : queries) {
        std::vector<T> values;
        auto range = oracle.equal_range(query);
        for (auto it = range.first; it != range.second; ++it) {
            values.push_back(it->second);
        }
        std::sort(values.begin(), values.end());
        expectedKeys.insert(expectedKeys.end(), values.size(), query);
        expectedValues.insert(expectedValues.end(), values.begin(), values.end());
    }
    REQUIRE(outKeys.CopyToHost(stream) == expectedKeys);
    REQUIRE(outValues.CopyToHost(stream) == expectedValues);
}

template <class T>
void CheckExportAndConditionalInsert(aclco::StaticMultimap<T, T>& map, std::vector<aclco::Pair<T, T>> const& pairs,
                                     Device<aclco::Pair<T, T>>& input, aclrtStream stream)
{
    Device<T> allKeys(pairs.size()), allValues(pairs.size());
    REQUIRE(map.RetrieveAll(allKeys.Data(), allValues.Data(), Extent(pairs.size()), stream) == pairs.size());
    std::vector<T> expectedKeys, expectedValues;
    for (auto pair : pairs) {
        expectedKeys.push_back(pair.first);
        expectedValues.push_back(pair.second);
    }
    REQUIRE(allKeys.CopyToHost(stream) == expectedKeys);
    REQUIRE(allValues.CopyToHost(stream) == expectedValues);
    map.Clear(stream);
    std::vector<uint32_t> selected(pairs.size());
    std::vector<aclco::Pair<T, T>> filtered;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        selected[i] = i % 2;
        if (selected[i]) {
            filtered.push_back(pairs[i]);
        }
    }
    Device<uint32_t> selection(selected.size());
    selection.CopyFromHostAsync(selected.data(), selected.size(), stream);
    REQUIRE((map.template InsertIf<uint32_t, Selected>(input.Data(), selection.Data(), Extent(pairs.size()), stream)) ==
            0);
    REQUIRE(map.Size(stream) == filtered.size());
    REQUIRE(map.RetrieveAll(allKeys.Data(), allValues.Data(), Extent(pairs.size()), stream) == filtered.size());
    REQUIRE(NormalizePairs(allKeys.CopyToHost(stream), allValues.CopyToHost(stream), filtered.size()) ==
            NormalizePairs(filtered));
}
} // namespace

TEMPLATE_TEST_CASE("static_multimap mixed workloads against unordered_multimap", "[static_multimap][bulk_sequence]",
                   int32_t, int64_t)
{
    using T = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    for (std::size_t capacity : {128u, 1027u, 8192u, 100000u}) {
        for (std::size_t multiplicity : {1u, 2u, 4u, 8u}) {
            CAPTURE(capacity, multiplicity);
            auto map = MakeMultimap<T, T>(capacity, stream);
            auto distinct = std::max<std::size_t>(1, capacity / (2 * multiplicity));
            auto pairs = MakePairs<T, T>(distinct, multiplicity);
            std::reverse(pairs.begin(), pairs.end());
            Device<aclco::Pair<T, T>> input(pairs.size());
            input.CopyFromHostAsync(pairs.data(), pairs.size(), stream);
            REQUIRE(map.Insert(input.Data(), Extent(pairs.size()), stream) == 0);
            REQUIRE(map.Size(stream) == pairs.size());
            auto oracle = MapOracle(pairs);
            for (double rate : {0.0, 0.1, 0.5, 1.0}) {
                CAPTURE(rate);
                auto queries = MakeQueries<T>(distinct, 1027, rate);
                std::vector<uint32_t> stencil(queries.size());
                for (std::size_t i = 0; i < stencil.size(); ++i) {
                    stencil[i] = i % 3;
                }
                Device<T> keys(queries.size()), found(queries.size());
                Device<unsigned char> contains(queries.size());
                Device<uint32_t> mask(stencil.size());
                keys.CopyFromHostAsync(queries.data(), queries.size(), stream);
                mask.CopyFromHostAsync(stencil.data(), stencil.size(), stream);
                REQUIRE(map.Count(keys.Data(), Extent(queries.size()), stream) == ExpectedCount(oracle, queries));
                CheckLookup(map, oracle, queries, stencil, keys, mask, contains, found, stream);
                CheckRetrieve(map, oracle, queries, keys, stream);
            }
            CheckExportAndConditionalInsert(map, pairs, input, stream);
        }
    }
}
