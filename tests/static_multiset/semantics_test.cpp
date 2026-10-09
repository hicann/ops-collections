/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <unordered_map>

#include "common/static_multiset_test_common.h"

using namespace aclco::test::static_multiset;
namespace {
struct KeyEqual {
    template <typename T>
    COLLECTION_SIMT_DEVICE bool operator()(T left, T right) const
    {
        return left == right;
    }
};
struct AtLeast {
    uint32_t threshold;
    COLLECTION_SIMT_DEVICE bool operator()(uint32_t value) const { return value >= threshold; }
};
template <typename Key>
void CheckCompactedQueries(aclco::StaticMultiset<Key>& set, const std::vector<Key>& keys,
                           aclco::test::DeviceBuffer<Key>& input, aclco::test::DeviceBuffer<Key>& output,
                           std::size_t capacity, aclrtStream stream)
{
    auto n = keys.size();
    set.Clear(stream);
    std::vector<Key> unique(capacity);
    std::iota(unique.begin(), unique.end(), Key{1});
    output.CopyFromHostAsync(unique.data(), capacity, stream);
    REQUIRE(set.Insert(output.Data(), capacity, stream) == 0);
    std::vector<Key> matches;
    for (Key key : keys) {
        if (key != EmptyKey<Key>()) {
            matches.push_back(key);
        }
    }
    aclco::test::DeviceBuffer<Key> probes(matches.size()), retrieved(matches.size());
    REQUIRE(set.Count(input.Data(), n, stream) == matches.size());
    REQUIRE(set.Retrieve(input.Data(), n, probes.Data(), retrieved.Data(), matches.size(), stream) == matches.size());
    RequireEqual(probes.CopyToHost(stream), matches);
    RequireEqual(retrieved.CopyToHost(stream), matches);
}

template <typename Key>
void CheckIntervalState(aclco::StaticMultiset<Key>& set, const std::vector<Key>& expected, Key base,
                        std::size_t capacity, aclrtStream stream)
{
    std::vector<Key> queries;
    for (int i = -3; i <= 12; ++i) {
        queries.push_back(base + i);
    }
    queries.push_back(EmptyKey<Key>());
    queries.push_back(base);
    aclco::test::DeviceBuffer<Key> query(queries.size()), output(capacity);
    aclco::test::DeviceBuffer<uint64_t> counts(queries.size());
    query.CopyFromHostAsync(queries.data(), queries.size(), stream);
    set.CountEach(query.Data(), counts.Data(), queries.size(), stream);
    std::vector<uint64_t> oracle;
    std::vector<Key> retrieved;
    for (Key key : queries) {
        auto count = static_cast<uint64_t>(std::count(expected.begin(), expected.end(), key));
        oracle.push_back(count);
        retrieved.insert(retrieved.end(), count, key);
    }
    RequireEqual(counts.CopyToHost(stream), oracle);
    REQUIRE(set.Count(query.Data(), queries.size(), stream) == retrieved.size());
    aclco::test::DeviceBuffer<Key> probes(retrieved.size()), matches(retrieved.size());
    REQUIRE(set.Retrieve(query.Data(), queries.size(), probes.Data(), matches.Data(), retrieved.size(), stream) ==
            retrieved.size());
    RequireEqual(probes.CopyToHost(stream), retrieved);
    RequireEqual(matches.CopyToHost(stream), retrieved);
    REQUIRE(set.RetrieveAll(output.Data(), capacity, stream) == expected.size());
    auto actual = output.CopyToHost(stream);
    actual.resize(expected.size());
    RequireEqual(actual, expected);
}

} // namespace

TEMPLATE_TEST_CASE("static_multiset preserves exact order and rejects short output", "[static_multiset][semantics]",
                   int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    std::vector<Key> keys{7, 2, 7, std::numeric_limits<Key>::max(), -4, 2, 9};
    aclco::test::DeviceBuffer<Key> input(keys.size()), output(8), probe(8), match(8);
    input.CopyFromHostAsync(keys.data(), keys.size(), stream);
    REQUIRE_THROWS_AS(MakeMultiset<Key>(std::numeric_limits<std::size_t>::max() / sizeof(uint64_t), stream),
                      std::invalid_argument);
    auto set = MakeMultiset<Key>(5, stream);
    for (int repeat = 0; repeat < 3; ++repeat) {
        set.Clear(stream);
        REQUIRE(set.Insert(input.Data(), keys.size(), stream) == 2);
        REQUIRE(set.RetrieveAll(output.Data(), 8, stream) == 5);
        auto actual = output.CopyToHost(stream);
        actual.resize(5);
        RequireEqual(actual, std::vector<Key>(keys.begin(), keys.begin() + 5));
        std::vector<Key> queries{7, 8, 7, -4};
        aclco::test::DeviceBuffer<Key> query(queries.size());
        query.CopyFromHostAsync(queries.data(), queries.size(), stream);
        REQUIRE(set.Count(query.Data(), queries.size(), stream) == 5);
        REQUIRE(set.Retrieve(query.Data(), queries.size(), probe.Data(), match.Data(), 8, stream) == 5);
        auto probes = probe.CopyToHost(stream);
        probes.resize(5);
        auto matches = match.CopyToHost(stream);
        matches.resize(5);
        RequireEqual(probes, std::vector<Key>{7, 7, 7, 7, -4});
        RequireEqual(matches, probes);
        std::vector<Key> canary(8, 42);
        output.CopyFromHostAsync(canary.data(), 8, stream);
        REQUIRE_THROWS_AS(set.RetrieveAll(output.Data(), 4, stream), std::length_error);
        RequireEqual(output.CopyToHost(stream), canary);
        probe.CopyFromHostAsync(canary.data(), 8, stream);
        match.CopyFromHostAsync(canary.data(), 8, stream);
        REQUIRE_THROWS_AS(set.Retrieve(query.Data(), queries.size(), probe.Data(), match.Data(), 4, stream),
                          std::length_error);
        RequireEqual(probe.CopyToHost(stream), canary);
        RequireEqual(match.CopyToHost(stream), canary);
    }
}

TEMPLATE_TEST_CASE("static_multiset predicate state, reserved key and move ownership", "[static_multiset][semantics]",
                   int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto stream = sg.stream;
    auto set = MakeMultiset<Key>(8, stream);
    std::vector<Key> keys{3, EmptyKey<Key>(), 3, 8, 9, -2};
    std::vector<uint32_t> stencil{3, 4, 5, 1, 0, 6};
    aclco::test::DeviceBuffer<Key> input(keys.size()), output(keys.size());
    aclco::test::DeviceBuffer<uint32_t> condition(keys.size());
    aclco::test::DeviceBuffer<uint64_t> counts(keys.size());
    input.CopyFromHostAsync(keys.data(), keys.size(), stream);
    condition.CopyFromHostAsync(stencil.data(), stencil.size(), stream);
    REQUIRE(set.InsertIf(input.Data(), keys.size(), condition.Data(), AtLeast{3}, stream) == 1);
    REQUIRE(set.Size(stream) == 3);
    set.CountEach(input.Data(), counts.Data(), keys.size(), stream);
    RequireEqual(counts.CopyToHost(stream), std::vector<uint64_t>{2, 0, 2, 0, 0, 1});
    set.FindIf(input.Data(), keys.size(), condition.Data(), AtLeast{5}, output.Data(), stream);
    RequireEqual(output.CopyToHost(stream),
                 std::vector<Key>{EmptyKey<Key>(), EmptyKey<Key>(), 3, EmptyKey<Key>(), EmptyKey<Key>(), -2});
    set.CountEach(input.Data(), keys.size(), KeyEqual{}, typename aclco::StaticMultiset<Key>::Hasher{}, counts.Data(),
                  stream);
    RequireEqual(counts.CopyToHost(stream), std::vector<uint64_t>{2, 0, 2, 0, 0, 1});
    set.CountEachOuter(input.Data(), keys.size(), KeyEqual{}, typename aclco::StaticMultiset<Key>::Hasher{},
                       counts.Data(), stream);
    RequireEqual(counts.CopyToHost(stream), std::vector<uint64_t>{2, 1, 2, 1, 1, 1});
    auto moved = std::move(set);
    REQUIRE(moved.Size(stream) == 3);
    REQUIRE_THROWS_AS(set.Clear(stream), std::logic_error);
    REQUIRE_THROWS_AS(moved.Count(nullptr, 1, stream), std::invalid_argument);
    REQUIRE(moved.Count(nullptr, 0, stream) == 0);
    // Query batches are independent of storage capacity (official outer tests use C+2).
    std::vector<Key> queries(10, Key{3});
    queries.back() = 99;
    aclco::test::DeviceBuffer<Key> many(queries.size());
    aclco::test::DeviceBuffer<uint64_t> manyCounts(queries.size());
    many.CopyFromHostAsync(queries.data(), queries.size(), stream);
    REQUIRE(moved.Count(many.Data(), queries.size(), stream) == 18);
    moved.CountEachOuter(many.Data(), manyCounts.Data(), queries.size(), stream);
    std::vector<uint64_t> expected(10, 2);
    expected.back() = 1;
    RequireEqual(manyCounts.CopyToHost(stream), expected);
}

TEMPLATE_TEST_CASE("static_multiset count exceeds uint32 and full interval lookup terminates",
                   "[static_multiset][semantics]", int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    constexpr std::size_t n = 65536;
    auto set = MakeMultiset<Key>(n, sg.stream);
    std::vector<Key> repeats(n, std::numeric_limits<Key>::max());
    aclco::test::DeviceBuffer<Key> input(n);
    input.CopyFromHostAsync(repeats.data(), n, sg.stream);
    REQUIRE(set.Insert(input.Data(), n, sg.stream) == 0);
    REQUIRE(set.Count(input.Data(), n, sg.stream) == (uint64_t{1} << 32));
    auto full = MakeMultiset<Key>(16, sg.stream);
    std::vector<Key> unique(16);
    std::iota(unique.begin(), unique.end(), Key{-8});
    aclco::test::DeviceBuffer<Key> uniqueInput(unique.size()), missInput(1);
    uniqueInput.CopyFromHostAsync(unique.data(), unique.size(), sg.stream);
    REQUIRE(full.Insert(uniqueInput.Data(), unique.size(), sg.stream) == 0);
    Key miss = 42;
    missInput.CopyFromHostAsync(&miss, 1, sg.stream);
    REQUIRE(full.Count(missInput.Data(), 1, sg.stream) == 0);
    REQUIRE(full.Insert(missInput.Data(), 1, sg.stream) == 1);
}

TEMPLATE_TEST_CASE("static_multiset full sparse hash lookup terminates and clear releases state",
                   "[static_multiset][semantics][full_hash]", int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto capacity = GENERATE(std::size_t{16}, std::size_t{1024});
    auto set = MakeMultiset<Key>(capacity, sg.stream);
    std::vector<Key> keys(capacity);
    // The range exceeds the table size and input order is descending: neither
    // direct addressing nor the interval representation can handle this input.
    for (std::size_t i = 0; i < capacity; ++i) {
        keys[i] = static_cast<Key>((capacity - i) * 4099);
    }
    std::vector<Key> queries{keys.front(), Key{-7}, keys.back(), EmptyKey<Key>()};
    aclco::test::DeviceBuffer<Key> input(capacity), query(queries.size()), output(capacity), found(queries.size());
    aclco::test::DeviceBuffer<uint64_t> counts(queries.size());
    query.CopyFromHostAsync(queries.data(), queries.size(), sg.stream);
    input.CopyFromHostAsync(keys.data(), keys.size(), sg.stream);
    // Empty queries must work before any hash or count storage is allocated.
    set.CountEach(query.Data(), counts.Data(), queries.size(), sg.stream);
    RequireEqual(counts.CopyToHost(sg.stream), std::vector<uint64_t>{0, 0, 0, 0});
    set.CountEachOuter(query.Data(), queries.size(), KeyEqual{}, typename aclco::StaticMultiset<Key>::Hasher{},
                       counts.Data(), sg.stream);
    RequireEqual(counts.CopyToHost(sg.stream), std::vector<uint64_t>{1, 1, 1, 1});
    REQUIRE(set.Insert(input.Data(), keys.size(), sg.stream) == 0);
    set.CountEach(query.Data(), counts.Data(), queries.size(), sg.stream);
    RequireEqual(counts.CopyToHost(sg.stream), std::vector<uint64_t>{1, 0, 1, 0});
    set.CountEach(query.Data(), queries.size(), KeyEqual{}, typename aclco::StaticMultiset<Key>::Hasher{},
                  counts.Data(), sg.stream);
    RequireEqual(counts.CopyToHost(sg.stream), std::vector<uint64_t>{1, 0, 1, 0});
    set.Find(query.Data(), found.Data(), queries.size(), sg.stream);
    RequireEqual(found.CopyToHost(sg.stream),
                 std::vector<Key>{keys.front(), EmptyKey<Key>(), keys.back(), EmptyKey<Key>()});
    REQUIRE(set.Count(query.Data(), queries.size(), sg.stream) == 2);
    REQUIRE(set.Insert(query.Data(), queries.size(), sg.stream) == queries.size());
    REQUIRE(set.RetrieveAll(output.Data(), capacity, sg.stream) == capacity);
    RequireEqual(output.CopyToHost(sg.stream), keys);
    auto moved = std::move(set);
    REQUIRE(moved.Count(query.Data(), queries.size(), sg.stream) == 2);
    moved.Clear(sg.stream);
    moved.Find(query.Data(), found.Data(), queries.size(), sg.stream);
    RequireEqual(found.CopyToHost(sg.stream), std::vector<Key>(queries.size(), EmptyKey<Key>()));
    REQUIRE(moved.Count(query.Data(), queries.size(), sg.stream) == 0);
    REQUIRE(moved.Insert(input.Data(), keys.size(), sg.stream) == 0);
    REQUIRE(moved.Count(query.Data(), queries.size(), sg.stream) == 2);
}

TEMPLATE_TEST_CASE("static_multiset stable compaction across blocks", "[static_multiset][semantics]", int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    constexpr std::size_t n = 131123, capacity = 60001;
    auto set = MakeMultiset<Key>(capacity, sg.stream);
    std::vector<Key> keys(n), expected;
    std::vector<uint32_t> stencil(n);
    uint64_t failures = 0;
    for (std::size_t i = 0; i < n; ++i) {
        keys[i] = i % 19 == 0 ? EmptyKey<Key>() : static_cast<Key>(i % 1009 + 1);
        stencil[i] = static_cast<uint32_t>(i % 7);
        if (stencil[i] < 2) {
            continue;
        }
        if (keys[i] == EmptyKey<Key>() || expected.size() == capacity) {
            ++failures;
        } else {
            expected.push_back(keys[i]);
        }
    }
    aclco::test::DeviceBuffer<Key> input(n), output(capacity);
    aclco::test::DeviceBuffer<uint32_t> condition(n);
    input.CopyFromHostAsync(keys.data(), n, sg.stream);
    condition.CopyFromHostAsync(stencil.data(), n, sg.stream);
    REQUIRE(set.InsertIf(input.Data(), n, condition.Data(), AtLeast{2}, sg.stream) == failures);
    REQUIRE(set.Size(sg.stream) == expected.size());
    REQUIRE(set.RetrieveAll(output.Data(), capacity, sg.stream) == expected.size());
    RequireEqual(output.CopyToHost(sg.stream), expected);
    // Raw-prefix storage during unconditional selection must be overwritten by
    // stable compaction when the input includes reserved keys and overflows.
    set.Clear(sg.stream);
    expected.clear();
    failures = 0;
    for (Key key : keys) {
        if (key == EmptyKey<Key>() || expected.size() == capacity) {
            ++failures;
        } else {
            expected.push_back(key);
        }
    }
    REQUIRE(set.Insert(input.Data(), n, sg.stream) == failures);
    REQUIRE(set.RetrieveAll(output.Data(), capacity, sg.stream) == expected.size());
    RequireEqual(output.CopyToHost(sg.stream), expected);
    CheckCompactedQueries(set, keys, input, output, capacity, sg.stream);
}

TEMPLATE_TEST_CASE("static_multiset direct addressing converts to hashing on range expansion",
                   "[static_multiset][semantics]", int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(32, sg.stream);
    std::vector<Key> first{-9, -9, -7, -2};
    std::vector<Key> second{std::numeric_limits<Key>::max(), -9, std::numeric_limits<Key>::lowest() + 1};
    std::vector<Key> expected = first;
    expected.insert(expected.end(), second.begin(), second.end());
    aclco::test::DeviceBuffer<Key> initial(first.size()), extra(second.size()), all(expected.size()),
        output(expected.size());
    aclco::test::DeviceBuffer<uint64_t> counts(expected.size());
    initial.CopyFromHostAsync(first.data(), first.size(), sg.stream);
    extra.CopyFromHostAsync(second.data(), second.size(), sg.stream);
    all.CopyFromHostAsync(expected.data(), expected.size(), sg.stream);
    REQUIRE(set.Insert(initial.Data(), first.size(), sg.stream) == 0);
    REQUIRE(set.Count(initial.Data(), first.size(), sg.stream) == 6);
    REQUIRE(set.Insert(extra.Data(), second.size(), sg.stream) == 0);
    REQUIRE(set.RetrieveAll(output.Data(), expected.size(), sg.stream) == expected.size());
    RequireEqual(output.CopyToHost(sg.stream), expected);
    set.CountEach(all.Data(), counts.Data(), expected.size(), sg.stream);
    RequireEqual(counts.CopyToHost(sg.stream), std::vector<uint64_t>{3, 3, 1, 1, 1, 3, 1});
    set.Clear(sg.stream);
    REQUIRE(set.Insert(extra.Data(), second.size(), sg.stream) == 0);
    REQUIRE(set.Count(all.Data(), expected.size(), sg.stream) == 5);
    set.Clear(sg.stream);
    // Dense ranges near either signed extreme must not overflow their indices.
    std::vector<Key> nearMax{std::numeric_limits<Key>::max() - 1, std::numeric_limits<Key>::max()};
    aclco::test::DeviceBuffer<Key> high(nearMax.size());
    high.CopyFromHostAsync(nearMax.data(), nearMax.size(), sg.stream);
    REQUIRE(set.Insert(high.Data(), nearMax.size(), sg.stream) == 0);
    REQUIRE(set.Count(high.Data(), nearMax.size(), sg.stream) == 2);
    REQUIRE(set.Count(extra.Data(), second.size(), sg.stream) == 1);
    REQUIRE(set.Insert(high.Data(), nearMax.size(), sg.stream) == 0);
    REQUIRE(set.Count(high.Data(), nearMax.size(), sg.stream) == 4);
}

TEMPLATE_TEST_CASE("static_multiset sparse unsorted keys retain exact frequencies", "[static_multiset][semantics]",
                   int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    constexpr std::size_t n = 10003;
    auto set = MakeMultiset<Key>(n * 2, sg.stream);
    std::vector<Key> keys(n);
    std::unordered_map<Key, uint64_t> oracle;
    for (std::size_t i = 0; i < n; ++i) {
        keys[i] = static_cast<Key>((static_cast<int64_t>((i * 7919) % 1234) - 617) * 100003);
        ++oracle[keys[i]];
    }
    aclco::test::DeviceBuffer<Key> input(n), output(2 * n);
    aclco::test::DeviceBuffer<uint64_t> counts(n);
    input.CopyFromHostAsync(keys.data(), n, sg.stream);
    REQUIRE(set.Insert(input.Data(), n, sg.stream) == 0);
    REQUIRE(set.Insert(input.Data(), n, sg.stream) == 0);
    set.CountEach(input.Data(), counts.Data(), n, sg.stream);
    auto actual = counts.CopyToHost(sg.stream);
    uint64_t total = 0;
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(actual[i] == oracle[keys[i]] * 2);
        total += actual[i];
    }
    REQUIRE(set.Count(input.Data(), n, sg.stream) == total);
    REQUIRE(set.RetrieveAll(output.Data(), n * 2, sg.stream) == n * 2);
    auto expected = keys;
    expected.insert(expected.end(), keys.begin(), keys.end());
    RequireEqual(output.CopyToHost(sg.stream), expected);
}

TEMPLATE_TEST_CASE("static_multiset interval transitions preserve arbitrary input semantics",
                   "[static_multiset][semantics]", int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto capacity = GENERATE(std::size_t{5}, std::size_t{10}, std::size_t{23});
    auto base = GENERATE(Key{-17}, static_cast<Key>(std::numeric_limits<Key>::max() - 12));
    auto set = MakeMultiset<Key>(capacity, sg.stream);
    std::vector<Key> interval;
    for (int i = 0; i < 8; ++i) {
        interval.push_back(base + i);
    }
    std::vector<std::vector<Key>> batches{interval,
                                          interval,
                                          {static_cast<Key>(base + 2), static_cast<Key>(base + 2),
                                           static_cast<Key>(base + 7), static_cast<Key>(base + 10)},
                                          {static_cast<Key>(base - 1), static_cast<Key>(base - 2), base},
                                          {EmptyKey<Key>(), base, static_cast<Key>(base + 3)}};
    std::vector<Key> expected;
    for (const auto& batch : batches) {
        aclco::test::DeviceBuffer<Key> input(batch.size());
        input.CopyFromHostAsync(batch.data(), batch.size(), sg.stream);
        uint64_t failures = 0;
        for (Key key : batch) {
            if (key == EmptyKey<Key>() || expected.size() == capacity) {
                ++failures;
            } else {
                expected.push_back(key);
            }
        }
        REQUIRE(set.Insert(input.Data(), batch.size(), sg.stream) == failures);
        // Moving an interval representation must preserve its complete state.
        auto moved = std::move(set);
        set = std::move(moved);
        CheckIntervalState(set, expected, base, capacity, sg.stream);
    }
    set.Clear(sg.stream);
    REQUIRE(set.Size(sg.stream) == 0);
    aclco::test::DeviceBuffer<Key> input(interval.size());
    input.CopyFromHostAsync(interval.data(), interval.size(), sg.stream);
    REQUIRE(set.Count(input.Data(), interval.size(), sg.stream) == 0);
    REQUIRE(set.Insert(input.Data(), interval.size(), sg.stream) ==
            interval.size() - std::min(capacity, interval.size()));
}

TEMPLATE_TEST_CASE("static_multiset weighted interval retrieval crosses warp boundaries",
                   "[static_multiset][semantics]", int32_t, int64_t)
{
    using Key = TestType;
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard sg;
    auto set = MakeMultiset<Key>(64, sg.stream);
    std::vector<Key> keys(11);
    std::iota(keys.begin(), keys.end(), Key{-900});
    aclco::test::DeviceBuffer<Key> input(keys.size());
    input.CopyFromHostAsync(keys.data(), keys.size(), sg.stream);
    for (int repeat = 0; repeat < 3; ++repeat) {
        REQUIRE(set.Insert(input.Data(), keys.size(), sg.stream) == 0);
    }
    constexpr std::size_t n = 131123;
    std::vector<Key> queries(n), expected;
    for (std::size_t i = 0; i < n; ++i) {
        if ((i / 64) % 7 == 0 || i % 17 == 0) {
            queries[i] = EmptyKey<Key>();
        } else if (i % 5 == 0) {
            queries[i] = 12345;
        } else {
            queries[i] = keys[(i * 7) % keys.size()];
            expected.insert(expected.end(), 3, queries[i]);
        }
    }
    aclco::test::DeviceBuffer<Key> query(n), probes(expected.size()), matches(expected.size());
    query.CopyFromHostAsync(queries.data(), queries.size(), sg.stream);
    REQUIRE(set.Count(query.Data(), n, sg.stream) == expected.size());
    REQUIRE(set.Retrieve(query.Data(), n, probes.Data(), matches.Data(), expected.size(), sg.stream) ==
            expected.size());
    RequireEqual(probes.CopyToHost(sg.stream), expected);
    RequireEqual(matches.CopyToHost(sg.stream), expected);
}
