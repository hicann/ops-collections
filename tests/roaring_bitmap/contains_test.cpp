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
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "roaring_bitmap.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"
#include "tests/common/roaring_bitmap_factory.h"

namespace {

enum class ContainerKind {
    Array,
    Bitset,
    Run,
};

struct TestContainer {
    uint16_t key;
    ContainerKind kind;
    std::vector<uint16_t> values;
    std::vector<std::pair<uint16_t, uint16_t>> runs;
};

void Append16(std::vector<uint8_t>& bytes, uint16_t value)
{
    bytes.push_back(static_cast<uint8_t>(value));
    bytes.push_back(static_cast<uint8_t>(value >> 8U));
}

void Append32(std::vector<uint8_t>& bytes, uint32_t value)
{
    Append16(bytes, static_cast<uint16_t>(value));
    Append16(bytes, static_cast<uint16_t>(value >> 16U));
}

void Append64(std::vector<uint8_t>& bytes, uint64_t value)
{
    Append32(bytes, static_cast<uint32_t>(value));
    Append32(bytes, static_cast<uint32_t>(value >> 32U));
}

void Store32(std::vector<uint8_t>& bytes, size_t offset, uint32_t value)
{
    bytes[offset] = static_cast<uint8_t>(value);
    bytes[offset + 1] = static_cast<uint8_t>(value >> 8U);
    bytes[offset + 2] = static_cast<uint8_t>(value >> 16U);
    bytes[offset + 3] = static_cast<uint8_t>(value >> 24U);
}

uint32_t Cardinality(TestContainer const& container)
{
    if (container.kind != ContainerKind::Run) {
        return static_cast<uint32_t>(container.values.size());
    }
    uint32_t cardinality = 0;
    for (auto const& run : container.runs) {
        cardinality += static_cast<uint32_t>(run.second) + 1U;
    }
    return cardinality;
}

void PrintProgress(char const* message)
{
    // These tests may spend several seconds in the device runtime. Keep CI logs alive
    // so a slow simulation is distinguishable from a stalled test.
    std::cout << "[RoaringBitmap] " << message << std::endl << std::flush;
}

void AppendContainer(std::vector<uint8_t>& bytes, TestContainer const& container)
{
    if (container.kind == ContainerKind::Array) {
        for (uint16_t value : container.values) {
            Append16(bytes, value);
        }
        return;
    }
    if (container.kind == ContainerKind::Bitset) {
        size_t start = bytes.size();
        bytes.resize(start + aclco::detail::kRoaringBitsetContainerBytes, 0);
        for (uint16_t value : container.values) {
            bytes[start + value / 8U] |= static_cast<uint8_t>(1U << (value % 8U));
        }
        return;
    }

    Append16(bytes, static_cast<uint16_t>(container.runs.size()));
    for (auto const& run : container.runs) {
        Append16(bytes, run.first);
        Append16(bytes, run.second);
    }
}

std::vector<uint8_t> MakeBitmap32(std::vector<TestContainer> const& containers)
{
    bool hasRun = std::any_of(containers.begin(), containers.end(),
                              [](TestContainer const& c) { return c.kind == ContainerKind::Run; });
    std::vector<uint8_t> bytes;
    if (hasRun) {
        REQUIRE_FALSE(containers.empty());
        Append32(bytes, aclco::detail::kRoaringSerialCookie | ((static_cast<uint32_t>(containers.size()) - 1U) << 16U));
        size_t runBitmapBytes = (containers.size() + 7U) / 8U;
        bytes.resize(bytes.size() + runBitmapBytes, 0);
        size_t runBitmapOffset = sizeof(uint32_t);
        for (size_t i = 0; i < containers.size(); ++i) {
            if (containers[i].kind == ContainerKind::Run) {
                bytes[runBitmapOffset + i / 8U] |= static_cast<uint8_t>(1U << (i % 8U));
            }
        }
    } else {
        Append32(bytes, aclco::detail::kRoaringSerialCookieNoRun);
        Append32(bytes, static_cast<uint32_t>(containers.size()));
    }

    for (auto const& container : containers) {
        uint32_t cardinality = Cardinality(container);
        REQUIRE(cardinality > 0);
        Append16(bytes, container.key);
        Append16(bytes, static_cast<uint16_t>(cardinality - 1U));
    }

    bool hasOffsets = !hasRun || containers.size() >= aclco::detail::kRoaringNoOffsetThreshold;
    size_t offsetsOffset = bytes.size();
    if (hasOffsets) {
        bytes.resize(bytes.size() + containers.size() * sizeof(uint32_t), 0);
    }
    for (size_t i = 0; i < containers.size(); ++i) {
        if (hasOffsets) {
            Store32(bytes, offsetsOffset + i * sizeof(uint32_t), static_cast<uint32_t>(bytes.size()));
        }
        AppendContainer(bytes, containers[i]);
    }
    return bytes;
}

std::vector<uint8_t> MakeBitmap64(std::initializer_list<std::pair<uint32_t, std::vector<uint8_t>>> buckets)
{
    std::vector<uint8_t> bytes;
    Append64(bytes, buckets.size());
    for (auto const& bucket : buckets) {
        Append32(bytes, bucket.first);
        bytes.insert(bytes.end(), bucket.second.begin(), bucket.second.end());
    }
    return bytes;
}

template <typename Key>
std::vector<uint8_t> RunContains(std::vector<uint8_t> const& bitmap, std::vector<Key> const& keys, bool async = false)
{
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    aclco::RoaringBitmap<Key> roaring(bitmap.data(), bitmap.size(), {}, stream);
    aclco::test::DeviceBuffer<Key> deviceKeys(keys.size());
    aclco::test::DeviceBuffer<uint8_t> deviceOutput(keys.size());
    deviceKeys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    deviceOutput.MemsetZero(stream);
    if (async) {
        roaring.ContainsAsync(deviceKeys.Data(), deviceOutput.Data(), aclco::Extent<size_t>{keys.size()}, stream);
        aclco::test::Sync(stream);
    } else {
        roaring.Contains(deviceKeys.Data(), deviceOutput.Data(), aclco::Extent<size_t>{keys.size()}, stream);
    }
    return deviceOutput.CopyToHost(stream);
}

void RequireResults(std::vector<uint8_t> const& actual, std::initializer_list<bool> expected)
{
    REQUIRE(actual.size() == expected.size());
    size_t i = 0;
    for (bool value : expected) {
        CAPTURE(i);
        REQUIRE((actual[i] != 0) == value);
        ++i;
    }
}

std::vector<uint8_t> ReadBinaryFile(std::string const& path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open " + path);
    }
    auto length = file.tellg();
    if (length < 0) {
        throw std::runtime_error("cannot determine size of " + path);
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!file) {
        throw std::runtime_error("cannot read " + path);
    }
    return bytes;
}

} // namespace

TEST_CASE("roaring_bitmap parses empty and rejects invalid data", "[roaring_bitmap][format]")
{
    PrintProgress("format validation started");
    auto empty32 = MakeBitmap32({});
    auto metadata32 = aclco::detail::ParseRoaringBitmap32(empty32.data(), empty32.size());
    REQUIRE(metadata32.numKeys == 0);
    REQUIRE(metadata32.numContainers == 0);
    REQUIRE(metadata32.sizeBytes == empty32.size());

    auto empty64 = MakeBitmap64({});
    auto metadata64 = aclco::detail::ParseRoaringBitmap64(empty64.data(), empty64.size());
    REQUIRE(metadata64.metadata.numKeys == 0);
    REQUIRE(metadata64.metadata.numBuckets == 0);
    REQUIRE(metadata64.metadata.sizeBytes == empty64.size());

    std::vector<uint8_t> badCookie(8, 0);
    REQUIRE_THROWS_AS(aclco::detail::ParseRoaringBitmap32(badCookie.data(), badCookie.size()), std::invalid_argument);
    REQUIRE_THROWS_AS(aclco::detail::ParseRoaringBitmap32(empty32.data(), empty32.size() - 1), std::invalid_argument);

    auto duplicateKeys = MakeBitmap32({
        {1, ContainerKind::Array, {1}, {}},
        {1, ContainerKind::Array, {2}, {}},
    });
    REQUIRE_THROWS_AS(aclco::detail::ParseRoaringBitmap32(duplicateKeys.data(), duplicateKeys.size()),
                      std::invalid_argument);

    auto invalidOffset = MakeBitmap32({{1, ContainerKind::Array, {1}, {}}});
    Store32(invalidOffset, 3U * sizeof(uint32_t), 0);
    REQUIRE_THROWS_AS(aclco::detail::ParseRoaringBitmap32(invalidOffset.data(), invalidOffset.size()),
                      std::invalid_argument);

    auto truncatedBitset = MakeBitmap32({{1, ContainerKind::Bitset, std::vector<uint16_t>(4097, 1), {}}});
    truncatedBitset.pop_back();
    REQUIRE_THROWS_AS(aclco::detail::ParseRoaringBitmap32(truncatedBitset.data(), truncatedBitset.size()),
                      std::invalid_argument);

    auto bucket = MakeBitmap32({{0, ContainerKind::Array, {1}, {}}});
    auto unorderedBuckets = MakeBitmap64({{2, bucket}, {1, bucket}});
    REQUIRE_THROWS_AS(aclco::detail::ParseRoaringBitmap64(unorderedBuckets.data(), unorderedBuckets.size()),
                      std::invalid_argument);
}

TEST_CASE("roaring_bitmap factory emits portable serialization", "[roaring_bitmap][format]")
{
    PrintProgress("factory serialization started");
    using aclco::test::roaring_bitmap_factory::Serialize;

    auto empty32 = Serialize(std::vector<uint32_t>{});
    auto parsedEmpty32 = aclco::detail::ParseRoaringBitmap32(reinterpret_cast<uint8_t const*>(empty32.data()),
                                                             empty32.size());
    REQUIRE(parsedEmpty32.numContainers == 0);
    REQUIRE(parsedEmpty32.numKeys == 0);

    std::vector<uint32_t> values32;
    for (uint32_t value = 0; value < 4097; ++value) {
        values32.push_back(value);
    }
    values32.push_back(0x00010002U);
    auto serialized32 = Serialize(values32);
    auto parsed32 = aclco::detail::ParseRoaringBitmap32(reinterpret_cast<uint8_t const*>(serialized32.data()),
                                                        serialized32.size());
    REQUIRE(parsed32.numContainers == 2);
    REQUIRE(parsed32.numKeys == values32.size());

    std::vector<uint64_t> values64{0ULL, 1ULL, (uint64_t{1} << 32U) | 2ULL, (uint64_t{2} << 32U) | 3ULL};
    auto serialized64 = Serialize(values64);
    auto parsed64 = aclco::detail::ParseRoaringBitmap64(reinterpret_cast<uint8_t const*>(serialized64.data()),
                                                        serialized64.size());
    REQUIRE(parsed64.metadata.numBuckets == 3);
    REQUIRE(parsed64.metadata.numKeys == values64.size());
}

TEST_CASE("roaring_bitmap creates uint32 and uint64 owners", "[roaring_bitmap][create]")
{
    PrintProgress("create acceptance test started");
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    SECTION("uint32")
    {
        auto serialized = MakeBitmap32({{1, ContainerKind::Array, {2, 4, 8}, {}}});
        aclco::RoaringBitmap<uint32_t> bitmap(serialized.data(), serialized.size(), {}, stream);
        REQUIRE(bitmap.Size() == 3);
        REQUIRE_FALSE(bitmap.Empty());
        REQUIRE(bitmap.Data() != nullptr);
        REQUIRE(bitmap.SizeBytes() == serialized.size());
    }

    SECTION("uint64")
    {
        auto bucket = MakeBitmap32({{1, ContainerKind::Array, {2, 4, 8}, {}}});
        auto serialized = MakeBitmap64({{7, bucket}});
        aclco::RoaringBitmap<uint64_t> bitmap(serialized.data(), serialized.size(), {}, stream);
        REQUIRE(bitmap.Size() == 3);
        REQUIRE_FALSE(bitmap.Empty());
        REQUIRE(bitmap.Data() != nullptr);
        REQUIRE(bitmap.SizeBytes() == serialized.size());
    }
}

TEST_CASE("roaring_bitmap destroys uint32 and uint64 owners", "[roaring_bitmap][destroy]")
{
    PrintProgress("destroy acceptance test started");
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    SECTION("uint32")
    {
        auto serialized = MakeBitmap32({{1, ContainerKind::Array, {2, 4, 8}, {}}});
        std::optional<aclco::RoaringBitmap<uint32_t>> bitmap;
        bitmap.emplace(serialized.data(), serialized.size(), aclco::DefaultAllocator<uint8_t>{}, stream);
        REQUIRE(bitmap->Data() != nullptr);
        bitmap.reset();
        REQUIRE_FALSE(bitmap.has_value());
    }

    SECTION("uint64")
    {
        auto bucket = MakeBitmap32({{1, ContainerKind::Array, {2, 4, 8}, {}}});
        auto serialized = MakeBitmap64({{7, bucket}});
        std::optional<aclco::RoaringBitmap<uint64_t>> bitmap;
        bitmap.emplace(serialized.data(), serialized.size(), aclco::DefaultAllocator<uint8_t>{}, stream);
        REQUIRE(bitmap->Data() != nullptr);
        bitmap.reset();
        REQUIRE_FALSE(bitmap.has_value());
    }
}

TEST_CASE("roaring_bitmap empty bitmaps and non-aligned query counts", "[roaring_bitmap][contains]")
{
    PrintProgress("empty and non-aligned contains test started");
    auto empty32 = MakeBitmap32({});
    auto empty64 = MakeBitmap64({});
    RequireResults(RunContains<uint32_t>(empty32, {0U, 1U, 0xffffffffU}, true), {false, false, false});
    RequireResults(RunContains<uint64_t>(empty64, {0ULL, 1ULL, 0xffffffffffffffffULL}, true), {false, false, false});

    auto bitmap = MakeBitmap32({{0, ContainerKind::Array, {1}, {}}});
    std::vector<uint32_t> keys(1025);
    for (size_t i = 0; i < keys.size(); ++i) {
        keys[i] = (i % 2U == 0) ? 1U : 2U;
    }
    auto actual = RunContains<uint32_t>(bitmap, keys, true);
    REQUIRE(actual.size() == keys.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        CAPTURE(i);
        REQUIRE((actual[i] != 0) == (i % 2U == 0));
    }
}

TEST_CASE("roaring_bitmap uint32 contains all container layouts", "[roaring_bitmap][contains]")
{
    PrintProgress("uint32 container layout test started");
    std::vector<uint16_t> bitsetValues(5000);
    for (uint32_t i = 0; i < bitsetValues.size(); ++i) {
        bitsetValues[i] = static_cast<uint16_t>(100U + i);
    }
    auto bitmap = MakeBitmap32({
        {0, ContainerKind::Array, {1, 3, 7}, {}},
        {2, ContainerKind::Bitset, bitsetValues, {}},
        {5, ContainerKind::Run, {}, {{10, 4}, {100, 2}}},
        {65535, ContainerKind::Array, {0, 65535}, {}},
    });

    auto actual = RunContains<uint32_t>(bitmap, {
                                                    1,
                                                    2,
                                                    (2U << 16U) | 100U,
                                                    (2U << 16U) | 5099U,
                                                    (2U << 16U) | 5100U,
                                                    (5U << 16U) | 9U,
                                                    (5U << 16U) | 10U,
                                                    (5U << 16U) | 14U,
                                                    (5U << 16U) | 15U,
                                                    (5U << 16U) | 102U,
                                                    0xffffffffU,
                                                    0xffff0001U,
                                                });
    RequireResults(actual, {true, false, true, true, false, false, true, true, false, true, true, false});
}

TEST_CASE("roaring_bitmap run container without offsets", "[roaring_bitmap][contains]")
{
    PrintProgress("run container offset test started");
    auto bitmap = MakeBitmap32({
        {3, ContainerKind::Run, {}, {{1, 2}, {100, 0}}},
    });
    auto metadata = aclco::detail::ParseRoaringBitmap32(bitmap.data(), bitmap.size());
    REQUIRE(metadata.offsetsInSerializedData == 0);
    REQUIRE(metadata.computedOffsets[0] != 0);

    auto actual = RunContains<uint32_t>(
        bitmap, {(3U << 16U), (3U << 16U) | 1U, (3U << 16U) | 3U, (3U << 16U) | 4U, (3U << 16U) | 100U}, true);
    RequireResults(actual, {false, true, true, false, true});
}

TEST_CASE("roaring_bitmap independently dispatches 16-bit and 32-bit aligned loads", "[roaring_bitmap][contains]")
{
    PrintProgress("aligned load dispatch test started");
    std::vector<TestContainer> containers;
    for (uint16_t key = 0; key < 8; ++key) {
        containers.push_back({key, ContainerKind::Array, {static_cast<uint16_t>(key + 1U)}, {}});
    }
    containers.push_back({8, ContainerKind::Run, {}, {{10, 3}}});

    auto bitmap = MakeBitmap32(containers);
    auto metadata = aclco::detail::ParseRoaringBitmap32(bitmap.data(), bitmap.size());
    REQUIRE(metadata.keyCards % alignof(uint16_t) == 0);
    REQUIRE(metadata.containerOffsets % alignof(uint32_t) != 0);

    auto actual = RunContains<uint32_t>(
        bitmap, {1U, (7U << 16U) | 8U, (8U << 16U) | 10U, (8U << 16U) | 13U, (8U << 16U) | 14U}, true);
    RequireResults(actual, {true, true, true, true, false});
}

TEST_CASE("roaring_bitmap uint64 portable buckets", "[roaring_bitmap][contains]")
{
    PrintProgress("uint64 portable bucket test started");
    auto low = MakeBitmap32({{0, ContainerKind::Array, {1, 7, 42}, {}}});
    auto high = MakeBitmap32({{2, ContainerKind::Run, {}, {{10, 3}}}});
    auto bitmap = MakeBitmap64({{0, low}, {0x80000000U, high}});

    auto actual = RunContains<uint64_t>(bitmap, {
                                                    1ULL,
                                                    2ULL,
                                                    42ULL,
                                                    (1ULL << 32U) | 1ULL,
                                                    (0x80000000ULL << 32U) | (2ULL << 16U) | 10ULL,
                                                    (0x80000000ULL << 32U) | (2ULL << 16U) | 13ULL,
                                                    (0x80000000ULL << 32U) | (2ULL << 16U) | 14ULL,
                                                });
    RequireResults(actual, {true, false, true, false, true, true, false});
}

TEST_CASE("roaring_bitmap metadata and move ownership", "[roaring_bitmap][lifetime]")
{
    PrintProgress("metadata and move ownership test started");
    auto bitmap = MakeBitmap32({{1, ContainerKind::Array, {2, 4, 8}, {}}});
    aclco::test::AclGlobalGuard acl;
    aclco::test::AclStreamGuard streamGuard;
    auto stream = streamGuard.stream;

    aclco::RoaringBitmap<uint32_t> source(bitmap.data(), bitmap.size(), {}, stream);
    REQUIRE(source.Size() == 3);
    REQUIRE_FALSE(source.Empty());
    REQUIRE(source.SizeBytes() == bitmap.size());
    REQUIRE(source.Data() != nullptr);
    REQUIRE(source.Ref().Size() == source.Size());

    aclco::RoaringBitmap<uint32_t> moved(std::move(source));
    REQUIRE(moved.Size() == 3);
    REQUIRE(source.Data() == nullptr);

    auto replacementBitmap = MakeBitmap32({{2, ContainerKind::Array, {16}, {}}});
    aclco::RoaringBitmap<uint32_t> assigned(replacementBitmap.data(), replacementBitmap.size(), {}, stream);
    assigned = std::move(moved);
    REQUIRE(assigned.Size() == 3);
    REQUIRE(moved.Data() == nullptr);

    aclco::test::DeviceBuffer<uint32_t> deviceKey(1);
    aclco::test::DeviceBuffer<uint8_t> deviceOutput(1);
    uint32_t key = (1U << 16U) | 4U;
    deviceKey.CopyFromHostAsync(&key, 1, stream);
    REQUIRE_THROWS_AS(moved.ContainsAsync(deviceKey.Data(), deviceOutput.Data(), aclco::Extent<size_t>{1}, stream),
                      std::invalid_argument);

    assigned.Contains(deviceKey.Data(), deviceOutput.Data(), aclco::Extent<size_t>{1}, stream);
    auto output = deviceOutput.CopyToHost(stream);
    REQUIRE(output[0] != 0);

    std::vector<uint32_t> noKeys;
    assigned.Contains(nullptr, nullptr, aclco::Extent<size_t>{0}, stream);
}

TEST_CASE("roaring_bitmap accepts RoaringFormatSpec reference files", "[roaring_bitmap][integration]")
{
    PrintProgress("reference file integration test started");
    char const* dataDirectory = std::getenv("ROARING_BITMAP_TEST_DATA_DIR");
    if (dataDirectory == nullptr || dataDirectory[0] == '\0') {
        SKIP("ROARING_BITMAP_TEST_DATA_DIR is not set");
    }
    std::string root{dataDirectory};

    std::vector<uint32_t> keys32;
    for (uint32_t key = 0; key < 100000; key += 1000) {
        keys32.push_back(key);
    }
    for (uint32_t key = 100000; key < 200000; ++key) {
        keys32.push_back(3U * key);
    }
    for (uint32_t key = 700000; key < 800000; ++key) {
        keys32.push_back(key);
    }

    for (std::string const& name : {"bitmapwithoutruns.bin", "bitmapwithruns.bin"}) {
        auto bitmap = ReadBinaryFile(root + "/" + name);
        auto actual = RunContains<uint32_t>(bitmap, keys32, true);
        CAPTURE(name);
        REQUIRE(std::all_of(actual.begin(), actual.end(), [](uint8_t value) { return value != 0; }));
    }

    std::vector<uint64_t> keys64;
    for (uint64_t key = 0; key < 0x09000ULL; ++key) {
        keys64.push_back(key);
    }
    keys64.push_back(0x09000ULL);
    for (uint64_t key = 0x0a000ULL; key < 0x10000ULL; ++key) {
        keys64.push_back(key);
    }
    keys64.push_back(0x10000ULL);
    keys64.push_back(0x20000ULL);
    keys64.push_back(0x20005ULL);
    for (uint64_t key = 0; key < 0x10000ULL; key += 2ULL) {
        keys64.push_back(0x80000ULL + key);
    }
    size_t lowBucketKeyCount = keys64.size();
    keys64.reserve(lowBucketKeyCount * 2U);
    for (size_t i = 0; i < lowBucketKeyCount; ++i) {
        keys64.push_back((1ULL << 32U) | keys64[i]);
    }
    auto bitmap64 = ReadBinaryFile(root + "/portable_bitmap64.bin");
    auto actual64 = RunContains<uint64_t>(bitmap64, keys64, true);
    REQUIRE(std::all_of(actual64.begin(), actual64.end(), [](uint8_t value) { return value != 0; }));
}
