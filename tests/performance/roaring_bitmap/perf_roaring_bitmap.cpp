/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../performance_test_framework.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <acl/acl.h>

#include "roaring_bitmap.h"

namespace aclco::test {

class AclEventGuard {
public:
    explicit AclEventGuard(char const* operation) { CheckAcl(aclrtCreateEvent(&event_), operation); }

    AclEventGuard(AclEventGuard const&) = delete;
    AclEventGuard& operator=(AclEventGuard const&) = delete;

    ~AclEventGuard()
    {
        if (event_ != nullptr) {
            (void)aclrtDestroyEvent(event_);
        }
    }

    aclrtEvent Get() const noexcept { return event_; }

private:
    aclrtEvent event_{nullptr};
};

std::string DataDirectory()
{
    char const* path = std::getenv("ROARING_BITMAP_TEST_DATA_DIR");
    if (path == nullptr || path[0] == '\0') {
        throw std::runtime_error("set ROARING_BITMAP_TEST_DATA_DIR to the Roaring portable testdata directory");
    }
    return path;
}

std::string JoinPath(std::string const& directory, char const* fileName)
{
    if (!directory.empty() && directory.back() == '/') {
        return directory + fileName;
    }
    return directory + "/" + fileName;
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
    auto const byteCount = static_cast<size_t>(length);
    if (byteCount > static_cast<size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("file is too large: " + path);
    }
    std::vector<uint8_t> bytes(byteCount);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byteCount));
    if (!file) {
        throw std::runtime_error("cannot read " + path);
    }
    return bytes;
}

char const* CaseFile(std::string const& caseName)
{
    if (caseName == "u32-no-runs") {
        return "bitmapwithoutruns.bin";
    }
    if (caseName == "u32-runs") {
        return "bitmapwithruns.bin";
    }
    if (caseName == "u64-portable") {
        return "portable_bitmap64.bin";
    }
    throw std::invalid_argument("unsupported RoaringBitmap case: " + caseName);
}

template <typename Key>
std::vector<Key> PositiveKeys();

template <>
std::vector<uint32_t> PositiveKeys<uint32_t>()
{
    std::vector<uint32_t> keys;
    keys.reserve(200100);
    for (uint32_t key = 0; key < 100000; key += 1000) {
        keys.push_back(key);
    }
    for (uint32_t key = 100000; key < 200000; ++key) {
        keys.push_back(3U * key);
    }
    for (uint32_t key = 700000; key < 800000; ++key) {
        keys.push_back(key);
    }
    return keys;
}

template <>
std::vector<uint64_t> PositiveKeys<uint64_t>()
{
    std::vector<uint64_t> keys;
    for (uint64_t key = 0; key < 0x09000ULL; ++key) {
        keys.push_back(key);
    }
    keys.push_back(0x09000ULL);
    for (uint64_t key = 0x0a000ULL; key < 0x10000ULL; ++key) {
        keys.push_back(key);
    }
    keys.push_back(0x10000ULL);
    keys.push_back(0x20000ULL);
    keys.push_back(0x20005ULL);
    for (uint64_t key = 0; key < 0x10000ULL; key += 2ULL) {
        keys.push_back(0x80000ULL + key);
    }
    return keys;
}

template <typename Key>
Key MissingKey(size_t index)
{
    if constexpr (std::is_same_v<Key, uint32_t>) {
        return 0xf0000000U | static_cast<uint32_t>(index & 0x0fffffffU);
    } else {
        return 0xfffffffe00000000ULL | static_cast<uint64_t>(index & 0xffffffffULL);
    }
}

template <typename Key>
void MakeQueries(size_t count, uint32_t hitRate, std::string const& queryMode, std::vector<Key>& queries)
{
    queries.resize(count);
    if (queryMode == "cuco-unique") {
        if (count > (uint64_t{1} << 32U)) {
            throw std::invalid_argument("cuco-unique query mode supports at most 2^32 queries");
        }
        bool const isPowerOfTwo = (count & (count - 1U)) == 0;
        if (isPowerOfTwo) {
            uint64_t const mask = count - 1U;
            for (size_t i = 0; i < count; ++i) {
                uint64_t value = i;
                value ^= value >> 16U;
                value = (value * 0x7feb352dU) & mask;
                value ^= value >> 15U;
                value = (value * 0x846ca68bU) & mask;
                value ^= value >> 16U;
                queries[i] = static_cast<Key>(value & mask);
            }
        } else {
            std::iota(queries.begin(), queries.end(), Key{0});
            std::shuffle(queries.begin(), queries.end(), std::mt19937{0x5eedU});
        }
        return;
    }

    if (queryMode != "hit-rate") {
        throw std::invalid_argument("query mode must be hit-rate or cuco-unique");
    }
    if (hitRate > 100) {
        throw std::invalid_argument("hit rate must be in [0, 100]");
    }

    auto positive = PositiveKeys<Key>();
    size_t hitIndex = 0;
    size_t missIndex = 0;
    for (size_t i = 0; i < count; ++i) {
        uint64_t mixed = static_cast<uint64_t>(i) * 11400714819323198485ULL;
        bool hit = static_cast<uint32_t>(mixed >> 32U) % 100U < hitRate;
        queries[i] = hit ? positive[hitIndex++ % positive.size()] : MissingKey<Key>(missIndex++);
    }
}

template <typename Key>
struct RoaringBitmapPerfContext {
    AclStreamGuard streamGuard;
    aclrtStream stream{nullptr};
    std::string caseName;
    std::string operation;
    std::string queryMode;
    uint32_t hitRate{50};
    std::vector<uint8_t> bitmapBytes;
    std::vector<Key> hostQueries;
    DeviceBuffer<Key> deviceQueries;
    DeviceBuffer<uint8_t> deviceOutput;
    std::optional<aclco::RoaringBitmap<Key>> bitmap;
};

template <typename Key>
RoaringBitmapPerfContext<Key>& GetRoaringBitmapContext()
{
    static RoaringBitmapPerfContext<Key> context;
    return context;
}

template <typename Key>
void SetupRoaringBitmap(std::string caseName, std::string operation, size_t queries, int hitRate, std::string queryMode)
{
    auto& context = GetRoaringBitmapContext<Key>();
    context.stream = context.streamGuard.stream;
    context.caseName = std::move(caseName);
    context.operation = std::move(operation);
    context.queryMode = std::move(queryMode);
    if (queries == 0 || hitRate < 0 || hitRate > 100) {
        throw std::invalid_argument("queries must be positive and hit rate must be in [0, 100]");
    }
    if (context.operation != "contains" && context.operation != "create" && context.operation != "destroy") {
        throw std::invalid_argument("operation must be contains, create, or destroy");
    }
    context.hitRate = static_cast<uint32_t>(hitRate);
    context.bitmapBytes = ReadBinaryFile(JoinPath(DataDirectory(), CaseFile(context.caseName)));

    context.bitmap.reset();
    context.hostQueries.clear();
    context.deviceQueries = DeviceBuffer<Key>();
    context.deviceOutput = DeviceBuffer<uint8_t>();
    if (context.operation != "contains") {
        return;
    }

    MakeQueries<Key>(queries, context.hitRate, context.queryMode, context.hostQueries);
    context.deviceQueries = DeviceBuffer<Key>(context.hostQueries.size());
    context.deviceQueries.CopyFromHostAsync(context.hostQueries.data(), context.hostQueries.size(), context.stream);
    context.deviceOutput = DeviceBuffer<uint8_t>(context.hostQueries.size());
    context.deviceOutput.MemsetZero(context.stream);
    context.bitmap.emplace(context.bitmapBytes.data(), context.bitmapBytes.size(),
                           typename aclco::RoaringBitmap<Key>::AllocatorType{}, context.stream);

    for (int i = 0; i < 5; ++i) {
        context.bitmap->ContainsAsync(context.deviceQueries.Data(), context.deviceOutput.Data(),
                                      aclco::Extent<size_t>{context.hostQueries.size()}, context.stream);
    }
    Sync(context.stream);
}

template <typename Key>
TestResult TestRoaringBitmapContains()
{
    auto& context = GetRoaringBitmapContext<Key>();
    AclEventGuard start{"aclrtCreateEvent start"};
    AclEventGuard end{"aclrtCreateEvent end"};
    auto cpuStart = std::chrono::high_resolution_clock::now();
    CheckAcl(aclrtRecordEvent(start.Get(), context.stream), "aclrtRecordEvent start");
    context.bitmap->ContainsAsync(context.deviceQueries.Data(), context.deviceOutput.Data(),
                                  aclco::Extent<size_t>{context.hostQueries.size()}, context.stream);
    CheckAcl(aclrtRecordEvent(end.Get(), context.stream), "aclrtRecordEvent end");
    CheckAcl(aclrtSynchronizeEvent(end.Get()), "aclrtSynchronizeEvent end");
    auto cpuEnd = std::chrono::high_resolution_clock::now();

    float deviceMs = 0.0F;
    CheckAcl(aclrtEventElapsedTime(&deviceMs, start.Get(), end.Get()), "aclrtEventElapsedTime");
    double cpuUs = std::chrono::duration_cast<std::chrono::microseconds>(cpuEnd - cpuStart).count();
    return TestResult(cpuUs, static_cast<double>(deviceMs) * 1000.0, context.hostQueries.size());
}

template <typename Key>
TestResult TestRoaringBitmapCreate()
{
    auto& context = GetRoaringBitmapContext<Key>();
    using Bitmap = aclco::RoaringBitmap<Key>;
    auto const start = std::chrono::steady_clock::now();
    std::optional<Bitmap> bitmap;
    bitmap.emplace(context.bitmapBytes.data(), context.bitmapBytes.size(), typename Bitmap::AllocatorType{},
                   context.stream);
    auto const end = std::chrono::steady_clock::now();
    bitmap.reset();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    return TestResult(us, us, 1);
}

template <typename Key>
TestResult TestRoaringBitmapDestroy()
{
    auto& context = GetRoaringBitmapContext<Key>();
    using Bitmap = aclco::RoaringBitmap<Key>;
    std::optional<Bitmap> bitmap;
    bitmap.emplace(context.bitmapBytes.data(), context.bitmapBytes.size(), typename Bitmap::AllocatorType{},
                   context.stream);
    auto const destroyStart = std::chrono::steady_clock::now();
    bitmap.reset();
    Sync(context.stream);
    auto const end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - destroyStart).count();
    return TestResult(us, us, 1);
}

REGISTER_PERFORMANCE_TEST(roaringBitmapContainsU32, (TestRoaringBitmapContains<uint32_t>),
                          (SetupRoaringBitmap<uint32_t>), std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_TEST(roaringBitmapContainsU64, (TestRoaringBitmapContains<uint64_t>),
                          (SetupRoaringBitmap<uint64_t>), std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_TEST(roaringBitmapCreateU32, (TestRoaringBitmapCreate<uint32_t>), (SetupRoaringBitmap<uint32_t>),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_TEST(roaringBitmapCreateU64, (TestRoaringBitmapCreate<uint64_t>), (SetupRoaringBitmap<uint64_t>),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_TEST(roaringBitmapDestroyU32, (TestRoaringBitmapDestroy<uint32_t>), (SetupRoaringBitmap<uint32_t>),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_TEST(roaringBitmapDestroyU64, (TestRoaringBitmapDestroy<uint64_t>), (SetupRoaringBitmap<uint64_t>),
                          std::string, std::string, size_t, int, std::string);

REGISTER_PERFORMANCE_ARGS(roaringBitmapContainsU32, "roaring_contains_u32_runs_80m",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u32-runs", "contains", 80000000ULL, 50, "cuco-unique"}}),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapContainsU64, "roaring_contains_u64_portable_80m",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u64-portable", "contains", 80000000ULL, 50, "cuco-unique"}}),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapCreateU32, "roaring_create_u32_runs",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u32-runs", "create", 80000000ULL, 50, "hit-rate"}}),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapCreateU64, "roaring_create_u64_portable",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u64-portable", "create", 80000000ULL, 50, "hit-rate"}}),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapDestroyU32, "roaring_destroy_u32_runs",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u32-runs", "destroy", 80000000ULL, 50, "hit-rate"}}),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapDestroyU64, "roaring_destroy_u64_portable",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u64-portable", "destroy", 80000000ULL, 50, "hit-rate"}}),
                          std::string, std::string, size_t, int, std::string);

} // namespace aclco::test
