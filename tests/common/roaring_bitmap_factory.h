/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#include "roaring_bitmap.h"
#include "extent.h"

#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::roaring_bitmap_factory {

template <typename T>
using RoaringBitmapT = aclco::RoaringBitmap<T>;

inline char const* TestDataFile(std::string const& caseName)
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

inline std::vector<uint8_t> LoadTestData(std::string const& caseName)
{
#ifndef ROARING_BITMAP_TEST_DATA_DIR
#error "ROARING_BITMAP_TEST_DATA_DIR must be provided by CMake"
#endif
    std::string path = std::string{ROARING_BITMAP_TEST_DATA_DIR} + "/" + TestDataFile(caseName);
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

template <typename T>
inline std::vector<T> AcceptanceKeys()
{
    if constexpr (std::is_same_v<T, uint32_t>) {
        return {(1U << 16U) | 2U, (1U << 16U) | 4U, (1U << 16U) | 8U};
    } else {
        return {(uint64_t{7} << 32U) | (uint64_t{1} << 16U) | 2U, (uint64_t{7} << 32U) | (uint64_t{1} << 16U) | 4U,
                (uint64_t{7} << 32U) | (uint64_t{1} << 16U) | 8U};
    }
}

/**
 * 生成唯一随机元素集合
 */
template <typename T>
inline std::vector<T> GenerateElements(std::size_t n, uint32_t seed = 1u)
{
    std::vector<T> out;
    out.reserve(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<std::uint64_t> dist(0, static_cast<std::uint64_t>(std::numeric_limits<T>::max()));
    std::unordered_set<T> seen;
    seen.reserve(n * 2 + 1);
    while (out.size() < n) {
        T v = static_cast<T>(dist(rng));
        if (seen.insert(v).second) {
            out.push_back(v);
        }
    }
    std::sort(out.begin(), out.end());
    return out;
}

inline void Append16(std::vector<std::byte>& bytes, uint16_t value)
{
    bytes.push_back(static_cast<std::byte>(value));
    bytes.push_back(static_cast<std::byte>(value >> 8U));
}

inline void Append32(std::vector<std::byte>& bytes, uint32_t value)
{
    Append16(bytes, static_cast<uint16_t>(value));
    Append16(bytes, static_cast<uint16_t>(value >> 16U));
}

inline void Append64(std::vector<std::byte>& bytes, uint64_t value)
{
    Append32(bytes, static_cast<uint32_t>(value));
    Append32(bytes, static_cast<uint32_t>(value >> 32U));
}

inline void Store32(std::vector<std::byte>& bytes, size_t offset, uint32_t value)
{
    bytes[offset] = static_cast<std::byte>(value);
    bytes[offset + 1] = static_cast<std::byte>(value >> 8U);
    bytes[offset + 2] = static_cast<std::byte>(value >> 16U);
    bytes[offset + 3] = static_cast<std::byte>(value >> 24U);
}

/** Serializes sorted, unique uint32 values in the Roaring portable format accepted by RoaringBitmap. */
inline std::vector<std::byte> SerializeUint32(const std::vector<std::uint32_t>& elements)
{
    struct Container {
        uint16_t key;
        std::vector<uint16_t> values;
    };

    std::vector<Container> containers;
    for (uint32_t element : elements) {
        uint16_t key = static_cast<uint16_t>(element >> 16U);
        if (containers.empty() || containers.back().key != key) {
            containers.push_back(Container{key, {}});
        }
        containers.back().values.push_back(static_cast<uint16_t>(element));
    }

    std::vector<std::byte> bytes;
    Append32(bytes, aclco::detail::kRoaringSerialCookieNoRun);
    Append32(bytes, static_cast<uint32_t>(containers.size()));
    for (auto const& container : containers) {
        Append16(bytes, container.key);
        Append16(bytes, static_cast<uint16_t>(container.values.size() - 1U));
    }

    size_t offsetsOffset = bytes.size();
    bytes.resize(offsetsOffset + containers.size() * sizeof(uint32_t));
    for (size_t i = 0; i < containers.size(); ++i) {
        auto const& values = containers[i].values;
        Store32(bytes, offsetsOffset + i * sizeof(uint32_t), static_cast<uint32_t>(bytes.size()));
        if (values.size() <= aclco::detail::kRoaringMaxArrayCardinality) {
            for (uint16_t value : values) {
                Append16(bytes, value);
            }
            continue;
        }

        size_t bitsetOffset = bytes.size();
        bytes.resize(bitsetOffset + aclco::detail::kRoaringBitsetContainerBytes, std::byte{0});
        for (uint16_t value : values) {
            bytes[bitsetOffset + value / 8U] |= static_cast<std::byte>(1U << (value % 8U));
        }
    }
    return bytes;
}

/** Serializes sorted, unique uint64 values as portable high-32-bit buckets containing portable U32 bitmaps. */
inline std::vector<std::byte> SerializeUint64(const std::vector<std::uint64_t>& elements)
{
    struct Bucket {
        uint32_t key;
        std::vector<uint32_t> values;
    };

    std::vector<Bucket> buckets;
    for (uint64_t element : elements) {
        uint32_t key = static_cast<uint32_t>(element >> 32U);
        if (buckets.empty() || buckets.back().key != key) {
            buckets.push_back(Bucket{key, {}});
        }
        buckets.back().values.push_back(static_cast<uint32_t>(element));
    }

    std::vector<std::byte> bytes;
    Append64(bytes, static_cast<uint64_t>(buckets.size()));
    for (auto const& bucket : buckets) {
        Append32(bytes, bucket.key);
        auto serialized = SerializeUint32(bucket.values);
        bytes.insert(bytes.end(), serialized.begin(), serialized.end());
    }
    return bytes;
}

/**
 * 序列化元素为 RoaringBitmap 格式字节
 */
template <typename T>
inline std::vector<std::byte> Serialize(const std::vector<T>& elements)
{
    if constexpr (std::is_same_v<T, std::uint32_t>) {
        return SerializeUint32(elements);
    } else {
        return SerializeUint64(elements);
    }
}

/**
 * 创建 RoaringBitmap
 */
template <typename T>
inline RoaringBitmapT<T> MakeRoaringBitmap(const std::vector<std::byte>& serialized, aclrtStream stream)
{
    return RoaringBitmapT<T>(serialized.data(), stream);
}

/**
 * 根据模式生成查询 key
 * - "in-set": 全部来自 elements
 * - "out-of-set": 全部不在 elements 中
 * - "mixed": 一半在集合内、一半在集合外
 * - "zero": 全 0
 * - "max": 全最大值
 */
template <typename T>
inline std::vector<T> GenerateQueryKeys(std::size_t n, const std::string& pattern, const std::vector<T>& elements,
                                        uint32_t seed = 1u)
{
    std::vector<T> out;
    out.reserve(n);
    std::unordered_set<T> elemSet(elements.begin(), elements.end());

    if (pattern == "in-set") {
        for (std::size_t i = 0; i < n && i < elements.size(); ++i) {
            out.push_back(elements[i]);
        }
        // 如果 n > elements.size()，循环使用
        while (out.size() < n) {
            for (std::size_t i = 0; out.size() < n && i < elements.size(); ++i) {
                out.push_back(elements[i]);
            }
        }
    } else if (pattern == "out-of-set") {
        std::mt19937_64 rng(seed);
        while (out.size() < n) {
            T v = static_cast<T>(rng());
            if (elemSet.find(v) == elemSet.end()) {
                out.push_back(v);
            }
        }
    } else if (pattern == "mixed") {
        std::size_t half = n / 2;
        // 前一半 from elements
        for (std::size_t i = 0; i < half && i < elements.size(); ++i) {
            out.push_back(elements[i]);
        }
        while (out.size() < half) {
            for (std::size_t i = 0; out.size() < half && i < elements.size(); ++i) {
                out.push_back(elements[i]);
            }
        }
        // 后一半 not in elements
        std::mt19937_64 rng(seed);
        while (out.size() < n) {
            T v = static_cast<T>(rng());
            if (elemSet.find(v) == elemSet.end()) {
                out.push_back(v);
            }
        }
    } else if (pattern == "zero") {
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(static_cast<T>(0));
        }
    } else if (pattern == "max") {
        T m = std::numeric_limits<T>::max();
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(m);
        }
    } else {
        // 默认：sequential
        for (std::size_t i = 0; i < n; ++i) {
            out.push_back(static_cast<T>(i));
        }
    }
    return out;
}

/**
 * 执行 Contains 查询，返回 host 侧结果向量
 */
template <typename T>
inline std::vector<unsigned char> ContainsKeys(const RoaringBitmapT<T>& bitmap, const std::vector<T>& hostKeys,
                                               aclrtStream stream)
{
    if (hostKeys.empty()) {
        return {};
    }
    aclco::test::DeviceBuffer<T> dKeys(hostKeys.size());
    dKeys.CopyFromHostAsync(hostKeys.data(), hostKeys.size(), stream);

    aclco::test::DeviceBuffer<unsigned char> dResult(hostKeys.size());
    dResult.MemsetZero(stream);

    bitmap.Contains(static_cast<void*>(dKeys.Data()), static_cast<void*>(dResult.Data()),
                    aclco::Extent<std::size_t>(hostKeys.size()), stream);
    return dResult.CopyToHost(stream);
}

/**
 * 计算精确匹配率：RoaringBitmap 为精确结构，结果必须与 ground truth 100% 一致。
 * 返回实际匹配比例（1.0 = 全部匹配）。若出现假阳性或假阴性则返回负值。
 */
template <typename T>
inline double ComputeMatchRate(const std::vector<T>& queries, const std::vector<unsigned char>& results,
                               const std::unordered_set<T>& groundTruth)
{
    std::size_t total = queries.size();
    std::size_t matched = 0;
    for (std::size_t i = 0; i < total; ++i) {
        bool expected = (groundTruth.find(queries[i]) != groundTruth.end());
        bool actual = (results[i] != 0);
        if (expected == actual) {
            ++matched;
        }
    }
    return total == 0 ? 1.0 : static_cast<double>(matched) / static_cast<double>(total);
}

} // namespace aclco::test::roaring_bitmap_factory
