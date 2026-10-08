/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#pragma once

#include "../performance_test_framework.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "hyperloglog.h"

namespace aclco::test {
namespace hll_performance {

constexpr std::uint64_t NUM_INPUTS = 100000000ULL;
constexpr std::uint64_t MULTIPLICITY = 1ULL;

template <typename Func>
TestResult Measure(Func&& func)
{
    auto const start = std::chrono::high_resolution_clock::now();
    func();
    auto const stop = std::chrono::high_resolution_clock::now();
    auto const cpu = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count();
    return TestResult(static_cast<double>(cpu), static_cast<double>(cpu), 0);
}

template <typename Key>
auto MakeHll(std::uint32_t sketchSizeKb, aclrtStream stream)
{
    return aclco::HyperLogLog<Key>::CreateWithSketchSizeKB(sketchSizeKb, stream);
}

template <typename Key>
std::vector<Key> MakeKeys(std::size_t count)
{
    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i)
        keys[i] = static_cast<Key>(i + 1);
    return keys;
}

template <typename Context>
Context& GetContext()
{
    static Context context;
    return context;
}

template <typename Key>
struct InputContext {
    AclStreamGuard streamGuard;
    std::size_t numInputs{};
    std::optional<aclco::HyperLogLog<Key>> hll;
    DeviceBuffer<Key> keys;
};

template <typename Key>
struct SketchSizeContext {
    AclStreamGuard streamGuard;
    std::uint32_t sketchSizeKb{};
};
} // namespace hll_performance
} // namespace aclco::test
