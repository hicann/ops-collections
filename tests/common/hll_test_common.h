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

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "extent.h"
#include "hyperloglog.h"
#include "tests/common/acl_env.h"
#include "tests/common/device_buffer.h"

namespace aclco::test::hll_cases {

// Keep one ACL context per test executable.
inline aclco::test::AclGlobalGuard& GlobalAcl()
{
    static aclco::test::AclGlobalGuard guard;
    return guard;
}

template <typename Key>
auto MakeBySketchSize(uint32_t sketch_size_kb, aclrtStream stream)
{
    return aclco::HyperLogLog<Key>::CreateWithSketchSizeKB(sketch_size_kb, stream);
}

template <typename Key>
auto MakeByStandardDeviation(double standard_deviation, aclrtStream stream)
{
    return aclco::HyperLogLog<Key>::CreateWithStandardDeviation(standard_deviation, stream);
}

template <typename Key>
auto MakeByPrecision(uint32_t precision, aclrtStream stream)
{
    return aclco::HyperLogLog<Key>::CreateWithPrecision(precision, stream);
}

template <typename Key>
std::vector<Key> MakeKeys(std::size_t count, std::size_t distinct, uint64_t seed = 1)
{
    if (count == 0) {
        return {};
    }
    if (distinct == 0) {
        distinct = 1;
    }
    std::vector<Key> dictionary(distinct);
    std::iota(dictionary.begin(), dictionary.end(), static_cast<Key>(1));
    std::mt19937_64 rng(seed);
    std::shuffle(dictionary.begin(), dictionary.end(), rng);

    std::vector<Key> keys(count);
    for (std::size_t i = 0; i < count; ++i) {
        keys[i] = dictionary[i % distinct];
    }
    std::shuffle(keys.begin(), keys.end(), rng);
    return keys;
}

template <typename Key>
uint64_t ExactCardinality(std::vector<Key> const& keys)
{
    return static_cast<uint64_t>(std::unordered_set<Key>(keys.begin(), keys.end()).size());
}

inline uint64_t RegisterCount(uint32_t sketch_size_kb) { return static_cast<uint64_t>(sketch_size_kb) * 1024ULL; }

inline bool EstimateWithinAcceptance(uint64_t estimate, uint64_t exact, uint32_t sketch_size_kb)
{
    if (exact == 0) {
        return estimate == 0;
    }
    const double relative_error = std::abs(static_cast<double>(estimate) - static_cast<double>(exact)) /
                                  static_cast<double>(exact);
    const double limit = 3.0 * 1.04 / std::sqrt(static_cast<double>(RegisterCount(sketch_size_kb)));
    return relative_error <= limit;
}

template <typename Key>
uint64_t AddAndEstimate(aclco::HyperLogLog<Key>& hll, std::vector<Key> const& keys, aclrtStream stream)
{
    aclco::test::DeviceBuffer<Key> d_keys(keys.size());
    if (!keys.empty()) {
        d_keys.CopyFromHostAsync(keys.data(), keys.size(), stream);
    }
    hll.Add(static_cast<void*>(d_keys.Data()), aclco::Extent<std::size_t>(keys.size()), stream);
    return hll.Estimate(stream);
}

} // namespace aclco::test::hll_cases
