/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../performance_test_framework.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "roaring_bitmap.h"
#include "tests/common/roaring_bitmap_factory.h"

namespace aclco::test {

template <typename Key>
struct CreateContext {
    AclStreamGuard streamGuard;
    aclrtStream stream{nullptr};
    std::vector<uint8_t> bitmapBytes;
};

template <typename Key>
CreateContext<Key>& GetCreateContext()
{
    static CreateContext<Key> context;
    return context;
}

template <typename Key>
void SetupCreate(std::string caseName, std::string operation, size_t queries, int hitRate, std::string queryMode)
{
    (void)queries;
    (void)hitRate;
    (void)queryMode;
    if (operation != "create") {
        throw std::invalid_argument("operation must be create");
    }
    auto& context = GetCreateContext<Key>();
    context.stream = context.streamGuard.stream;
    context.bitmapBytes = roaring_bitmap_factory::LoadTestData(caseName);
}

template <typename Key>
TestResult TestCreate()
{
    auto& context = GetCreateContext<Key>();
    using Bitmap = aclco::RoaringBitmap<Key>;
    auto const start = std::chrono::steady_clock::now();
    Bitmap bitmap(context.bitmapBytes.data(), context.bitmapBytes.size(), typename Bitmap::AllocatorType{},
                  context.stream);
    auto const end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    return TestResult(us, us, 1);
}

REGISTER_PERFORMANCE_TEST(roaringBitmapCreateU32, (TestCreate<uint32_t>), (SetupCreate<uint32_t>), std::string,
                          std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_TEST(roaringBitmapCreateU64, (TestCreate<uint64_t>), (SetupCreate<uint64_t>), std::string,
                          std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapCreateU32, "roaring_create_u32_runs",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u32-runs", "create", 80000000ULL, 50, "hit-rate"}}),
                          std::string, std::string, size_t, int, std::string);
REGISTER_PERFORMANCE_ARGS(roaringBitmapCreateU64, "roaring_create_u64_portable",
                          (std::initializer_list<std::tuple<std::string, std::string, size_t, int, std::string>>{
                              {"u64-portable", "create", 80000000ULL, 50, "hit-rate"}}),
                          std::string, std::string, size_t, int, std::string);

} // namespace aclco::test
