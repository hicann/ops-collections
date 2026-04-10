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

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

static void MapInsertBenchmark(std::unordered_map<int, int>& m, int loop)
{
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < loop; ++i)
  {
    m[i] = i;
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto us =
    std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  std::cout << "insert " << loop << " elements use " << us << " us\n";
  CHECK(us >= 0);
}

TEST_CASE("CPU unordered_map insert micro-benchmark (hidden)", "[perf]")
{
  bool full = false;
  if (auto v = std::getenv("COLLECTION_PERF_FULL")) {
    if (*v) {
      full = std::atoi(v) != 0;
    }
  }

  std::unordered_map<int, int> m;
  if (!full) {
    m.reserve(200000);
    MapInsertBenchmark(m, 100000);
    return;
  }

  m.reserve(200000);
  MapInsertBenchmark(m, 100000);

  m.clear();
  m.reserve(20000000);
  MapInsertBenchmark(m, 10000000);

  m.clear();
  m.reserve(200000000);
  MapInsertBenchmark(m, 100000000);
}