/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <string>

#ifndef ROARING_BITMAP_TEST_DATA_DIR
#error "ROARING_BITMAP_TEST_DATA_DIR must be provided by CMake"
#endif

TEST_CASE("RoaringBitmap generated test data path is available", "[utility][roaring_testdata]")
{
    std::filesystem::path const dataDirectory{ROARING_BITMAP_TEST_DATA_DIR};

    REQUIRE(dataDirectory.is_absolute());
    REQUIRE(std::filesystem::file_size(dataDirectory / "bitmapwithoutruns.bin") == 72616U);
    REQUIRE(std::filesystem::file_size(dataDirectory / "bitmapwithruns.bin") == 48056U);
    REQUIRE(std::filesystem::file_size(dataDirectory / "portable_bitmap64.bin") == 16506U);
}
