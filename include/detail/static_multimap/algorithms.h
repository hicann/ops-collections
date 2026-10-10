/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#pragma once
#include "static_multimap_ref.h"

namespace aclco::detail::multimap {
COLLECTION_SIMT_DEVICE uint64_t InclusiveWarpSum(uint64_t sum, uint32_t lane)
{
    constexpr uint32_t WARP = 32;
    for (uint32_t delta = 1; delta < WARP; delta *= 2) {
        auto previous = AscendC::Simt::WarpShflUpSync(sum, delta, WARP);
        if (lane >= delta) {
            sum += previous;
        }
    }
    return sum;
}
template <class Value>
COLLECTION_SIMT_DEVICE void SiftDown(__gm__ Value* values, std::size_t root, std::size_t n)
{
    while (root < n / 2) {
        auto child = root * 2 + 1;
        if (child + 1 < n && values[child] < values[child + 1]) {
            ++child;
        }
        if (!(values[root] < values[child])) {
            break;
        }
        Value tmp = values[root];
        values[root] = values[child];
        values[child] = tmp;
        root = child;
    }
}

template <class Value>
COLLECTION_SIMT_DEVICE void HeapSort(__gm__ Value* values, uint64_t count)
{
    for (auto root = count / 2; root > 0; --root) {
        SiftDown(values, root - 1, count);
    }
    for (auto end = count; end > 1; --end) {
        Value tmp = values[0];
        values[0] = values[end - 1];
        values[end - 1] = tmp;
        SiftDown(values, 0, end - 1);
    }
}
} // namespace aclco::detail::multimap
