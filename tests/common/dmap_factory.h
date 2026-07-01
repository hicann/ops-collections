/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

#include <unordered_map>
#include <vector>

#include "dynamic_map.h"
#include "extent.h"
#include "hash_functions.h"
#include "probing_scheme.h"
#include "storage.h"
#include "tests/common/device_buffer.h"
#include "tests/common/generators.h"

namespace aclco::test::dmap_factory
{
template <typename Key>
using DefaultHasher = aclco::murmurhash3_32<Key>;
template <typename Key>
using LinearProbing = aclco::LinearProbing<DefaultHasher<Key>>;
template <typename Key>
using DoubleHashing = aclco::DoubleHashing<DefaultHasher<Key>>;

template <typename Key, typename Value, int BucketSize, typename ProbingScheme, typename KeyEqual>
using DynamicMapT =
    aclco::DynamicMap<Key, Value, aclco::Extent<std::size_t>, KeyEqual, ProbingScheme, aclco::Storage<BucketSize>>;

template <typename Key, typename Value, int BucketSize, typename ProbingScheme, typename KeyEqual = aclco::EqualTo<Key>>
inline DynamicMapT<Key, Value, BucketSize, ProbingScheme, KeyEqual> MakeDynamicMap(
    std::size_t initialCapacity, Sentinels<Key, Value> const& s, aclrtStream stream,
    ProbingScheme const& probing = ProbingScheme{}, KeyEqual const& predicate = {})
{
    return DynamicMapT<Key, Value, BucketSize, ProbingScheme, KeyEqual>{aclco::Extent<std::size_t>(initialCapacity),
                                                                        s.emptyKey,
                                                                        s.emptyValue,
                                                                        s.erasedKey,
                                                                        predicate,
                                                                        probing,
                                                                        aclco::Storage<BucketSize>{},
                                                                        stream};
}

template <typename Key, typename Value, typename MapT>
inline std::vector<Value> RetrieveViaFind(MapT& map, std::vector<Key> const& keys, aclrtStream stream)
{
    const std::size_t n = keys.size();
    std::vector<Value> out(n);
    if (n == 0) return out;
    DeviceBuffer<Key> dKeys(n);
    DeviceBuffer<Value> dVals(n);
    dKeys.CopyFromHostAsync(keys.data(), n, stream);
    Sync(stream);
    map.Find(static_cast<void*>(dKeys.Data()), static_cast<void*>(dVals.Data()), aclco::Extent<std::size_t>(n), stream);
    Sync(stream);
    return dVals.CopyToHost(stream);
}

template <typename Key, typename MapT>
inline std::vector<uint8_t> RetrieveViaContains(MapT& map, std::vector<Key> const& keys, aclrtStream stream)
{
    const std::size_t n = keys.size();
    std::vector<uint8_t> out(n);
    if (n == 0) return out;
    DeviceBuffer<Key> dKeys(n);
    DeviceBuffer<uint8_t> dFlags(n);
    dKeys.CopyFromHostAsync(keys.data(), n, stream);
    Sync(stream);
    map.Contains(static_cast<void*>(dKeys.Data()), static_cast<void*>(dFlags.Data()), aclco::Extent<std::size_t>(n),
                 stream);
    Sync(stream);
    return dFlags.CopyToHost(stream);
}

}  // namespace aclco::test::dmap_factory
