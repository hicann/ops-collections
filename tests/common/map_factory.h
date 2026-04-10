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

#include "static_map.h"
#include "probing_scheme.h"
#include "hash_functions.h"
#include "storage.h"
#include "extent.h"

#include "generators.h"

namespace aclco::test::map_factory
{

/**
 * Map type helper aligning with NV style template matrix:
 * - Key, Value are type params
 * - BucketSize is non-type param
 * - ProbingScheme is type param (we also support choosing via enum externally if needed)
 */
template <typename Key,
          typename Value,
          int BucketSize,
          typename ProbingScheme,
          typename KeyEqual>
using StaticMapT = aclco::StaticMap<Key,
                                Value,
                                aclco::Extent<std::size_t>,
                                KeyEqual,
                                ProbingScheme,
                                aclco::Storage<BucketSize>>;

template <typename Key, typename Value, int BucketSize, typename ProbingScheme, typename KeyEqual = aclco::EqualTo<Key>>
inline StaticMapT<Key, Value, BucketSize, ProbingScheme, KeyEqual>
MakeStaticMap(std::size_t capacity,
              Sentinels<Key, Value> const& s,
              aclrtStream /*stream*/,
              ProbingScheme const& probing = ProbingScheme{},
              KeyEqual const& predicate = {})
{
  // Note: aclcorrent StaticMap ctor doesn't accept stream; stream is passed to Insert/Erase.
  return StaticMapT<Key, Value, BucketSize, ProbingScheme, KeyEqual>{
    aclco::Extent<std::size_t>(capacity),
    s.emptyKey,
    s.emptyValue,
    predicate,
    probing,
    aclco::Storage<BucketSize>{}};
}

template <typename Key>
using DefaultHasher = aclco::murmurhash3_32<Key>;

template <typename Key>
using LinearProbing = aclco::LinearProbing<DefaultHasher<Key>>;

template <typename Key>
using DoubleHashing = aclco::DoubleHashing<DefaultHasher<Key>>;
} // namespace aclco::test