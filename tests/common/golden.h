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

#include "pair.h"
#include "generators.h"

namespace aclco::test
{

/**
 * CPU golden model for insert-only semantics:
 * - Each key is inserted once; duplicates are ignored (or counted as failures by test)
 */
template <typename K, typename V>
inline std::unordered_map<K, V> GoldenInsert(std::vector<aclco::Pair<K, V>> const& pairs,
                                             Sentinels<K, V> const& s)
{
  std::unordered_map<K, V> m;
  m.reserve(pairs.size() * 2 + 1);
  for (auto const& p : pairs)
  {
    if (p.first == s.emptyKey) {
      continue;
    }
    if (s.hasErased && p.first == s.erasedKey) {
      continue;
    }
    // Insert-only: keep first occurrence
    if (m.find(p.first) == m.end()) {
      m.emplace(p.first, p.second);
    }
  }
  return m;
}

/**
 * CPU golden model for insert-only semantics:
 * - Each key is inserted once; duplicates are ignored (or counted as failures by test)
 */
template <typename K>
inline std::unordered_set<K> GoldenInsert(std::vector<K> const& keys,
                                          K const& emptyKey)
{
  std::unordered_set<K> m;
  m.reserve(keys.size() * 2 + 1);
  for (auto const& key : keys)
  {
    if (key == emptyKey) {
      continue;
    }
    // Insert-only: keep first occurrence
    if (m.find(key) == m.end()) {
      m.emplace(key);
    }
  }
  return m;
}

template <typename K, typename V>
inline std::unordered_map<K, V> GoldenAfterErase(std::unordered_map<K, V> base,
                                                 std::vector<K> const& eraseKeys)
{
  for (auto const& k : eraseKeys) {
    base.erase(k);
  }
  return base;
}

template <typename K>
inline std::unordered_set<K> GoldenAfterErase(std::unordered_set<K> base,
                                              std::vector<K> const& eraseKeys)
{
  for (auto const& k : eraseKeys) {
    base.erase(k);
  }
  return base;
}

template <typename K, typename V>
inline std::unordered_map<K, V> GoldenInsertOrAssign(std::unordered_map<K, V> base,
                                                     std::vector<aclco::Pair<K, V>> const& pairs,
                                                     Sentinels<K, V> const& s)
{
  for (auto const& p : pairs) {
    if (p.first == s.emptyKey) {
      continue;
    }
    if (s.hasErased && p.first == s.erasedKey) {
      continue;
    }
    base[p.first] = p.second;
  }
  return base;
}

} // namespace aclco::test