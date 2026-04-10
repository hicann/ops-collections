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

#include <sstream>

#include "pair.h"

namespace aclco::test
{

template <typename K, typename V>
inline std::string DiffAsMap(std::vector<aclco::Pair<K, V>> const& observed,
                             std::unordered_map<K, V> const& golden)
{
  std::unordered_map<K, V> obs;
  obs.reserve(observed.size() * 2 + 1);
  for (auto const& p : observed) {
    // If duplicate keys appear in table dump (shouldn't), keep last.
    obs[p.first] = p.second;
  }

  std::ostringstream oss;
  bool any = false;

  // missing or mismatch
  for (auto const& [k, v] : golden) {
    auto it = obs.find(k);
    if (it == obs.end())
    {
      any = true;
      oss << "missing key=" << k << "\n";
    }
    else if (!(it->second == v))
    {
      any = true;
      oss << "value mismatch key=" << k << " golden=" << v << " observed=" << it->second << "\n";
    }
  }

  // extras
  for (auto const& [k, v] : obs) {
    if (golden.find(k) == golden.end()) {
      any = true;
      oss << "extra key=" << k << " value=" << v << "\n";
    }
  }

  if (!any) {
    return {};
  }
  return oss.str();
}

template <typename K, typename V>
inline bool EqualAsMap(std::vector<aclco::Pair<K, V>> const& observed,
                       std::unordered_map<K, V> const& golden,
                       std::string* outDiff = nullptr)
{
  auto d = DiffAsMap(observed, golden);
  if (outDiff) {
    *outDiff = d;
  }
  return d.empty();
}

template <typename K>
inline std::string DiffAsSet(std::vector<K> const& observed,
                             std::unordered_set<K> const& golden)
{
  std::unordered_set<K> obs;
  obs.reserve(observed.size() * 2 + 1);
  for (auto const& k : observed) {
    obs.insert(k);
  }

  std::ostringstream oss;
  bool any = false;

  // missing or mismatch
  for (auto const& k : golden) {
    auto it = obs.find(k);
    if (it == obs.end())
    {
      any = true;
      oss << "missing key=" << k << "\n";
    }
  }

  // extras
  for (auto const& k : obs) {
    if (golden.find(k) == golden.end()) {
      any = true;
      oss << "extra key=" << k << "\n";
    }
  }

  if (!any) {
    return {};
  }
  return oss.str();
}

template <typename K>
inline bool EqualAsSet(std::vector<K> const& observed,
                       std::unordered_set<K> const& golden,
                       std::string* outDiff = nullptr)
{
  auto d = DiffAsSet(observed, golden);
  if (outDiff) {
    *outDiff = d;
  }
  return d.empty();
}

} // namespace aclco::test