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

#include <random>
#include <unordered_set>
#include <functional>

#include "pair.h"
#include "utility/is_same.h"
#include "tests/common/fp16_function.h"

namespace aclco::test
{

template <typename K>
constexpr K DefaultEmptyKey()
{
  K result{};
  if constexpr (std::is_same_v<K, fp16_t>) {
    StaticCast(result, static_cast<uint64_t>(-1));
  } else if constexpr (std::is_signed_v<K>) {
    result = static_cast<K>(-1);
  }
  else {
    result = std::numeric_limits<K>::max();
  }
  return result;
}

template <typename V>
constexpr V DefaultEmptyValue()
{
  V result{};
  if constexpr (std::is_same_v<V, fp16_t>) {
    StaticCast(result, static_cast<uint64_t>(-1));
  } else if constexpr (std::is_signed_v<V>) {
    result = static_cast<V>(-1);
  }
  else {
    result = std::numeric_limits<V>::max();
  }
  return result;
}

template <typename K, typename V>
struct Sentinels
{
  K emptyKey{DefaultEmptyKey<K>()};
  V emptyValue{DefaultEmptyValue<V>()};

  // Placeholder for future erased sentinel support (NV has erasedKey).
  // When StaticMap supports erase tombstone, wire this in.
  K erasedKey{};
  bool hasErased{false};
};

template <typename K, typename V = void>
inline auto MakeDefaultSentinels()
{
  if constexpr (!std::is_void_v<V>) {
    Sentinels<K, V> s{};
    if constexpr (std::is_same_v<K, fp16_t>) {
      StaticCast(s.erasedKey, static_cast<uint64_t>(-2));
    } else if constexpr (std::is_signed_v<K>) {
      s.erasedKey = static_cast<K>(-2);
    } else {
      s.erasedKey = s.emptyKey - 1;
    }
    s.hasErased = false;
    return s;
  } else {
    K result{};
    if constexpr (std::is_same_v<K, fp16_t>) {
      StaticCast(result, static_cast<uint64_t>(-1));
    } else if constexpr (std::is_signed_v<K>) {
      result = static_cast<K>(-1);
    } else {
      result = std::numeric_limits<K>::max();
    }
    return result;
  }
}


template<typename T>
constexpr auto MaxExactInteger() {
  if constexpr (std::is_floating_point_v<T>) {
      constexpr int mantissa_bits = std::numeric_limits<T>::digits;
      return static_cast<T>(1ULL << mantissa_bits);
  } else if constexpr (std::is_integral_v<T>) {
    return std::numeric_limits<T>::max();
  } else if constexpr (std::is_same_v<T, fp16_t>) {
    return static_cast<uint64_t>(2048);
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type");
  }
}

template <typename K>
uint64_t GetUpper(size_t n)
{
  uint64_t upper = 0;
  constexpr auto kMax = MaxExactInteger<K>();
  if (n > kMax / 2) {
      upper = kMax;
  } else {
      upper = static_cast<uint64_t>(n) * 2ULL;
  }

  if (n > upper) {
    std::cerr << "can not create a pairs, when pair number: " << n << " is bigger than the max exact integer of Key type: " << upper << std::endl;
    upper = 0;
  }

  return upper;
}

template <typename T>
using DistFunc = std::function<T()>;

// Helper function to compute zeta constant for Zipfian distribution
inline double ComputeZeta(uint64_t n, double theta) {
    double zeta = 0.0;
    for (uint64_t i = 1; i <= n; ++i) {
        zeta += 1.0 / std::pow(static_cast<double>(i), theta);
    }
    return zeta;
}

template <typename T>
inline DistFunc<T> MakeDistribution(const std::string& distName, T upper, std::mt19937& rng) {
    if (distName == "uniform") {
        if constexpr (std::is_integral_v<T>) {
            return [&, upper]() {
                std::uniform_int_distribution<T> u(1, upper);
                return u(rng);
            };
        } else if constexpr (std::is_floating_point_v<T>) {
            return [&, upper]() {
                std::uniform_real_distribution<T> u(1.0, upper);
                return u(rng);
            };
        } else if constexpr (std::is_same_v<T, fp16_t>) {
            return [&, upper]() {
                std::uniform_real_distribution<float> u(1.0f, static_cast<float>(upper));
                T result{};
                SetFp16FromFloat(result, u(rng));
                return result;
            };
        } else {
            static_assert(sizeof(T) == 0, "Unsupported type for uniform distribution");
        }
    } else if (distName == "zipfian") {
        double theta = 1.0;
        double zeta = ComputeZeta(static_cast<uint64_t>(upper), theta);
        if constexpr (std::is_same_v<T, fp16_t>) {
            return [&, upper, zeta, theta]() {
                std::uniform_real_distribution<double> u(0.0, 1.0);
                double uVal = u(rng);
                double sum = 0.0;
                for (uint64_t k = 1; k <= static_cast<uint64_t>(upper); ++k) {
                    sum += 1.0 / (std::pow(static_cast<double>(k), theta) * zeta);
                    if (sum > uVal) {
                        T result{};
                        SetFp16FromFloat(result, static_cast<float>(k));
                        return result;
                    }
                }
                T result{};
                SetFp16FromFloat(result, static_cast<float>(upper));
                return result;
            };
        } else {
            return [&, upper, zeta, theta]() {
                std::uniform_real_distribution<double> u(0.0, 1.0);
                double uVal = u(rng);
                double sum = 0.0;
                for (uint64_t k = 1; k <= static_cast<uint64_t>(upper); ++k) {
                    sum += 1.0 / (std::pow(static_cast<double>(k), theta) * zeta);
                    if (sum > uVal) {
                        return static_cast<T>(k);
                    }
                }
                return static_cast<T>(upper);
            };
        }
    } else if (distName == "normal") {
        if constexpr (std::is_same_v<T, fp16_t>) {
            return [&, upper]() {
                std::normal_distribution<double> n(static_cast<double>(upper) / 2.0, static_cast<double>(upper) / 6.0);
                double val = n(rng);
                if (val < 1.0) val = 1.0;
                if (val > static_cast<double>(upper)) val = static_cast<double>(upper);
                T result{};
                SetFp16FromFloat(result, static_cast<float>(val));
                return result;
            };
        } else {
            return [&, upper]() {
                std::normal_distribution<double> n(static_cast<double>(upper) / 2.0, static_cast<double>(upper) / 6.0);
                double val = n(rng);
                if (val < 1.0) val = 1.0;
                if (val > static_cast<double>(upper)) val = static_cast<double>(upper);
                return static_cast<T>(val);
            };
        }
    } else if (distName == "exponential") {
        double lambda = 2.0 / static_cast<double>(upper);
        if constexpr (std::is_same_v<T, fp16_t>) {
            return [&, upper, lambda]() {
                std::uniform_real_distribution<double> u(0.0, 1.0);
                while (true) {
                    double x = -std::log(u(rng)) / lambda;
                    if (x >= 1.0 && x <= static_cast<double>(upper)) {
                        T result{};
                        SetFp16FromFloat(result, static_cast<float>(x));
                        return result;
                    }
                }
            };
        } else {
            return [&, upper, lambda]() {
                std::uniform_real_distribution<double> u(0.0, 1.0);
                while (true) {
                    double x = -std::log(u(rng)) / lambda;
                    if (x >= 1.0 && x <= static_cast<double>(upper)) {
                        return static_cast<T>(x);
                    }
                }
            };
        }
    } else {
        throw std::invalid_argument("Unknown distribution: " + distName);
    }
}

/**
 * Generate key-value pairs.
 * 
 * @param seed Random seed
 * @param n Number of pairs to generate
 * @param s Sentinel values
 * @param keyDistribution Distribution type for keys ("uniform", "zipfian", "normal", "exponential")
 * @param unique If true, keys are guaranteed to be unique; if false, duplicates are allowed
 */
template <typename K, typename V = void, typename SentinelT>
inline auto MakeExamples(
    uint32_t seed,
    std::size_t n,
    SentinelT const& s,
    const std::string& keyDistribution = "uniform",
    bool unique = true)
{
    using Slot = std::conditional_t<std::is_void_v<V>, K, aclco::Pair<K, V>>;
    std::mt19937 rng(seed);
    const uint64_t upper = GetUpper<K>(n);
    if (upper == 0) {
        return std::vector<Slot>{};
    }

    auto dist = MakeDistribution<K>(keyDistribution, upper, rng);
    std::vector<Slot> out;
    out.reserve(n);

    if (unique) {
        // Generate unique keys
        std::unordered_set<K> seen;
        seen.reserve(n * 2 + 1);

        while (out.size() < n) {
          K k = dist();

          if constexpr (isSameV<SentinelT, Sentinels<K, V>>) {
            if (k == s.emptyKey) continue;
            if (s.hasErased && k == s.erasedKey) continue;
          } else {
            if (k == s) continue;
          }

          if (!seen.insert(k).second) continue;

          if constexpr (!std::is_void_v<V>) {
            V v{};
            if constexpr (std::is_same_v<V, fp16_t>) {
              v = dist();
            } else {
              v = static_cast<V>(dist());
            }
            if constexpr (isSameV<SentinelT, Sentinels<K, V>>) {
                if (v == s.emptyValue) {
                  if constexpr (std::is_same_v<V, fp16_t>) {
                    v = dist();
                  } else {
                    v = static_cast<V>(dist());
                  }
              }
            }
            out.emplace_back(k, v);
          } else {
            out.push_back(k);
          }
      }
    } else {
        // Generate pairs without uniqueness guarantee
        while (out.size() < n) {
          K k = dist();
          if constexpr (isSameV<SentinelT, Sentinels<K, V>>) {
            if (k == s.emptyKey) continue;
            if (s.hasErased && k == s.erasedKey) continue;
          } else {
            if (k == s) continue;
          }
          if constexpr (!std::is_void_v<V>) {
            V v{};
            if constexpr (std::is_same_v<V, fp16_t>) {
              v = dist();
            } else {
              v = static_cast<V>(dist());
            }
            if constexpr (isSameV<SentinelT, Sentinels<K, V>>) {
              if (v == s.emptyValue) {
                  if constexpr (std::is_same_v<V, fp16_t>) {
                    v = dist();
                  } else {
                    v = static_cast<V>(dist());
                  }
              }
            }
            out.emplace_back(k, v);
          } else {
            out.push_back(k);
          }
        }
    }
    return out;
}

/**
 * Generate pairs with duplicates: total n, but only `uniqueN` unique keys.
 * Duplicates reuse keys and vary values.
 * 
 * @param seed Random seed
 * @param n Total number of pairs to generate
 * @param uniqueN Number of unique keys
 * @param s Sentinel values
 * @param keyDistribution Distribution type for keys ("uniform", "zipfian", "normal", "exponential")
 */
template <typename K, typename V = void, typename SentinelT>
inline auto MakeExamplesWithDuplicates(
    uint32_t seed,
    std::size_t n,
    std::size_t uniqueN,
    SentinelT const& s,
    const std::string& keyDistribution = "uniform")
{
  using Slot = std::conditional_t<std::is_void_v<V>, K, aclco::Pair<K, V>>;
  if (uniqueN == 0) {
    uniqueN = 1;
  }
  if (uniqueN > n) {
    uniqueN = n;
  }

  auto base = MakeExamples<K, V>(seed, uniqueN, s, keyDistribution, true);

  std::mt19937 rng(seed ^ 0xA5A5A5A5u);

  uint64_t upper = GetUpper<K>(n);
  if (upper == 0) {
      return std::vector<Slot>{};
  }
  std::uniform_int_distribution<std::size_t> pick(0, uniqueN - 1);
  auto dist = MakeDistribution<K>(keyDistribution, upper, rng);

  std::vector<Slot> out;
  out.reserve(n);

  // First put all unique keys once
  for (auto const& p : base) {
    out.push_back(p);
  }

  // Then add duplicates
  while (out.size() < n) {
    auto const& chosen = base[pick(rng)];

    if constexpr (!std::is_void_v<V>) {
      K k = chosen.first;
      V v{};
      if constexpr (std::is_same_v<V, fp16_t>) {
        v = dist();
      } else {
        v = static_cast<V>(dist());
      }
      if constexpr (isSameV<SentinelT, Sentinels<K, V>>) {
        if (v == s.emptyValue) {
          if constexpr (std::is_same_v<V, fp16_t>) {
            v = dist();
          } else {
            v = static_cast<V>(dist());
          }
        }
      }
      out.emplace_back(k, v);      
    } else {
      out.push_back(chosen);
    }
  }
  return out;
}

//根据已有的来生成重复的
template <typename K, typename V = void, typename SentinelT>
inline auto MakeDuplicateExamples(
    std::vector<std::conditional_t<std::is_void_v<V>, K, aclco::Pair<K, V>>> const& base,
    std::size_t n,
    SentinelT const& s,
    uint32_t seed)
{
  using Slot = std::conditional_t<std::is_void_v<V>, K, aclco::Pair<K, V>>;
  std::vector<Slot> out;
  out.reserve(n);

  if (base.empty()) {
    return out;
  }

  std::mt19937 rng(seed ^ 0xA5A5A5A5u);
  std::uniform_int_distribution<std::size_t> pick(0, base.size() - 1);

  for (std::size_t i = 0; i < n; ++i) {
    auto const& chosen = base[pick(rng)];

    if constexpr (!std::is_void_v<V>) {
      V newValue{};
      do {
        if constexpr (std::is_same_v<V, fp16_t>) {
          std::uniform_real_distribution<float> uVal(1.0f, 2048.0f);
          SetFp16FromFloat(newValue, uVal(rng));
        } else {
          newValue = static_cast<V>(rng());
        }
      } while (newValue == chosen.second ||
               (isSameV<SentinelT, Sentinels<K, V>> && newValue == s.emptyValue));

      out.emplace_back(chosen.first, newValue);
    } else {
      out.push_back(chosen);
    }
  }

  return out;
}

} //namespace aclco::test