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
#include "tests/common/device_buffer.h"
#include "utility/is_same.h"

namespace aclco::test
{
/**
 * Dump raw table slots (Pair<Key,Value>) from device and filter out sentinel keys.
 * This is Phase-1 observable when we don't yet have retrieve APIs.
 */
template <typename Key, typename Value = void, typename ContainerT, typename SentinelT>
inline auto DumpTable(ContainerT const& container,
                                  SentinelT const& sentinel,
                                  aclrtStream stream)
{
  using Slot = std::conditional_t<std::is_void_v<Value>, Key, aclco::Pair<Key, Value>>;

  std::size_t total = static_cast<std::size_t>(container.Capacity());
  std::vector<Slot> host(total);

  CheckAcl(aclrtMemcpyAsync(host.data(),
                            sizeof(Slot) * total,
                            container.Data(),
                            sizeof(Slot) * total,
                            ACL_MEMCPY_DEVICE_TO_HOST,
                            stream),
            "aclrtMemcpyAsync dump_table D2H");
  Sync(stream);

  std::vector<Slot> out;
  out.reserve(total);
  for (auto const& slot : host) {
    if constexpr (isSameV<SentinelT, Sentinels<Key, Value>>) {
      if (slot.first == sentinel.emptyKey) {
        continue;
      }
      if (sentinel.hasErased && slot.first == sentinel.erasedKey) {
        continue;
      } 
    }else {
        if (slot == sentinel) {
          continue;
        }
      }
      out.push_back(slot);
    }
    return out;
}

} // namespace aclco::test