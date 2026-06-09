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


#define COLLECTION_GLOBAL __global__ [aicore]

#define COLLECTION_SIMT_DEVICE __simt_callee__ __forceinline__ [aicore]

#define COLLECTION_HOST_DEVICE __forceinline__ [host, aicore]

#define COLLECTION_AIV_GLOBAL __attribute__((aiv)) __global__ [aicore]

#define COLLECTION_SIMT_VF __simt_vf__ [aicore]
