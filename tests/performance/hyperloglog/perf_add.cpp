/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * This program is free software, you can redistribute it and/or modify it under
 * the terms and conditions of CANN Open Software License Agreement Version 2.0
 * (the "License"). Please refer to the License for details. You may not use
 * this file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON
 * AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */
#include "hyperloglog_perf_common.h"

namespace aclco::test {
namespace {
using namespace hll_performance;

template <typename Key>
using Context = InputContext<Key>;

template <typename Key>
void SetupAdd(std::string distribution, std::uint64_t numInputs, std::uint32_t sketchSizeKb, std::uint64_t multiplicity)
{
    if (distribution != "UNIFORM" || multiplicity != 1) {
        throw std::invalid_argument("Expected UNIFORM distribution and Multiplicity=1");
    }
    auto& context = GetContext<Context<Key>>();
    context.numInputs = static_cast<std::size_t>(numInputs);
    context.hll.emplace(MakeHll<Key>(sketchSizeKb, context.streamGuard.stream));
    auto keys = MakeKeys<Key>(context.numInputs);
    context.keys.Resize(context.numInputs);
    context.keys.CopyFromHostAsync(keys.data(), keys.size(), context.streamGuard.stream);
    Sync(context.streamGuard.stream);
}

template <typename Key>
TestResult TestAdd()
{
    auto& context = GetContext<Context<Key>>();
    auto result = Measure([&] {
        context.hll->Add(context.keys.Data(), aclco::Extent<std::size_t>(context.numInputs),
                         context.streamGuard.stream);
    });
    context.hll->Clear(context.streamGuard.stream);
    return result;
}
} // namespace

REGISTER_PERFORMANCE_TEST(hyperLogLogAddI32, (TestAdd<std::int32_t>), (SetupAdd<std::int32_t>), std::string,
                          std::uint64_t, std::uint32_t, std::uint64_t);
REGISTER_PERFORMANCE_TEST(hyperLogLogAddI64, (TestAdd<std::int64_t>), (SetupAdd<std::int64_t>), std::string,
                          std::uint64_t, std::uint32_t, std::uint64_t);

REGISTER_PERFORMANCE_ARGS(hyperLogLogAddI32,
                          "HyperLogLog add (Distribution, NumInputs, SketchSizeKB, Multiplicity) I32",
                          (std::initializer_list<std::tuple<std::string, std::uint64_t, std::uint32_t, std::uint64_t>>{
                              {"UNIFORM", NUM_INPUTS, 8, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 16, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 32, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 64, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 128, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 256, MULTIPLICITY}}),
                          std::string, std::uint64_t, std::uint32_t, std::uint64_t);
REGISTER_PERFORMANCE_ARGS(hyperLogLogAddI64,
                          "HyperLogLog add (Distribution, NumInputs, SketchSizeKB, Multiplicity) I64",
                          (std::initializer_list<std::tuple<std::string, std::uint64_t, std::uint32_t, std::uint64_t>>{
                              {"UNIFORM", NUM_INPUTS, 8, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 16, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 32, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 64, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 128, MULTIPLICITY},
                              {"UNIFORM", NUM_INPUTS, 256, MULTIPLICITY}}),
                          std::string, std::uint64_t, std::uint32_t, std::uint64_t);
} // namespace aclco::test
