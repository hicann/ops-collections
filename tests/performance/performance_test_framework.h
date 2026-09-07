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

#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <functional>
#include <memory>
#include <initializer_list>
#include <cstddef>
#include <chrono>
#include <numeric>
#include <cmath>
#include <tuple>
#include <iomanip>
#include <optional>

#include "acl_env.h"
#include "device_buffer.h"
#include "generators.h"
#include "map_factory.h"
#include "set_factory.h"
#include "dump_table.h"
#include "golden.h"
#include "matchers.h"

namespace aclco::test {

class TestResult {
public:
    double cpuTimeUs{0.0};
    double deviceTimeUs{0.0};
    std::size_t iterations{0};

    TestResult() = default;
    TestResult(double cpu, double device, std::size_t iter) : cpuTimeUs(cpu), deviceTimeUs(device), iterations(iter) {}

    void Print() const
    {
        std::cout << "CPU: " << cpuTimeUs << " us, GPU: " << deviceTimeUs << " us, Iterations: " << iterations
                  << std::endl;
    }
};

// 测试统计结果结构体
struct TestStatistics {
    double meanCpuTime{0.0};
    double meanDeviceTime{0.0};
    double stddevDeviceTime{0.0};
    size_t iterations{0};
    std::string stopReason;
};

struct CaseBase {
    virtual ~CaseBase() = default;
    virtual void Run() = 0;
};

using PerformanceTestsFunc = std::function<TestResult()>;

template <typename... Args>
using PerformanceTestsSetupFunc = std::function<void(Args...)>;

// 注册器类
template <typename... Args>
class PerformanceTestsRegister : public CaseBase {
public:
    explicit PerformanceTestsRegister(PerformanceTestsFunc func) : func_(func), setup_func_(nullptr) {}

    void SetSetup(PerformanceTestsSetupFunc<Args...> setup) { setup_func_ = setup; }

    void RegisterArgs(const std::string& argsName, std::initializer_list<std::tuple<Args...>> args)
    {
        argsTable_[argsName] = std::vector<std::tuple<Args...>>(args);
    }

    void Run() override
    {
        for (auto& kv : argsTable_) {
            for (auto& sublist : kv.second) {
                CallFunc(kv.first, sublist, std::index_sequence_for<Args...>{});
            }
        }
    }

private:
    // 执行测试迭代并收集时间数据
    void RunTestIterations(std::vector<double>& cpuTimes, std::vector<double>& deviceTimes, std::string& stopReason)
    {
        const int max_iters = 50;
        const double minDeviceTimeS = 0.5;
        const double maxCpuTimeS = 1.0;
        double totalDeviceTimeS = 0.0;
        double totalCpuTimeS = 0.0;
        stopReason = "Max iterations reached";

        for (int iter = 0; iter < max_iters; ++iter) {
            TestResult r = func_();
            cpuTimes.push_back(r.cpuTimeUs);
            deviceTimes.push_back(r.deviceTimeUs);
            totalDeviceTimeS += r.deviceTimeUs / 1e6;
            totalCpuTimeS += r.cpuTimeUs / 1e6;

            if (ShouldStopIteration(deviceTimes, totalDeviceTimeS, totalCpuTimeS, minDeviceTimeS, maxCpuTimeS,
                                    stopReason)) {
                break;
            }
        }
    }

    // 检查是否应该停止迭代
    bool ShouldStopIteration(const std::vector<double>& deviceTimes, double totalDeviceTimeS, double totalCpuTimeS,
                             double minDeviceTimeS, double maxCpuTimeS, std::string& stopReason)
    {
        if (totalDeviceTimeS < minDeviceTimeS) {
            return false;
        }

        double mean = CalculateMean(deviceTimes);
        double stddev = CalculateStdDev(deviceTimes, mean);
        double relStd = stddev / mean;

        if (relStd < 0.005) {
            stopReason = "Stable (<0.5% RSD)";
            return true;
        }
        if (relStd < 0.05) {
            stopReason = "Noise stable (<5% RSD)";
            return true;
        }

        if (totalCpuTimeS > maxCpuTimeS) {
            stopReason = "Exceed max CPU time";
            return true;
        }
        return false;
    }

    // 计算均值
    double CalculateMean(const std::vector<double>& values) const
    {
        if (values.empty())
            return 0.0;
        return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    }

    // 计算标准差
    double CalculateStdDev(const std::vector<double>& values, double mean) const
    {
        if (values.empty())
            return 0.0;
        double sqSum = 0.0;
        for (auto t : values)
            sqSum += (t - mean) * (t - mean);
        return std::sqrt(sqSum / values.size());
    }

    // 计算测试统计数据
    TestStatistics CalculateStatistics(const std::vector<double>& cpuTimes, const std::vector<double>& deviceTimes,
                                       const std::string& stopReason)
    {
        TestStatistics stats;
        stats.meanCpuTime = CalculateMean(cpuTimes);
        stats.meanDeviceTime = CalculateMean(deviceTimes);
        stats.stddevDeviceTime = CalculateStdDev(deviceTimes, stats.meanDeviceTime);
        stats.iterations = deviceTimes.size();
        stats.stopReason = stopReason;
        return stats;
    }

    // 输出测试结果
    template <size_t... II>
    void PrintTestResult(const std::string& argsName, const std::tuple<Args...>& sublist, std::index_sequence<II...>,
                         const TestStatistics& stats)
    {
        std::cout << "========================================" << std::endl;
        std::cout << "Test Case: " << argsName << std::endl;
        std::cout << "Parameters: ";
        PrintTuple(sublist, std::index_sequence_for<Args...>{});
        std::cout << std::endl;
        std::cout << "Iterations: " << stats.iterations << " | Stop Reason: " << stats.stopReason << std::endl;
        std::cout << "Mean CPU Time: " << std::fixed << std::setprecision(3) << stats.meanCpuTime << " us" << std::endl;
        std::cout << "Mean Device Time: " << std::fixed << std::setprecision(3) << stats.meanDeviceTime << " us (±"
                  << std::fixed << std::setprecision(3) << stats.stddevDeviceTime << " us)" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    // 打印参数元组
    template <size_t... II>
    void PrintTuple(const std::tuple<Args...>& t, std::index_sequence<II...>)
    {
        std::cout << "[";
        size_t index = 0;
        ((std::cout << (index++ == 0 ? "" : ", ") << std::get<II>(t)), ...);
        std::cout << "]";
    }

    // 主测试函数
    template <size_t... II>
    void CallFunc(const std::string& argsName, const std::tuple<Args...>& sublist, std::index_sequence<II...>)
    {
        std::vector<double> cpuTimes;
        std::vector<double> deviceTimes;
        std::string stopReason;

        if (setup_func_) {
            setup_func_(std::get<II>(sublist)...);
        }

        RunTestIterations(cpuTimes, deviceTimes, stopReason);

        TestStatistics stats = CalculateStatistics(cpuTimes, deviceTimes, stopReason);

        PrintTestResult(argsName, sublist, std::index_sequence_for<Args...>{}, stats);
    }

    std::map<std::string, std::vector<std::tuple<Args...>>> argsTable_;
    PerformanceTestsFunc func_;
    PerformanceTestsSetupFunc<Args...> setup_func_;
};

class PerformanceTestsRegisterManager {
public:
    template <typename... Args>
    PerformanceTestsRegisterManager(const std::string& name, PerformanceTestsFunc func,
                                    PerformanceTestsSetupFunc<Args...> setup = nullptr)
    {
        auto registerPtr = std::make_shared<PerformanceTestsRegister<Args...>>(func);
        if (setup)
            registerPtr->SetSetup(setup);
        cases()[name] = registerPtr;
    }

    static CaseBase* GetCase(const std::string& name)
    {
        auto it = cases().find(name);
        return (it != cases().end()) ? it->second.get() : nullptr;
    }

    template <typename... Args>
    static void SetSetup(const std::string& name, PerformanceTestsSetupFunc<Args...> setup)
    {
        auto* c = GetCase(name);
        if (c)
            static_cast<PerformanceTestsRegister<Args...>*>(c)->SetSetup(setup);
    }

    static void RunAll()
    {
        std::cout << "\n========== Performance Test Suite ==========\n" << std::endl;
        for (auto& kv : cases()) {
            std::cout << "Running test suite: " << kv.first << std::endl;
            kv.second->Run();
        }
        std::cout << "\n========== All Tests Completed ==========\n" << std::endl;
    }

private:
    static std::map<std::string, std::shared_ptr<CaseBase>>& cases()
    {
        static std::map<std::string, std::shared_ptr<CaseBase>> inst;
        return inst;
    }
};

#define REGISTER_PERFORMANCE_TEST(TEST_NAME, FUNC_PTR, SETUP_FUNC, ...)     \
    static PerformanceTestsRegisterManager performanceRegister_##TEST_NAME( \
        #TEST_NAME, PerformanceTestsFunc(+FUNC_PTR), PerformanceTestsSetupFunc<__VA_ARGS__>(+SETUP_FUNC))

#define REGISTER_PERFORMANCE_ARGS(TEST_NAME, ARGS_NAME, VALUES, ...)                                     \
    static struct ArgsRegister_##TEST_NAME {                                                             \
        ArgsRegister_##TEST_NAME()                                                                       \
        {                                                                                                \
            auto* c = PerformanceTestsRegisterManager::GetCase(#TEST_NAME);                              \
            if (c)                                                                                       \
                static_cast<PerformanceTestsRegister<__VA_ARGS__>*>(c)->RegisterArgs(ARGS_NAME, VALUES); \
        }                                                                                                \
    } argsRegister_##TEST_NAME

} // namespace aclco::test
