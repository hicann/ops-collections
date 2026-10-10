/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 * See LICENSE in the root of the software repository for the full text.
 */
#pragma once
#include <acl/acl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace aclco::detail::multimap {
inline void Check(aclError error, char const* operation)
{
    if (error != ACL_SUCCESS) {
        throw std::runtime_error(std::string("StaticMultimap: ") + operation + " failed (ACL " + std::to_string(error) +
                                 ")");
    }
}
inline void Sync(aclrtStream stream) { Check(aclrtSynchronizeStream(stream), "synchronize"); }
inline void RequirePointer(void const* pointer)
{
    if (!pointer) {
        throw std::invalid_argument("StaticMultimap: null nonempty buffer");
    }
}
template <class T>
inline std::size_t Bytes(std::size_t n)
{
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::overflow_error("StaticMultimap: buffer byte size overflow");
    }
    return n * sizeof(T);
}
template <class T>
class Buffer {
public:
    Buffer() = default;
    explicit Buffer(std::size_t n)
    {
        if (n) {
            Check(aclrtMalloc(reinterpret_cast<void**>(&data_), Bytes<T>(n), ACL_MEM_MALLOC_HUGE_FIRST), "allocate");
        }
    }
    ~Buffer()
    {
        if (data_) {
            (void)aclrtFree(data_);
        }
    }
    Buffer(Buffer const&) = delete;
    Buffer& operator=(Buffer const&) = delete;
    Buffer(Buffer&& other) noexcept : data_(std::exchange(other.data_, nullptr)) {}
    Buffer& operator=(Buffer&& other) noexcept
    {
        if (this != &other) {
            if (data_) {
                (void)aclrtFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
        }
        return *this;
    }
    T* Data() const { return data_; }

private:
    T* data_ = nullptr;
};
template <class T>
inline T Read(T const* pointer)
{
    T result{};
    Check(aclrtMemcpy(&result, sizeof(T), pointer, sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST), "read scalar");
    return result;
}
} // namespace aclco::detail::multimap
