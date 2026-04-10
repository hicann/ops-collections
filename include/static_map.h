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

#include <memory>

#include <acl/acl.h>

#include "detail/open_addressing/open_addressing_impl.h"
#include "hash_functions.h"
#include "macros.h"
#include "pair.h"
#include "extent.h"

namespace aclco {
static constexpr size_t defaultMapBucketSize = 5;

template <class Key,
          class T,
          class Extent = Extent<size_t>,
          class KeyEqual = aclco::EqualTo<Key>,
          class ProbingScheme = aclco::LinearProbing<aclco::murmurhash3_32<Key>>,
          class Storage = Storage<defaultMapBucketSize>>
class StaticMap {
  static_assert(sizeof(Key) <= 8, "Container dose not support key type size bigger than 8 bytes.");

  static_assert(sizeof(T) <= 8, "Container dose not support payload type size bigger than 8 bytes.");

  using ImplType = OpenAddressingImpl<Key,
                                      Pair<Key, T>,
                                      Extent,
                                      KeyEqual,
                                      ProbingScheme,
                                      Storage>;

 public:
  static constexpr auto bucketSize = ImplType::bucketSize;

  using SizeType = typename ImplType::SizeType;
  using KeyType = typename ImplType::KeyType;
  using ValueType = typename ImplType::ValueType;

  StaticMap(StaticMap const&) = delete;
  StaticMap& operator=(StaticMap const&) = delete;

  StaticMap(StaticMap&&) = default;
  StaticMap& operator=(StaticMap &&) = default;
  ~StaticMap() = default;

  /**
   * @brief 构造函数：创建指定容量的 static_map 容器
   * 
   * @param capacity map的容量
   * @param emptyKey 表示空键的标记值
   * @param emptyValue 表示空值的标记值
   * @param preb 键比较器，默认为 KeyEqual()
   * @param probingScheme 探测策略，默认为 ProbingScheme()
   * @param storage 存储策略，默认为 Storage<BucketSize>()
   * @param stream ACL流，默认为 nullptr
   * 
   * @note 实际容量会向上取整到BucketSize的倍数
   * 
   * @warning Key和Value类型大小不能超过8字节
   */
  constexpr StaticMap(Extent capacity,
                      Key emptyKey,
                      T emptyValue,
                      KeyEqual const& pred = {},
                      ProbingScheme const& probingScheme = {},
                      Storage storage = {},
                      aclrtStream stream = nullptr);

  /**
   * @brief 同步插入键值对到map中
   * 
   * @param values Device侧指向键值对数组的指针
   * @param valueNum 要插入的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 插入失败的键值对数量
   * 
   * @note 这是一个同步操作，会阻塞直到插入完成
   * 
   * @warning valueNum 参数必须与 values 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 values.size() 作为 valueNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see InsertAsync 用于异步插入操作
   */
  SizeType Insert(void *values, Extent valueNum, aclrtStream stream);

  /**
   * @brief 异步插入键值对到map中
   * 
   * @param values Device侧指向键值对数组的指针
   * @param valueNum 要插入的键值对数量，必须与values指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see Insert 用于同步插入操作
   */
  void InsertAsync(void *values, Extent valueNum, aclrtStream stream);

  /**
   * @brief 同步删除指定键的键值对
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要删除的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @return 删除失败的键数量（即不存在的键的数量）
   * 
   * @note 这是一个同步操作，会阻塞直到删除完成
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see EraseAsync 用于异步删除操作
   */
  SizeType Erase(void *keys, Extent keyNum, aclrtStream stream);// void* 由于无法知道用户传递的到底是什么类型，有可能用户传入与要求输入的类型不一致，后续建议封装一个自己的vector作为输入

  /**
   * @brief 异步删除指定键的键值对
   * 
   * @param keys Device侧指向键数组的指针
   * @param keyNum 要删除的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see Erase 用于同步删除操作
   */
  void EraseAsync(void *keys, Extent keyNum, aclrtStream stream);

  /**
   * @brief 同步清空 static_map 中所有的元素
   * 
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到清空完成
   * 
   * @see ClearAsync 用于异步清空操作
   */
  void Clear(aclrtStream stream);

  /**
   * @brief 异步清空 static_map 中所有的元素
   * 
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * 
   * @warning 必须确保在调用此函数后，流被正确同步，否则可能导致数据竞争
   * 
   * @see Clear 用于同步清空操作
   */
  void ClearAsync(aclrtStream stream) noexcept;

  /**
   * @brief 同步查找键对应的值
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到查找完成
   * @note 如果键不存在，返回空值（emptyValue）
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see FindAsync 用于异步查找操作
   */
  void Find(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // void* 获取用户传入的key，KeyNum表示传入要查找的key的个数

    /**
   * @brief 异步查找键对应的值
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 如果键不存在，返回空值（emptyValue）
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see Find 用于同步查找操作
   */
  void FindAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream);

    /**
   * @brief 同步检查指定键是否存在
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个同步操作，会阻塞直到检查完成
   * @note 无返回值，检查结果通过 outputValues 输出（bool类型）
   * @note 输出值为 true 表示键存在，false 表示键不存在
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see ContainsAsync 用于异步检查操作
   */
  void Contains(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // outputValues中的元素为bool类型

    /**
   * @brief 异步检查指定键是否存在
   * 
   * @param keys Device侧指向键数组的指针
   * @param outputValues Device侧指向输出值数组的指针
   * @param keyNum 要查找的键数量，必须与keys指向的数组实际大小一致
   * @param stream ACL流
   * 
   * @note 这是一个异步操作，不会阻塞调用线程
   * @note 无返回值，检查结果通过 outputValues 输出（bool类型）
   * @note 输出值为 true 表示键存在，false 表示键不存在
   * 
   * @warning keyNum 参数必须与 keys 指向的数组实际大小一致，否则可能导致越界访问或数据不完整
   * @warning 建议使用 keys.size() 作为 keyNum 参数，确保一致性
   * @warning 传入的指针中数据类型需要和map中的相对应
   * 
   * @see Contains 用于同步检查操作
   */
  void ContainsAsync(void *keys, void *outputValues, Extent keyNum, aclrtStream stream); // outputValues中的元素为bool类型

  /**
   * @brief 获取map的实际容量
   * 
   * @return 实际容量
   * 
   * @note 实际容量会向上取整到BuketSize的倍数
   */
  constexpr auto Capacity() const noexcept;

  /**
   * @brief 获取当前使用的键比较器
   * 
   * @return 当前键比较器的实例
   * 
   * @note 该方法返回构造函数中的指定的键比较器
   */
  constexpr KeyEqual GetKeyEqual() const noexcept;

  /**
   * @brief 获取Device侧map内部数据的指针
   * 
   * @return 指向Device侧map内部数据的指针
   * 
   * @note 该方法返回Device侧map内部存储的原始指针
   * 
   * @warning 返回Device侧的内存指针，不能在host侧直接访问
   */
  ValueType* Data() const;

 private:
  std::unique_ptr<ImplType> impl_;
  Key emptyKey_;
  T emptyValue_;
};
}

#include "detail/static_map/static_map.inl"