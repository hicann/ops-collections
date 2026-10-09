# StaticMultiset API 和使用示例

`aclco::StaticMultiset<Key>` 是容量固定的多值集合，支持 `int32_t` 和 `int64_t`。
每次成功插入都占用一个元素，包括重复键。容量在构造后保持不变。

## 构造和生命周期

```cpp
#include "static_multiset.h"
#include <limits>

aclco::StaticMultiset<int64_t> set(
    aclco::Extent<std::size_t>(1024), std::numeric_limits<int64_t>::lowest(), stream);
// keys、counts、probes 和 matches 均由调用方在 Device 上分配。
auto failed = set.Insert(keys, aclco::Extent<std::size_t>(n), stream);
set.CountEach(keys, counts, aclco::Extent<std::size_t>(n), stream);
auto total = set.Count(keys, aclco::Extent<std::size_t>(n), stream);
auto written = set.Retrieve(keys, aclco::Extent<std::size_t>(n), probes, matches,
                            aclco::Extent<std::size_t>(outputCapacity), stream);
```

构造参数依次为容量、保留空键、ACL 流。容量 0 或分配字节数溢出时抛出
`std::invalid_argument`；ACL 分配或执行失败时抛出 `std::runtime_error`。
析构释放容器自己的 Device 存储，不释放调用方的输入、输出或流。
容器不可拷贝，可以移动；移动后的源对象只允许析构、重新赋值和查询 Capacity。

所有批量接口均在返回前完成当前流上的工作。多个流或线程对同一实例的调用必须由调用方串行化。
输入依赖另一个流时，调用方必须先建立流间依赖。析构必须在 ACL 上下文销毁前完成。

## 接口

以下 `Extent` 为 `aclco::Extent<std::size_t>`，所有数组指针均指向 Device 存储。
流参数可以省略，默认值为 `nullptr`，表示 ACL 默认流。

| 接口 | 行为 |
| --- | --- |
| `Capacity()` | 返回逻辑容量。 |
| `Size(stream)` | 返回元素总数，包含重复次数。 |
| `Clear(stream)` | 清空所有元素，容量不变。 |
| `Insert(keys, n, stream)` | 插入键并返回失败数量，重复键正常插入。 |
| `InsertIf<Stencil, Predicate>(keys, stencil, n, stream)` | 使用 `Predicate{}` 选择元素；未选中的元素不计入失败数。 |
| `Contains(keys, output, n, stream)` | 逐项写入一个字节的 0 或 1。 |
| `ContainsIf<Stencil, Predicate>(keys, stencil, output, n, stream)` | 查询满足条件的元素，其他位置写 0。 |
| `Find(keys, output, n, stream)` | 命中时写入键，未命中时写入保留空键。 |
| `FindIf<Stencil, Predicate>(keys, stencil, output, n, stream)` | 不满足条件或未命中的位置写入保留空键。 |
| `Count(keys, n, stream)` | 返回每个查询键匹配数的总和，重复查询重复计数。 |
| `CountEach(keys, output, n, stream)` | 每个查询写入 `uint64_t` 匹配数。 |
| `CountEachOuter(keys, output, n, stream)` | 每个查询写入 `max(匹配数, 1)`，输出为 `uint64_t`。 |
| `Retrieve(keys, n, probes, matches, outputCapacity, stream)` | 按查询输入顺序输出所有匹配对，返回对数。 |
| `RetrieveAll(output, outputCapacity, stream)` | 按成功插入的顺序输出全部键，返回元素数。 |

Contains、Find、CountEach 和 CountEachOuter 同时提供 `(keys, n, output, stream)` 重载。
条件接口还提供传入谓词对象的参考顺序重载，保留该对象的状态：

```cpp
set.InsertIf(keys, n, stencil, pred, stream);
set.ContainsIf(keys, n, stencil, pred, output, stream);
set.FindIf(keys, n, stencil, pred, output, stream);
```

CountEach 和 CountEachOuter 还提供 `(keys, n, equal, hash, output, stream)` 重载。
探测的等价关系必须与键相等一致，哈希必须与建表哈希一致；默认哈希可以使用
`StaticMultiset<Key>::Hasher`，分别为 I32 的 MurmurHash3 Fmix32 和 I64 的 Fmix64，种子为 0。
不兼容的探测策略不保证正确结果。

## 边界、容量与确定性

零长度输入允许空指针。非零长度 `Insert(nullptr, n)` 返回 n；其他查询的非零长度空输入或空输出
抛出 `std::invalid_argument`。条件接口的非零长度空 stencil 同样抛出异常。
查询批量大小独立于容器容量；例如容量为 128 的容器可以接受 130 个查询。
输出不得与输入数组重叠。
指针背后的实际数组长度由调用方负责，接口不能从原始指针推断分配长度。

被选中的保留空键计为插入失败。容量不足时，优先保留输入中较早的有效、被选中的元素，返回其余
有效元素和保留空键的失败总数，已经成功的插入不回滚。Retrieve 和 RetrieveAll 的显式输出容量
不足时抛出 `std::length_error`，不写输出。计数总量可能超出 uint64 范围时，保守地报溢出。

例如容器存有 `[2, 2, 3]`，查询为 `[2, 2, 4]`，CountEach 为 `[2, 2, 0]`，Count 为 4，
Retrieve 输出四对 `(2, 2)`。清空后以同一批次顺序重建，RetrieveAll 仍输出 `[2, 2, 3]`。
内部哈希槽位不是公开状态，不承诺并行重建后的物理槽位布局一致。

## 实现和内存

有效键的数值范围小于内部槽位数时，使用直接寻址的 uint64 计数表；数值范围较大时，
使用保存键和 uint64 重复次数的开放寻址表。策略由第一次有效插入的键范围决定，
不依赖固定形状、具体键值或测试模式。直接寻址模式下，后续有效键超出当前范围时，
先用已有元素重建哈希表，再执行本次插入。另一个静态数组保存成功插入的键序列。
逻辑容量 C 与内部哈希槽位数分开。槽位数 P 是不小于 C 的最小 2 的幂。键和 uint64 计数分别采用连续数组，
按实际表示分别分配；另有 73968 字节分段计数与前缀和工作空间。静态 Device 分配如下，不含调用方输入输出与 ACL 分配对齐：

| 当前表示 | 分配字节数（加固定工作空间） |
| --- | --- |
| 空容器或连续区间 | `C*sizeof(Key)`。 |
| 直接寻址计数表 | `C*sizeof(Key) + 8*P`。 |
| 一般哈希表 | `C*sizeof(Key) + (sizeof(Key)+8)*P`。 |

例如 C 为 2 亿时，区间表示的 I32/I64 分别约占 0.800 GB 和 1.600 GB；一般哈希表示
分别约占 4.021 GB 和 5.895 GB。首次一般插入或区间转换时才分配和初始化所需数组，
这些成本属于该次 Insert/InsertIf；不能仅比较 Create 时延而忽略后续成本。
Clear 同步后释放计数与哈希键数组，保留元素数组和固定工作空间。
输入不会另行复制到临时 Device 数组。
有序数组是容器持有的元素存储，用于稳定检索和固定的溢出保留规则。

`static_multiset_ref.h` 提供只读 Count、Contains 和 Find 设备引用；不支持并发读写。
一般哈希表采用有界线性探测。逻辑容量为 2 的幂且不同键填满表时，未命中查询最坏遍历全部槽位。
连续区间输入的性能结果不能推广到这一情形。

## 构建和验证

使用 CANN 9.0.0-beta.2 或以上及支持 950 的 ccec/bisheng。沿用仓库 Catch2 构建方法。
功能测试位于 `tests/static_multiset/`，性能测试位于 `tests/performance/static_multiset/`。

```bash
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name static_multiset
bash scripts/build.sh -p
```

也可以在配置好的 CMake 构建目录中定向构建目标、串行执行：

```bash
cmake --build build_cmake/ccec_build --target collection_tests_static_multiset_insert_test
ctest --test-dir build_cmake/ccec_build -R '^collection_tests_static_multiset_' -j1 --output-on-failure
```

每个性能文件生成独立的 `static_multiset_perf_*` 程序。性能采用原测试的同步接口墙钟时间，
框架报告中的 Mean CPU Time 和 Mean Device Time 来源相同，不能作为两个独立指标。
原始 14 个功能测试和 10 个性能文件保留断言与参数；公共辅助头位于 `tests/common/static_multiset_test_common.h`。
附加语义测试覆盖检索精确频数与顺序、超容量保留规则、输出容量不足时不写入、带状态谓词、
保留空键、I64 极值、uint64 总计数、移动所有权和跨计算核的稳定压缩。
