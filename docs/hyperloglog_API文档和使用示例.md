# HyperLogLog API 与使用示例

`aclco::HyperLogLog<Key>` 是持有 Device Sketch 的基数估算容器。`Key` 支持 `int32_t` 和 `int64_t`。

## 使用示例

```cpp
#include "hyperloglog.h"

auto hll = aclco::HyperLogLog<int64_t>::CreateWithSketchSizeKB(32, stream);
hll.Add(deviceKeys, aclco::Extent<std::size_t>(count), stream);
uint64_t cardinality = hll.Estimate(stream);
hll.Merge(other, stream);
hll.Clear(stream);
```

## 接口约束

- `CreateWithSketchSizeKB` 接受 8、16、32、64、128、256 KiB。
- `CreateWithPrecision` 接受 13–18。
- `CreateWithStandardDeviation` 选择满足 `1.04/sqrt(2^p) <= standardDeviation` 的最小支持精度；参数必须为正有限数，且要求不能超过 256 KiB Sketch 的能力。
- `Add` 接受 Device 数组首地址和元素个数。非空输入必须按 `Key` 对齐，属于当前 Device 的单个 ACL 分配块，并至少包含指定数量的有效同类型元素。
- `Merge` 要求两侧容器的 `Key`、精度和 Device 一致。来源容器保持不变。
- `Clear`、`Add`、`Merge` 和 `Estimate` 在返回前完成指定流上的操作。调用方负责建立不同流之间的依赖，并串行化同一容器上的操作。
- 容器禁止复制，支持移动。移动后的对象不再持有 Sketch。

每个寄存器占 8 bit。Add 使用 seed 为 0 的 xxhash64 选择寄存器并更新 rank；Merge 对对应寄存器取最大值；Estimate 使用 HLL++ 修正并返回 `uint64_t`。除 Sketch、直方图和固定工作空间外，实现不创建与输入数量成比例的 Device 副本。

`HyperLogLogRef<Key>` 是不拥有存储的 Device 引用。调用方必须提供按 `uint32_t` 对齐、已清零且至少包含 `2^p` 字节的 Device 存储，并保证其生命周期覆盖所有使用该引用的 kernel。

## 构建与测试

加载目标 CANN 环境后执行：

```bash
bash scripts/build.sh -b
bash scripts/build.sh -r --test-name hyperloglog
bash scripts/build.sh -p
bash scripts/build.sh -rp
```

功能测试为任务附件要求的 Create、Destroy、Clear、Add、Merge 和 Estimate 六组用例。性能测试使用任务附件的 72 个参数组合；输出单位为微秒，验收时换算为毫秒，并检查 `实测时延 <= 标杆时延 / 0.4`。
