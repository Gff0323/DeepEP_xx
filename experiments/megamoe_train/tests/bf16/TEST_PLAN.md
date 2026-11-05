# MegaMoE BF16 典型尺寸测试方案

更新日期：2026-11-05。当前 [cases.json](cases.json) 仅发布 12 条典型模型尺寸用例：
Flash、Pro、GLM 各取每 rank M=512、2048、4096、8192。
其他通用边界、数据分布、保存模式和历史矩阵用例已在仓库外本地归档，不属于当前发布清单。
清单与设备验收分开记录；设备精度、性能结果对应具体源码提交与构建产物。

## 1. 范围与模型参数

范围为 Ascend950、CANN 9.2、BF16 训练公共接口的前向、四项反向梯度及独立计时。
模型参数取自用户提供的 Flash、Pro、GLM 配置；使用合成输入和缩减专家数，
不加载真实 checkpoint，不复现模型 router、共享专家或量化计算。

| 用例前缀 | hidden H | SwiGLU F / GMM1 N | TopK | 每 rank 专家数 | clipping | 路由权重缩放 | 条数 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `flash_` | 5120 | 2304 / 4608 | 6 | 4 | 10 | 1.5 | 4 |
| `pro_` | 7168 | 3072 / 6144 | 6 | 4 | 10 | 2.5 | 4 |
| `glm_` | 6144 | 2048 / 4096 | 8 | 8 | 0，关闭 | 2.5 | 4 |

| 后缀 | 每 rank M | 选取理由 | Flash/Pro 每专家行数 | GLM 每专家行数 |
| --- | ---: | --- | ---: | ---: |
| `m512` | 512 | 最小典型 token 档位，覆盖多块计算 | 768 | 512 |
| `m2048` | 2048 | 2K token 的完整前后向验证 | 3072 | 2048 |
| `m4096` | 4096 | 4K token，增加权重梯度归约长度；保留历史失败形状 | 6144 | 4096 |
| `m8192` | 8192 | 8K token，覆盖持续多块执行与较大参考数据 | 12288 | 8192 |

前缀与后缀组合即 12 个唯一 ID，例如 `flash_m512`、`pro_m4096`、`glm_m8192`。
全部使用 balanced 路由、random 数据、`save_gmm1_output=bf16` 和 seed=20260825。
均衡路由下每专家行数为 M*TopK/E_local，每 rank 接收 M*TopK 条路由。

M 是进入算子的每 rank token 总数，不能直接等同于模型序列长度 S。
没有序列切分时 M=B*S；仅设置 EP 不会自动将 M 除以卡数。
EP 来自设备列表，本组按 EP2/4/8 检查配置合法性，设备资格需分别验证。
总专家数为 EP*E_local，不代表完整模型固定的 384/384/256 个专家。

## 2. 输入生成与反向含义

单入口是 [test_mega_moe.py](test_mega_moe.py)。每个 source rank 和输入流使用
独立、固定种子的 Torch CPU Generator；所有参考与融合实现共用同一份 BF16 输入快照。
x/dy 从 [-0.1,0.1] 均匀采样，w1/w2 从 [-1,1] 均匀采样；路由权重由
FP32 softmax(logits) 乘模型缩放系数后转换为 BF16。路由 ID 固定、合法且每 token 不重复。

本测试将固定随机 dy 作为上游梯度，执行 y.backward(dy)，数学上对应测试损失
sum(y*dy)。随机的是反向输入，dx、dtopk_weights、dw1、dw2 都由实际反向计算得到。
本组不证明真实模型 loss、router 参数梯度或训练收敛。

## 3. 参考与验收

- [cpu_reference.py](cpu_reference.py)：Torch CPU FP64 前向和 autograd 真值，不插入 BF16 中间舍入。
- [npu_baseline.py](npu_baseline.py)：`fp32-boundaries-v2` NPU BigOp 作为 ATK 误差基线，
  使用 FP32 激活、导数、dp 乘积/归约和按 TopK slot 顺序的 FP32 Y/dX 归并，最终张量写回 BF16。
  两个参考角色复用公共数值实现；BF16 保存模式下，ATK 和混合容差直接复用同一份小算子结果。
  保存模式仅决定 Gate/Up 的保存、重算或解码，不选择不同的舍入规则。
- [atk_compare.py](atk_compare.py)：使用 ATK 26.9.8/private 0.1.0 的
  cv_fused_double_benchmark，逐 rank、逐输出执行，不混合不同输出。
- [precision_metrics.py](precision_metrics.py)：精确 counts 和现有混合容差报告。

每条用例完整检查 y、dx、dtopk_weights、dw1、dw2，不抽样 token、不跳反向。
ATK 最大/平均/RMSE 误差比例门限为 5/1.5/1.5，小值域数量比例为 2。
BF16 辅助混合容差为 atol=rtol=2^-7、匹配比例>=99%，
最大绝对误差<=max(0.1,32*ULP)。当前测试要求混合容差、ATK 和 counts 均通过；
辅助参考不是独立 CPU 真值，混合容差通过不能覆盖 ATK 失败。

## 4. 执行方式

辅助参考的 CPU 回归使用 `python -m unittest discover -s experiments/megamoe_train/tests/bf16 -p test_reference_contract.py -v`。
它需要 CPU Torch，检查公共 FP32 H/dH、dp、BF16 输出边界、clipping、按 slot 归并、保存复用，
以及原有 FP64 真值和 MXFP8 诊断重放；不启动 NPU 或 ATK。

`priority`、`model`、`all` 当前均选择这 12 条，不再提供已归档的 smoke/core/extended 套件。
`--case-id` 可重复指定，仅选择清单内的 ID。套件参数由 JSON 决定，不被单例 shape 参数覆盖。
`--list-cases` 不导入 Torch/ATK，不启动设备：

```bash
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --list-cases
```

安装目标分支构建的完整 wheel，并确认同一 Python 中已安装 ATK 及内部插件。
执行者确认设备获准且空闲，再运行精度；示例中 SHMEM 端口和端口+2 都需空闲：

```bash
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --skip performance --atk-ratios 5 1.5 1.5 \
  --shmem-ip-port tcp://127.0.0.1:29862
```

`--skip precision` 仅运行有限值、计数检查和性能计时，不依赖 ATK；不加 --skip 则先精度后性能。
默认跨用例复用 worker/runtime，输入和参考每次重新计算；`--isolate-cases` 使用独立执行器。
CPU FP64 参考和完整权重梯度可能占用较多主机内存，需要先核对实际容器资源。

每次输出新目录，保留 suite_summary.json、suite.log、逐用例 summary.json 和逐 rank 指标。
ATK 失败保留 actual/benchmark/golden 张量。数值失败使整套失败；运行错误停止后续用例，
不得把 ERROR/NOT_RUN 当作 PASS。记录源码、dirty diff、wheel/SO 哈希、环境、卡号、命令和 seed。

## 5. 历史证据与未验收范围

2026-11-04 的旧快照基于 4dffcfc 加模型用例，Kernel SO SHA256 为
40925bc7a99d61f1ea5b8cbff9749c9dea7ac975a669e8b1557580eefeb8be26。
按本组 12 条筛选旧 EP8 结果：Flash 四条、Pro 四条 PASS；GLM M2048/M8192 PASS，
M4096 的 rank7 dtopk_weights 最大误差比例 6.6435790853>5，判为 FAIL；
GLM M512 尚无完整设备数值结果。该最坏点 CPU=0.0101755898675151、
融合=-3.09375、BigOp=-0.45703125。

这只是具名旧快照的证据，不是当前目标分支新二进制的资格验证。
该历史记录沿用当时的原始 BF16 BigOp。`fp32-boundaries-v2` 升级激活及归并精度后，
结果应另起报告并标注参考版本，保留旧记录；CPU FP64 真值、输入和精度阈值保持不变。
当前不声称已覆盖其他路由、数据分布、保存模式、完整专家数、多机、并发、生命周期或真实训练收敛。

2026-11-05 对 PR #18 的同输入缓存复核：GLM 四个 M 档位的 dp 在
`fp32-boundaries-v2` BigOp 下通过 ATK（4×8=32 个 rank-output），复用原有融合结果与
CPU FP64 真值，仅重算 BigOp。其余四项输出没有在这轮缓存复核中重测。另以仓库外临时
脚本完成 EP2、H7168 的 BF16/none/MXFP8 完整块、尾块及 clipping 前后向对照，18 个
rank-case 全部通过现有混合容差和精确 counts；此轮未执行 CPU FP64 或 ATK。
原始结果与临时脚本保存在仓库外，PR 说明记录各轮的版本与具体覆盖。
