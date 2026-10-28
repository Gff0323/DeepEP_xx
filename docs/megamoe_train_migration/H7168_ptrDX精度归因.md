# MegaMoE BF16：H7168 ptrDX 精度失败归因

日期：2026-10-28。对象：GitCode `ascend_megamoe_bf16@b8cff14854d12aef77b6ae54b5890123759b4d28`。

## 1. 结论

这 19 个 ptrDX 超差点，来源已定位到**上游 GMM1 的 Gate/Up、反向 GMM2 的 dHp 出现不同 BF16 舍入结果，差异经过 SwiGLU 梯度传播，再在 dX 点积的正负项抵消中显现**。

本次证据不支持“ptrDX 矩阵乘算错”或“SwiGLU 梯度公式写错”。golden 的梯度公式、张量布局和 BF16 张量边界也没有发现这类错误。需要区分：

- **实际数值差异存在**：kernel 和 CPU golden 的上游 BF16 张量并非逐位一致，不能把失败直接称为假阳性。
- **原 ptrDX 检查是完整链路对照**：它使用 CPU 自己算出的 gradZ，不能据此单独判定设备的 `gradZ @ W1` 有错。
- **保持原判定**：原端到端检查仍为 rank0 13 个、rank1 6 个元素超差，没有调整阈值，也没有将原用例改成 PASS。

归因已到 GMM 输出的 BF16 舍入边界；本轮未改 kernel 抓取 L0C 的 FP32 累加值，因此没有进一步断言是哪条硬件累加指令、归约顺序或 Fixpipe 转换细节造成跨边界。也不能由此宣称所有 shape 的 kernel 都正确。

## 2. 验证范围与不变项

原失败配置：EP2，M65，H7168，F4096，N8192，每卡 2 专家，TopK4，balanced，保存 bf16，clip=0，seed=20260825。每卡有效 packed 行数 260，专家计数 `[130, 130]`。

原测试节点被其他作业占用后，在另一空闲 Ascend950 节点的独立目录与容器中复现。同一 SO 仍复现相同的 13/6 个超差、相同首个失败坐标和值。

- kernel SHA256：`5054308109a2c556bcc52eb94cfecda8532e04635ee04ed0c48845b50ea46da9`
- Host SHA256：`0890cf709fd61d36f43222e094d1ea8cc2ce83fecb79731ff21fc552eb8ec8a6`
- CANN libascendcl SHA256：`a49ef2a9d5edcc764355d4456e11407401a4efb76889cf83efd360b5416d6c08`
- 原测试文件哈希与本地提交一致；kernel、Host、Python 包、golden、阈值均未修改，没有重新编译、提交或推送代码。
- 只新增仓库外诊断脚本、捕获数据和本报告。原检查仍使用 `abs_error > 0.05` 且 `relative_error > 0.15` 判定超差，要求超差数为 0。

## 3. 分层证据

### 3.1 末级 ptrDX 矩阵乘：固定实际输入后通过

从原运行保存实际 `gradZ`、`ptrDX` 和权重，CPU 计算 `BF16(actual_gradZ @ W1)`：

| rank / expert | 检查元素数 | 原阈值超差数 |
| --- | ---: | ---: |
| 0 / 0 | 931,840 | 0 |
| 0 / 1 | 931,840 | 0 |
| 1 / 2 | 931,840 | 0 |
| 1 / 3 | 931,840 | 0 |

共 3,727,360 个元素均通过。这里“通过”不是全矩阵逐位一致，CPU 与设备矩阵乘仍有少量末位差异。

进一步把原来失败的 19 个点分别做 FP64 点积，再舍入到 BF16：**19/19 与设备 ptrDX 逐位一致**。保存的最小数据包在 Windows 本地用独立 NumPy 再算一遍，也是 19/19 一致。因此这 19 个失败点不是末级矩阵乘或 ptrDX 读取偏移造成的。

### 3.2 上游差异集中在四行

| rank | 全局 expert | packed row | 全局 token | ptrDX 超差点数 | 主要上游来源 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 73 | 73 | 3 | dHp 的 BF16 舍入不同 |
| 0 | 1 | 222 | 92 | 7 | Gate 的 BF16 舍入不同 |
| 0 | 1 | 250 | 120 | 3 | Gate 的 BF16 舍入不同 |
| 1 | 2 | 117 | 117 | 6 | dHp 的 BF16 舍入不同 |

将 CPU golden 的 Gate/Up 换成实际设备 Gate/Up，仅作离线归因，10 个以 Gate 差异为主的超差消失；另 9 个仍存在。原 golden 和原验收结论均保留。

### 3.3 不改 SO，测出同输入 dHp

原 kernel 的 dHp GM 区随后被 Hp 覆盖，返回后无法直接从该区读取 dHp。使用既有 `backward_debug` 入口做诊断输入实验：保持 dY、W2、路由、维度、保存模式和原 SO 不变，在独立的 saved-state 副本中设置 Gate=64、Up=0，并设置 p=1/64。此时：

```text
dHp = BF16(dY @ W2)                  # 不依赖 Gate、Up、p
dH  = BF16(dHp / 64)
dUp = BF16(dH × SiLU(64)) = dHp      # 在本输入范围内，2 的幂缩放可精确还原
```

`1 + exp(-64)` 在 FP32 中为 1，故该路径 SiLU(64)=64。这个构造只用来观察 dHp，不是原模型的合法训练前向/反向配对，不计作精度验收。原运行的 saved-state 没有被替换。

第一次诊断脚本试图通过 autograd 对修改后的 saved-state 求导，被 PyTorch 版本检查拒绝，尚未执行反向 kernel；日志保留。随后改为独立副本和已有 `backward_debug` 输入接口完成观察，未绕过或修改原测试。

实测 dHp 与从实际 dGate/dUp 反推的值一致。将**实际 Gate/Up + 测得的 dHp + 原 p**放入 CPU SwiGLU 公式：

| 全局 expert | gradZ 元素数 | 不逐位一致数 | 最大绝对差 | 原阈值超差数 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,064,960 | 0 | 0 | 0 |
| 1 | 1,064,960 | 0 | 0 | 0 |
| 2 | 1,064,960 | 1 | 0.0009765625 | 0 |
| 3 | 1,064,960 | 0 | 0 | 0 |

4,259,840 个 gradZ 元素中仅 1 个末位差异，全部通过原阈值。再沿用这些输入计算 ptrDX，四个专家均为 0 超差。这将主要误差定位在 SwiGLU 之前的 GEMM 输出，而不是梯度公式或数据搬运。

## 4. 具体数值：为何很小的累加差异变成 ptrDX 超差

### 4.1 rank0、expert0、token73 的 dHp[1043]

| 项目 | 数值 |
| --- | ---: |
| FP64 点积参考 | -5.609375923479092 |
| CPU FP32 点积 | -5.609375953674316 |
| 相邻 BF16 数的分界 | -5.609375 |
| CPU golden 舍入后的 dHp | -5.625 |
| 原 SO 诊断测得 dHp | -5.59375 |

FP64 点积距离分界仅约 `9.23e-7`。两个 BF16 结果相邻，相差 `0.03125`。固定实际 Gate=4.375、Up=-2.203125、p=0.54296875 后：

| 中间量 | CPU golden | 设备 |
| --- | ---: | ---: |
| dGate[1043] | 7.0 | 6.96875 |
| dUp[1043]，即 gradZ[5139] | -13.1875 | -13.125 |

这两个 gradZ 分量在 ptrDX[row73,col608] 中的误差贡献分别为 `-0.0201416015625` 和 `-0.060546875`，其余分量合计约 `-0.00134239`。FP64 点积差为 `-0.0820308700`，足以解释最终差异：

```text
CPU golden ptrDX =  0.0634765625
设备 ptrDX       = -0.018310546875
```

该点的乘积绝对值之和约为 5978，而正负相抵后的结果仅约 0.02～0.06，因此对少数 BF16 末位变化敏感。不是最后一个小数凭空算错。

### 4.2 rank1、expert2、token117 的 dHp[807]

FP64 点积参考 `5.265622859958967`，距离 BF16 分界 `5.265625` 约 `2.14e-6`。CPU golden 为 `5.25`，原 SO 测得 `5.28125`。相应 dGate 从 golden 的 `-10.375` 变成设备的 `-10.4375`，dUp 从 `5.0625` 变成 `5.09375`，解释该行 6 个 ptrDX 超差的主要部分。

### 4.3 rank0、expert1、token92 的 Gate[714]

FP64 GMM1 点积参考 `3.6796878483801265`，距离 BF16 分界 `3.6796875` 约 `3.48e-7`。CPU Gate 为 `3.6875`，设备为 `3.671875`。使用实际 Gate 重算后，该行 7 个 ptrDX 超差消失。

CPU 也不是所有坐标上都更接近 FP64：例如 token73 的 dHp[950]，FP64 为 `-0.7792968222493073`，设备 `-0.78125` 更符合高精度舍入，而 CPU golden 为 `-0.77734375`。这说明不能把一个有限精度 GEMM 实现直接当作另一个实现的逐位标准。

## 5. 对 kernel / golden 的判断及下一步

| 判断对象 | 本次结论 |
| --- | --- |
| ptrDX 地址、布局、末级矩阵乘 | 19 个失败点均得到实际输入的 FP64 逐位复核；没有发现此处计算错误 |
| SwiGLU 反向公式、张量 BF16 边界 | 固定真实上游输入后约 426 万个元素仅 1 个末位差异，没有发现公式或边界错误 |
| 前向 GMM1 / 反向 GMM2 的数值一致性 | 已确认输出 BF16 舍入档位与 CPU 不同，是本次失败的上游来源 |
| 原 packed golden | 数学链路成立，但串联自己的上游近似，不能充当末级单算子的同输入验证 |
| 原端到端精度验收 | 继续保留 FAIL；本报告不替代原验收，不放宽阈值 |

建议讨论后再动代码：保留独立端到端 golden，同时新增“固定实际输入”的分阶段检查，用来区分 GMM1、GMM2、SwiGLU 和 dX 的责任。分阶段检查通过不应覆盖端到端失败。

如果要求此 shape 在现有端到端阈值下严格做到 0 超差，下一步应单独评估前向 GMM1 / 反向 GMM2 的累加精度和 BF16 输出策略，并检查性能代价；不能只把 golden 改成设备结果使它通过。若要精确到硬件累加/转换细节，则需要另行同意的临时 FP32 累加输出诊断，本轮未做。

不能简单把 CPU GEMM 换成 `torch_npu` GEMM 作为逐位标准：本轮独立库 GEMM 在两个关键 dHp 坐标仍与原 CATLASS kernel 不同，说明归约路径也需要明确。

另有一个独立的诊断打印问题：原 `diagnose_grad_x` 只累加本 rank 所属专家的贡献，却与跨 EP 汇总后的最终 dX 比较，因此日志中的 recon/got 约 0.67 不能解释为丢了约三分之一梯度。这不参与原 ptrDX 判定，本轮未修改。

## 6. 源码与证据

源码位于 `本地诊断工作区（公开仓不含原始张量）`，保持 `b8cff148`，工作区干净。

- `test/megamoe_bf16_test/test_mega_moe_backward.py:207`：CPU 自行计算 Gate/Up。
- 同文件 `:239`：CPU 自行计算 dHp；`:248`：gradZ；`:255`：ptrDX；`:464`、`:870`：原检查逻辑与阈值。
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward_catlass_impl.h:306`：dHp/Hp 复用；`:390`：反向 GMM2 的 CATLASS 调用。
- `kernels/mega_moe_bf16/include/block_epilogue_swiglu_backward_bf16.h:1008`：FP32 SwiGLU 与 dH/dZ 的 BF16 边界。
- `kernels/mega_moe_bf16/include/mega_moe_bf16_backward.h:1057`：ptrDX 输出。
- 测试文件 `:549`：只累加本地专家的旧诊断打印。

本地证据目录：`本地诊断工作区（公开仓不含原始张量）`。

- `capture.log`：原用例完整复现，原阈值和 FAIL 保留。
- `analysis_rank_0.json` / `analysis_rank_1.json`：分阶段统计、19 个失败坐标与误差贡献。
- `probe_analysis.json`：同输入 dHp 诊断和 gradZ/ptrDX 重建证据。
- `minimal_evidence.npz`、`failed_points.json`：可离线复算 19 个点的输入；`local_verification.json` 记录 Windows FP64 复核。
- `evidence_sha256.json`：最小数据包、原测试和 SO 指纹。
- `cleanup.json`、`final_cleanup.txt`：本任务容器已停止、设备预约已释放。

远端保留完整冻结张量于 `独立诊断目录中的 ptrdx_capture`，以便后续讨论后复查，无需再次占卡即可分析已有数据。

## 7. 同步前的 ascend_megamoe_bf16 是否已有问题：旧版实测补充

**有。同步前 `73c1b0ce780ca2a47dab0de04b9da7975a958338` 已有同类上游舍入差异及 ptrDX 超差，不是本次同步新引入。**

使用同步前已经编译并保存的原 SO，未改旧源码、未重新编译，加载上文冻结的完全相同输入，在同一隔离节点与环境验证。旧 kernel SHA256 为 `4c8fc8229ced888cf04c22ff66048344ea834bee98566d985a0b0dbdf1f79bab`；旧 Host SHA256 为 `5392e0305f329c37dadda5477142dd07f67e115e070bbb85a6b198091d4dcd6e`。

最关键证据：**旧版与新版的 Gate、Up、诊断测得 dHp 在两张卡上全部逐位一致**，每种张量合计 2,129,920 个元素。这包括前述临近 BF16 分界、导致差异放大的关键值。

旧版 SwiGLU 额外将 SiLU 和 dS 存为 BF16，不能直接用新版数值公式判旧版。离线分析按旧源码重建以下边界，原测试文件和阈值不动：

```text
dH    = BF16(p * dHp)
SiLUb = BF16(gate * sigmoid(gate))
dUp   = BF16(dH * SiLUb)
dS    = BF16(dH * up)
dGate = BF16(dS * (sigmoid(gate) * (1 + gate * (1 - sigmoid(gate)))))
ptrDX = BF16(concat(dGate, dUp) @ W1)
```

按各自数值语义的 CPU 链路对照，均使用 atol=0.05、rtol=0.15、超差数要求为 0：

| 版本 | rank0 ptrDX 超差 | rank1 ptrDX 超差 |
| --- | ---: | ---: |
| 同步前 73c1b0c | 13 / 1,863,680 | 7 / 1,863,680 |
| 同步后 b8cff148 | 13 / 1,863,680 | 6 / 1,863,680 |

这是固定输入的诊断结果，不表示新版对所有场景精度更好；新旧 SwiGLU 舍入语义不同，具体超差坐标也不完全相同。

旧版另外两项交叉验证均通过：

- 固定旧设备实际 gradZ，CPU 重算 ptrDX：四个专家、3,727,360 个元素均为 0 超差。
- 固定旧设备实际 Gate/Up 和测得 dHp，按旧 SwiGLU 规则重算：4,259,840 个 gradZ 元素**全部逐位一致**。

还确认了旧测试的缺口：`73c1b0c` 的 packed reference 没有 ptrDX 项，decoder 和检查仍把返回后的复用内存当成 ptrDO，但旧 kernel 已明确 `ptrDX = ptrDO`（saved 模式）。因此“旧日志没有 ptrDX 失败”不能说明旧版没有问题。上表旧版计数是本轮补充的语义匹配诊断结果，不冒充旧测试原先就有该检查。

证据：同目录 `baseline_check.log`、`baseline_rank_0.json`、`baseline_rank_1.json`、`baseline_summary.json`、`baseline_runtime_provenance.txt`。原/新分支源码和阈值均未修改；`baseline_cleanup.json` 确认本任务容器停止、预约释放。
