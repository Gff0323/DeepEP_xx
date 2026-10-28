> 源 b8cff148 的算法/历史说明；本次 GitHub 迁移验证以 [首轮迁移验收](../首轮迁移验收.md) 为准。

# MegaMoE BF16：同步 train 优化与简化验证（2026-10-28）

## 结论与版本

把 `train@6d8f83359e992712218764496794fa7e6063d81b` 的三组修改同步到 `ascend_megamoe_bf16`，目标基线为 `73c1b0ce780ca2a47dab0de04b9da7975a958338`。
使用保留双方历史的合并，不用 train 整树覆盖开源分支。当前工作只更新 GitCode 的 BF16 开源源分支，不代表已接入 GitHub DeepEP。

编译、CPU 回归和五组设备精度用例通过。另一组 H7168 用例的四类最终梯度通过原有阈值，但额外 packed ptrDX 检查失败；同输入在原 train 中完全复现，保留为已知验收缺口，不能称为精度全部通过。

## 同步范围

1. `014e154`：前反向 SwiGLU 向量计算、路由权重批量读取、X/dY 分组搬运、TopK 两窗口，以及反向 Dispatch 行进度调度。
2. `20f79db`：前向所有符合条件专家的 M192/Ready384 细分、前反向静态 producer 元数据、Host 默认每卡专家数 7/8 的分界及配套 workspace。
3. `e59c09b`：反向重建 H 与 dUp/dGate 使用与前向一致的 FP32 中间量，保留最终 BF16 张量边界、clipping 掩码与原梯度定义。

保留 Ascend950 专用定位、`ascend_megamoe` 包名、四种保存模式、Python/autograd 接口及无 kernel Timer 的开源结构。底层反向 ready-k 入口与 Host 配套调整，必须同时重编 Host 和 kernel。

CATLASS 固定 `abeb80893e45c917d8148de3249290b0a23741c8`，344 个头文件与该提交逐项核对（两侧统一 LF 后相同）。SHMEM 继续固定 `f4a4dfdb6bc47a9ba4aded2b20c38638a3ddb44c`，验证复用已构建的同版本 TCP 80 秒重试补丁库；没有升级依赖或引入 CATCCOS。

另外修复开源整理遗留的测试依赖：从 train 的 W8A8 golden 原样提取纯 MXFP8 量化参考函数，移除对已删除 W8A8 文件和头文件的依赖；清除 autograd mock 中已不存在的 os 注入。未引入 W8A8 算子，未放宽阈值。

17 个变化 kernel 文件逐项与 train 核对，除命名/路径及 Timer 专用参数、代码、常量清理外，算法保持一致。Host 的策略查询、workspace 容量、反向符号声明及调用参数配套核对。

## 简化精度验证

- CPU：77 个 Python 测试、22 个子用例；4 个 C++ 调度、覆盖与容量测试通过。
- 两版 Ascend950 kernel 与 PyTorch C++ 扩展均编译成功。构建使用相同 CANN 9.2 beta.2、CATLASS 和 SHMEM。
- 双卡前向独立 CPU golden：M1025/H256/F512/每卡2专家/TopK4/bf16，覆盖多 M 块与尾块，检查 Y、专家计数和 packed TopK 权重，通过原阈值。
- 双卡、小 shape：M129/H256/F512/每卡2专家/TopK4。bf16+clip0、none+clip0.05、mxfp8+clip0.05、auto+clip0 均通过最终四类梯度和中间量检查。
- 双卡、M65/H3584/F3072/每卡2专家/TopK4/bf16：最终梯度和中间量检查通过。
- 双卡、M65/H7168/F4096/每卡2专家/TopK4/bf16：最终 dX、dTopK、dW1、dW2 按现有阈值通过。额外 ptrDX 检查，rank0 为 13/1,863,680、rank1 为 6/1,863,680 个元素超差，最大绝对误差 2；因此该用例整体判定失败。

H7168 对照：原 train 的 82 个 native 文件与复用参考构建源码完全相符，参考 SO 指纹已核对；用相同输入运行，全部中间量和最终梯度检查统计与同步版一致。本次检查未观察到因移植新增的误差，尚未确定原中间量误差的根因。未修改算法来消除它，也未调整检查阈值。

最终梯度“通过”沿用仓库的误差比例门槛，不等于每个元素逐位相等或零超差。该 H7168 用例最终 dX 仍存在约 1.3% 元素超过单元素门槛，但未超过原测试允许的比例；完整统计在结构化记录中保留。

## 常用 shape 性能短测

仅测 BF16 保存模式，balanced 路由、无 clipping。旧版/新版/新版/旧版，每个窗口 5 次 warmup、20 次采样。使用同机同卡、相同输入，独立进程加载对应 Host/kernel。
指标为：每次迭代取所有 rank 的最大 NPU Event 耗时，再取窗口 p50；表中是同版本两窗口的均值。变化为新/旧-1，负数更快。前向和反向分别测量，不能将二者之和当成整网 step 实测。

| 场景 | EP | M/H/F/每卡专家/TopK | 前向旧→新 ms | 变化 | 反向旧→新 ms | 变化 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| k3 | 4 | 8192 / 3584 / 3072 / 7 / 16 | 24.322 → 23.728 | -2.44% | 49.010 → 47.719 | -2.63% |
| common | 8 | 4096 / 7168 / 4096 / 4 / 16 | 29.512 → 29.242 | -0.91% | 61.897 → 61.167 | -1.18% |

所有窗口保存完整输入/输出哈希、计数、finite 检查及原始 Event 样本。反向数值语义有意修正，不要求旧/新梯度逐位相同；该性能检查不代替独立大 shape 梯度 golden。
测试节点原有 Alarm/Warning，因此这里只作为短配对回归，不能外推其他设备、路由分布或更大 EP。逐窗口 p50/p95、AB/BA 方向和数值采样差异保留于结构化记录。

## 验证边界和后续事项

1. H7168 packed ptrDX 与 CPU 分阶段参考的少量差异需后续排查；不能把源 train 中也复现当作该检查本身已通过。
2. 尚未做独立完整大 shape 梯度 golden、整网 loss 收敛、多 microbatch 生命周期、长稳、大 EP 或跨节点回归；性能仅覆盖上述两组 BF16 场景。
3. 首次验证目录漏拷贝 SHMEM 库而在导入阶段失败，按正式构建脚本补齐后重跑。参考 train 目录也补齐了原包依赖的基础 kernel 库。这些启动失败没有触及 NPU 算子计算，日志保留。


本次结构化摘要：[bf16_train_sync_20261028.json](https://gitcode.com/fft__/ascend_deepep_hkq/blob/b8cff14854d12aef77b6ae54b5890123759b4d28/docs/data/bf16_train_sync_20261028.json)。其中包含原始两轮窗口、输入输出哈希、精度检查统计与验证边界。
本模块使用 `python experiments/megamoe_train/scripts/check_source_import.py` 检查完整导入；原 GitCode 分支发布门禁不直接用于本 fork。
