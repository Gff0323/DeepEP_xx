# MegaMoE 训练迁移与双人复核方案

更新：2026-10-28。迁移目标是公开个人 fork `xixing-321/DeepEP`；
`lueelu/DeepEP` 为 upstream，后续投稿另行安排。私有仓保留，不参与本轮。

## 固定版本与交付范围

- 目标起点：`3a7f692e668e459aac59dfe459ad70a88b14271b`。
- 唯一源：`ascend_megamoe_bf16` 的 `b8cff14854d12aef77b6ae54b5890123759b4d28`。
- 完整迁移 Ascend950 BF16 前向/反向、四项梯度、四种保存模式和 SwiGLU clipping。
- `experiments/megamoe_train` 与推理 `experiments/megamoe` 同级；
  `csrc/bf16` 放算法 Host，`csrc/kernels/bf16` 放内核和专用组件，Python 在
  `deep_ep_experimental.megamoe_train`，测试在 `tests/bf16`。
- 后续 W8A8 分别增加 `csrc/w8a8`、`csrc/kernels/w8a8`、Python `w8a8.py`
  与 `tests/w8a8`；不预建空实现，不把 BF16 保存格式 MXFP8 当作 W8A8 交付。

## 新分工

第一阶段由原内核负责人先把双方的实施工作整体完成并统一集成，包括内核、算法
Host、框架 Host、运行时、构建、Python/autograd、测试与文档。第二阶段两人针对
固定候选专项复核，仍由第一阶段负责人统一收敛修改。

| 复核方 | 重点 |
| --- | --- |
| 内核负责人 | 源范围完整性、tiling、容量、ready/generation、数值边界、性能 |
| 框架/验证同事 | 干净安装、EP 分组、stream、saved state、microbatch、错误处理、短训练 |
| 双方 | ABI、SHMEM 生命周期、公共文件变更、已知失败和最终发布范围 |

## 实施和验收

1. 更新中文方案，固定源/目标提交和整批组件清单，建立独立迁移分支。
2. 迁移全部 BF16 源组件，保留原版权与许可，记录源路径和目标路径。
3. 训练复用根 Session 唯一 owner；初始化、对称窗口、context 和显式 close 有契约。
4. 独立默认关闭的构建开关与 wheel；使用根依赖锁，禁止静默加载旧 SO。
5. CPU 契约检查、NPU native 编译、两卡真实前向/反向/释放闭环。
6. 覆盖四保存模式、clipping、全部梯度、布局、容量、路由与重复调用；逐项留证。
7. 常用 shape 同输入配对性能；安装包脱离源码验证；受影响旧模块回归。
8. 固定交接 commit，记录通过、失败、未测和环境，开展两人专项复核。

迁移前后的依赖不同须单列，不把目录迁移和依赖升级的影响混为一谈。测试期间
使用隔离工作区和库路径，确认设备空闲；不得替换其他任务的 SO 或终止他人进程。

## 已接受问题及后续决定

H7168 ptrDX 中间量的既有误差本轮保留，原阈值和原失败保持可见；新增迁移差异
仍须定位。最终梯度通过不等于实际 loss 收敛已验证，真实短训练由第二阶段推进。

团队还需明确首个框架接入方、microbatch 调度、长期主备维护人、设备/CI 资源、
CANN/SHMEM/CATLASS 升级节奏、W8A8 正式源版本和投稿 upstream 的时间。
这些事项不阻止本轮完整算子迁移，也不能因合入现有仓就自动视为已有人负责。

## 本轮执行状态

目录、BF16 代码、运行时、构建、接口和测试已迁移；训练精度与生命周期的实际结果
见 [首轮验收记录](VALIDATION.md)。性能、正式 CANN 配套、多机、整网与专项复核仍待
完成。训练接入使用根 MTE Session，与源测试的 UDMA 初始化选择不同，必须列入
多机/EP 专项复核。代码迁移完成不等于所有发布门槛通过。
