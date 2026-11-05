# MegaMoE 训练迁移：两人协作资料（仅当前分支）

本目录仅服务 `xixing-321/DeepEP:feat/megamoe-train-bf16` 的迁移、专项复核和细粒度
优化。**本目录及其中过程数据不得进入个人仓 main，也不得进入 lueelu/DeepEP 的正式 PR。**
目录名、`.gitignore` 和 `export-ignore` 都不能自动阻止 Git 合并。

## 阅读顺序

首次参与验证，先看 [GitHub 获取代码与协作验证指南](GitHub获取代码与协作验证指南.md)：获取训练分支、蓝区离线传输、找到验证入口，以及提交反馈/PR。

当前公开文档为[用户指南](../../experiments/megamoe_train/docs/BF16_USER_GUIDE_ZH.md)与[前后向设计](../../experiments/megamoe_train/docs/mega_moe_bf16_design.md)，分别说明接入、接口、窗口容量与限制，以及代码架构、计算、状态和同步。原[框架设计历史快照](BF16_FRAMEWORK_DESIGN_ZH.md)只保留迁移沿革，不作为当前接口说明。

1. [实施与分工方案](实施与分工方案.md)：固定源 b8cff148，你先整体迁移，再两人专项复核。
2. [首轮迁移验收](首轮迁移验收.md)：构建、精度、生命周期、已知失败与尚未验证的范围。
3. [EP4 / EP8 性能对照](EP4_EP8性能对照_20261028.md)：本轮同输入、同节点 ABBA 配对结果。
4. [H7168 ptrDX 精度归因](H7168_ptrDX精度归因.md)：保留原失败，不放宽阈值。
5. [BF16 宏清理与验证](BF16宏清理与验证_20261031.md)：统一前反向整专家逃生开关，保留 SO/API，记录 CANN 9.2 完整构建与基本精度结果。
6. [H3584 清理前后同输入对照](H3584清理前后同输入对照_20261031.md)：原 `599edd4` 与当前默认输出、参考数组及误差记录一致，确认所测误差不是清理引入。
7. [正式文档与注释整理记录](正式文档与注释整理记录_20261031.md)：保留移出的过程说明、原文出处、正式入口及本轮检查结果。
8. `historical_source/`：源版本的历史说明；其中旧数据不能代替当前迁移结果。
9. `data/`：各轮结构化记录，保留输入/输出哈希、每 rank 原始时间样本与汇总。
10. `implementation_notes/`：迁移期间的框架、数值、形状与拓扑、Dispatch 和向量搬运说明；其中必要的公开契约已整理进上述两份正式文档。

## 2026-11-04 说明归档

以下六份工作说明从 `experiments/megamoe_train/docs/` 移入本目录的 `implementation_notes/`，供迁移审阅追溯：

- [框架说明](implementation_notes/BF16_FRAMEWORK_DESIGN_ZH.md)
- [SwiGLU 数值说明](implementation_notes/BF16_SWIGLU_NUMERICS.md)
- [形状与拓扑说明](implementation_notes/BF16_SHAPE_AND_TOPOLOGY_LIMITS.md)
- [反向 Dispatch 说明](implementation_notes/BF16_BACKWARD_DISPATCH_READY.md)
- [前反向就绪策略说明](implementation_notes/BF16_DISPATCH_READY_DEFAULTS.md)
- [向量与路由搬运说明](implementation_notes/BF16_VECTOR_ROUTING_OPTIMIZATION.md)

正式文档及模块 README 不依赖上述工作说明。上游的 `experiments/megamoe_train/docs/visual/` 继续作为公开源码导览维护；本目录的归档不移动该导览及其维护说明。

硬件地址、个人机器路径、SSH 信息、完整终端日志和冻结张量不上传公开仓。原始证据
继续保留在本地诊断目录。公开 JSON 中的哈希和时长足以复核本文的统计计算。

## 版本与责任

- 源：`b8cff14854d12aef77b6ae54b5890123759b4d28`。
- 首轮迁移及 EP4/EP8 性能报告被测代码：`ae530ad9d1b50d95d785f5ea2d506a6be88dcf2e`，
  该轮文档整理不改 kernel、Python 执行路径或 native SO。后续宏清理以 `599edd4` 为基点，
  精度及清理前后 H3584 对照见上述独立报告；各轮实际加载产物以对应报告的 SHA256 为准。
- 你负责整体集成，以及 kernel / Host 算法、容量、协议、数值与性能复核。
- 同事负责框架接入、安装、EP 分组、多机、microbatch / stream 和实际短训练复核。
- 双方共同确认兼容性、维护主备、配套升级和发布范围。

## 正式交付的排除规则

当前协作分支的过程资料不属于正式交付。本目录在该分支可见，也会出现在以该分支为
head 的 PR diff 中；**禁止把包含本目录的协作 PR 直接合入 main 或 upstream**。

1. 产品代码/长期接口文档与本目录分别提交，避免代码修改混在过程资料提交中。
2. 正式投稿时，从确认的 main/upstream 基点另建干净交付分支，只导出审查通过的代码、
   测试和长期接口/算法文档；不要整支 merge，也不要保留中间资料提交后再简单删目录。
   若需要净化历史，应从净变更形成正式提交，而非把本协作历史带入目标仓。
3. 该分支不得含 `docs/megamoe_train_migration/`；同时检查旧位置没有残留的迁移方案、
   定位日志、源同步报告、精度/性能原始数据或指向临时资料的链接。
4. 对实际 PR 的 base/head 复核完整 diff 和提交范围。以下检查应无输出：

   ```bash
   git diff --name-only <base>...<release-head> -- docs/megamoe_train_migration
   git ls-tree -r --name-only <release-head> -- docs/megamoe_train_migration
   ```

5. 临时材料排除后，重新执行来源检查、文档链接检查和发布验证；双方确认后再决定
   合并。正式 README 保留必要的支持限制和已知问题，不靠临时记录提供使用契约。

本协作分支的 `MANIFEST.in` 另行排除本目录，避免开发阶段的源码包夹带资料；这仅
控制打包，不能代替上述 Git/PR 排除检查。本轮不创建正式交付分支，不合并任何 PR。

## 源文档整理说明

原 `experiments/megamoe_train/docs/MIGRATION_PLAN_ZH.md`、`VALIDATION.md` 和
`BF16_TRAIN_SYNC_20261028.md` 中的过程资料已集中到本目录。其余算法说明删除实验
日期、旧性能表和迁移流水记录，保留长期计算/调度契约。原始来源映射与必需文件检查
工具已在统一署名整理中移除，来源说明保留在模块 README；导入文件数从首轮的 82 变为 81。
