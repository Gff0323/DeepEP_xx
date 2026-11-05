# MegaMoE BF16 形状与拓扑限制（用户指南）

> 迁移期间的实现说明，2026-11-04 归档。当前公开接口、限制和算法说明以[用户指南](../../../experiments/megamoe_train/docs/BF16_USER_GUIDE_ZH.md)与[前后向设计](../../../experiments/megamoe_train/docs/mega_moe_bf16_design.md)为准；本文保留迁移审阅上下文。

> 适用：`experiments/megamoe_train` 的 BF16 training 组件（`MegaMoeBF16Train`），仅面向 **Ascend 950**。
> 本文件回答两个问题：**哪些量有上限 / 必须满足什么条件**，以及 **为什么**。
> 代码中的硬检查均以 `TORCH_CHECK` 实现（超限直接报错）；文中标注"静默"的项超限**不会报错**，需要用户自己保证。

完整安装、接口与容量使用方法见[用户指南](../../../experiments/megamoe_train/docs/BF16_USER_GUIDE_ZH.md)；下文保留各限制及其实现原因。

## 1. 专家与拓扑限制（硬检查，超限报错）

| 限制 | 值 | 说明 |
|------|-----|------|
| EP rank 数 `world_size` | **2 ≤ world_size ≤ 1024** | EP 组必须至少 2 卡 |
| 本地专家/卡 `expert_per_rank` | **≤ 32** | 每张卡上切的专家数 |
| 总专家数 | **`expert_per_rank × world_size ≤ 2048`** | EP 组内全部专家 |
| `topk` | **`≤ expert_per_rank × world_size`** | 不能路由到比总专家数更多的专家 |
| `world_size` vs 机器 AIC 数 | **`world_size ≤ 10 × aic_num`** | 仅非 direct-dispatch 模式检查 |
| 拓扑 | **当前仅单节点已认证** | 多节点/EP 子组/更大 EP 需另行认证 |

## 2. 形状与对齐限制（硬检查，超限报错）

| 限制 | 值 |
|------|-----|
| 每卡每次调用的 token 数 `m` | **≤ 16384**（batch 的 token 按 rank 摊，总 batch 可达 `m × world_size`） |
| hidden `k` | **`k % 256 == 0`** |
| `n` | **`n % 512 == 0`**（且 k>0、n>0、topk>0、m>0） |
| `max_recv_tokens` | **> 0**；backward 还要求 **≤ int32_max** |

## 3. dtype 与张量形状（硬检查）

- `x` / `grad_y` / `topk_weights` / `weight1` / `weight2`：**bfloat16**；
- `topk_ids`、`expert_token_nums`：**int32**；
- `x`、`topk_ids` 为 rank-2；`weight1/2` 为 rank-3；`topk_weights` 的 shape 必须与 `topk_ids` 一致；
- 所有 tensor 必须 contiguous、在 NPU 上。

## 4. 软限制（超限不报错，退化为粗粒度 / 老路径）

| 限制 | 值 | 后果 |
|------|-----|------|
| **compact 计数交换** | 本地专家 `≤ 8` 才可用；EP ≥ 512 才自动启用 | 超限或 EP 小 → 退回 dense 计数交换（旧布局，仍正确） |
| **saved 模式的 row-ready 流水** | 默认在本地专家 `≤ 7` 时启用；还受保存模式、实际行数和构建选项约束 | 其他情况使用整专家完成标志；强制整专家构建始终关闭 row-ready |

## 5. ⚠️ 静默边界：`max_recv_tokens`

`max_recv_tokens` 是**当前 rank 所有本地专家共享的总接收行容量**。容量不足时设备路径会截断，Host 入口不会根据实际路由计数提前报错；要完整计算该批输入，调用方必须保证容量覆盖总接收行数。

- **必须覆盖最繁忙 rank 的总接收行数**。在每个 token 的 expert ID 不重复时，可保守取：
  ```
  max_recv_tokens ≥ EP × M × min(E_local, topk)
  ```
  （`E_local` 是每个 rank 的本地专家数，`M` 是每个 rank 的 token 数；若已知实际路由需求，可按该需求设置。）
- 重复的 expert id **不属于支持的契约**。

## 6. 为什么是这些限制

| 限制 | 根因 |
|------|------|
| 1024 EP / 2048 总专家 / 32 本地 | **workspace 与 peermem 布局按这些上界静态开尺寸**（rank 窗口数组 `epHcclBuffer[1024]`、同步槽区、每卡专家表），代码里有 `static_assert` 锁死 |
| 16384 token | 路由/dispatch 簿记容量：`expandedRowIdx`（m×topk）、dispatch 缓冲、UB 批处理都以它为预算 |
| `topk ≤ 总专家数` | 语义性约束：不可能路由到不存在的专家 |
| `k%256`、`n%512` | **DMA 32B 对齐 + tile 划分**：DataCopy/DataCopyPad 的对齐要求；n=2F，F 需要对齐 |
| `world_size ≤ 10×aic_num` | dispatch 簿记（每源 rank 的 producer 计数槽）按 AIC 核数预算；direct-dispatch 模式不经过这套簿记所以豁免 |
| compact 交换 ≤8 本地专家 | compact 计数包的布局每个目标 rank 只留 8 个专家的计数位（`static_assert` 保证不越界） |
| row-ready 默认 ≤7 专家 | `DefaultBackwardDispatchSplitExperts` 的默认策略阈值；进度缓冲按实际启用数量规划，不是固定 7 专家的内存上限 |
| `max_recv_tokens` | 路由数据按此容量分配、截断；数值范围检查不等于实际接收行数充足 |

## 7. 误用会得到什么

- **第 1~3 节**：违反已有 Host 检查时抛出异常；输入合法性仍需满足接口约定，不能把表格视为所有误用的完整检测清单。
- **第 4 节**：按实际策略选择行就绪或整专家路径。构建选项与输入条件共同决定执行路径。
- **第 5 节**：容量不足会截断路由，无法得到完整的前后向结果；调用前应按总接收行需求规划容量。

## 8. 与其它文档的关系

- 前后向详细设计（计算/状态/同步）：[mega_moe_bf16_design.md](../../../experiments/megamoe_train/docs/mega_moe_bf16_design.md)
- dispatch 策略：BF16_DISPATCH_READY_DEFAULTS.md、BF16_BACKWARD_DISPATCH_READY.md
- 数值语义：BF16_SWIGLU_NUMERICS.md
