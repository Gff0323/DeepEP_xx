# MegaMoE BF16 训练框架设计

`MegaMoeBF16Train` 是面向 Ascend950 的 `torch.nn.Module`，提供融合前向和自定义自动求导。调用方提供 token、专家 ID、路由权重和本 rank 的专家参数；模块完成 routing、dispatch、GMM1、SwiGLU、GMM2 和 combine，并在反向返回输入、路由权重及两层专家参数的梯度。

专家参数和优化器由训练模型持有。模块不在内部注册 `w1/w2` 参数，不执行数据并行梯度归约。门控网络可以通过 `topk_weights` 的梯度继续反向传播；离散的 `topk_ids` 和返回的专家计数不参与求导。

本文说明当前接口、计算图状态和通信资源的归属。设备调度与数值细节见[就绪策略](BF16_DISPATCH_READY_DEFAULTS.md)及[SwiGLU 数值语义](BF16_SWIGLU_NUMERICS.md)，构建命令见[模块 README](../README.md)。

## 对外接口与调用契约

```python
import torch
import torch_npu

from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

# 调用方已创建 ep_group，并设置各 rank 一致、可达的独立
# DEEPEP_SHMEM_ENDPOINT=tcp://<host>:<unused-port>。
torch.npu.set_device(local_device)
op = MegaMoeBF16Train(ep_group, local_mem_size, timeout=120)

y, counts = op(
    x, topk_ids, topk_weights, w1, w2, max_recv_tokens,
    swiglu_limit=0.0,
    save_gmm1_output="bf16",
)
(y * grad_y).sum().backward()  # 示例；实际训练使用模型的 loss。

# 所有 rank 完成所需 backward 后，以相同顺序关闭，再销毁 ep_group。
op.close()
```

构造函数为 `MegaMoeBF16Train(ep_group, local_mem_size, *, timeout=120)`。计算接口为 `forward(x, topk_ids, topk_weights, w1, w2, max_recv_tokens, *, swiglu_limit=0.0, save_gmm1_output="bf16")`。

令 `M` 为当前 rank 的输入 token 数，`H` 为 hidden 维，`F` 为专家中间维，`T` 为每个 token 的路由数量，`E` 为每个 rank 的本地专家数，`EP` 为进程组大小。

| 参数或返回值 | 类型 / 形状 | 契约 |
| --- | --- | --- |
| `ep_group` | 已有 ProcessGroup 或 StoreGroup | 提供 EP 身份和初始化协调；不能为 `None`，由调用方持有 |
| `local_mem_size` | 整数，字节 | 当前 rank 的对称窗口容量，至少 2 MiB，按 2 MiB 对齐 |
| `timeout` | 整数，1..3600 秒 | SHMEM 初始化和控制操作超时；group 自身超时另由调用方设置 |
| `x` | BF16 `[M,H]` | 当前源 rank 的输入 |
| `topk_ids` | int32 `[M,T]` | 全局专家编号，按 `ep_rank * E + local_expert` 解释 |
| `topk_weights` | BF16 `[M,T]` | 对应路由权重，可参与求导 |
| `w1` | BF16 `[E,2F,H]` | 当前 rank 专家的第一层权重 |
| `w2` | BF16 `[E,H,F]` | 当前 rank 专家的第二层权重 |
| `max_recv_tokens` | 正整数，不超过 int32 上界 | 当前 rank 所有本地专家合计接收的路由行容量 |
| `swiglu_limit` | 可表示为 float32 的有限数值 | `<=0` 关闭 clipping；正数配置截断 |
| `save_gmm1_output` | `bf16 / none / mxfp8 / auto` | 选择供反向使用的 Gate/Up 保存形式 |
| `y` | BF16 `[M,H]` | combine 回当前源 rank 的输出 |
| `counts` | int32 `[E]` | 本地各专家接收的路由行数，不参与求导 |

Tensor 输入必须连续并位于构造时选定的同一 NPU。也支持权重布局 `w1=[E,H,2F]`、`w2=[E,F,H]`；Host 按形状识别，返回的权重梯度保持输入布局。

Host 的形状限制包括维度为正、`M<=16384`、`E<=32`、`EP*E<=2048`、`H%256==0`、`F%256==0` 及 `T<=EP*E`。训练 Runtime 的代码范围为 `2<=EP<=1024`；[StoreGroup](../../../deep_ep/_store.py) 自身还要求 `size<=128`。这些范围不代表对应拓扑都已通过设备验收。

调用方保证专家 ID 合法、同一 token 的专家 ID 不重复，且接收容量足够。一条 token 路由占一行；`max_recv_tokens` 不是原始 token 数，也不是窗口字节数。对各 rank 均有 `M` 个 token、每 token 专家 ID 唯一的情况，`EP*M*min(E,T)` 是当前 rank 路由行需求的保守上界；若采用更小容量，调用方需要能保证实际路由需求。

同一次集体计算要求各 rank 的形状、路由解释、保存模式和操作顺序相容。初始化时有配置协调，但每次前后向不逐项协商所有参数，也不在 Host 全量检查设备路由数据。

## 计算调用链与代码分层

```text
训练模型
  → MegaMoeBF16Train / _MegaMoeBF16TrainFunction
  → 训练 _C.BF16 / C++ BF16Executor
  → Host 校验、tiling 和 Torch Tensor 分配
  → launcher 提交设备入口
  → BF16 forward/backward kernel

MegaMoeBF16Train
  → _TrainRuntime / deep_ep._runtime.Runtime
  → 根 deep_ep._C / deepep::Session
  → SHMEM 对称窗口
      └─ 执行器借用地址；设备内核访问 peer 窗口
```

| 阅读入口 | 职责 |
| --- | --- |
| [bf16.py](../python/deep_ep_experimental/megamoe_train/bf16.py) | 公开 Module、自定义 autograd、流和生命周期检查 |
| [bindings.cpp](../csrc/bindings.cpp) | 注册训练 `_C.BF16` 及命名结果类型 |
| [training.hpp](../csrc/bf16/training.hpp) / [training.cpp](../csrc/bf16/training.cpp) | C++ 执行器、Tensor 契约、内存组织及前后向 Host 函数 |
| [tiling.cpp](../csrc/bf16/tiling.cpp) / [tiling_impl.hpp](../csrc/bf16/tiling_impl.hpp) | 平台资源、形状、保存模式及容量计算 |
| [launcher.cpp](../csrc/bf16/launcher.cpp) / [kernel_launch.hpp](../csrc/bf16/kernel_launch.hpp) | peer context 初始化、当前流与导出 C 入口 |
| [前向入口](../csrc/kernels/bf16/mega_moe_bf16_forward.cpp) / [反向入口](../csrc/kernels/bf16/mega_moe_bf16_backward.cpp) | 提交对应 Ascend C 主控类 |
| [layout](../csrc/kernels/bf16/layout) / [schedule](../csrc/kernels/bf16/schedule) / [epilogue](../csrc/kernels/bf16/epilogue) | Host/Device 布局契约、就绪调度与激活计算 |
| [Runtime](../../../deep_ep/_runtime.py) / [根绑定](../../../csrc/bindings/module.cpp) / [Session](../../../csrc/runtime/session.cpp) | 进程级通信资源和训练工厂 |

Host/Device 共用的 [mega_moe_layout.hpp](../csrc/kernels/bf16/layout/mega_moe_layout.hpp) 定义保存状态及工作区布局；[mega_moe_workspace_view.hpp](../csrc/kernels/bf16/layout/mega_moe_workspace_view.hpp) 将基址映射为设备地址视图。Python 不另行维护布局或量化模式镜像。

建议先按 `bf16.py → training.hpp → bindings.cpp → training.cpp → launcher.cpp → 设备入口` 阅读，再进入布局、调度和 epilogue。

## 自动求导与单次前向状态

`MegaMoeBF16Train.forward()` 经 `_MegaMoeBF16TrainFunction.apply()` 调用 C++ 执行器。Function 的 `ctx.op` 保留同一个 Python 模块，供反向检查设备、流和关闭状态；每次前向各有独立的 autograd context。

`ctx.save_for_backward()` 保存六个 Tensor：

```text
topk_weights, w1, w2, counts, saved_state, dispatch_output
```

同时保存本次 `max_recv_tokens`、`swiglu_limit` 和请求的保存模式。原始 `x/topk_ids` 不额外放入 `save_for_backward()`，反向通过 retained dispatch payload 和路由元数据取得所需信息。

前向返回 `y/counts`，并将 `counts` 标为 non-differentiable。反向先把 `grad_y` 连续化，再用对应前向保存的状态调用执行器，返回与九个 Function 输入位置对应的梯度：

| Function 输入 | 梯度 |
| --- | --- |
| `op` | `None` |
| `x` | `dx` |
| `topk_ids` | `None` |
| `topk_weights` | `dtopk_weights` |
| `w1` / `w2` | `dw1` / `dw2` |
| `max_recv_tokens` / `swiglu_limit` / `mode` | 各自为 `None` |

C++ 计算完整四类梯度，没有根据单个输入的 `requires_grad` 选择另一条内核路径。PyTorch 决定将哪些梯度继续传给上游。Function 使用 `once_differentiable`，不支持二阶求导。

| 保存模式 | Gate/Up 的保存与反向使用 |
| --- | --- |
| `bf16` | 保存 BF16 Gate/Up，反向直接使用 |
| `none` | 不保存 Gate/Up 数据面；反向从 retained token 重算 GMM1 |
| `mxfp8` | 保存 MXFP8 Gate/Up；开启 clipping 时还保存三个位平面 |
| `auto` | 关闭 clipping 时选择 MXFP8，开启时选择 BF16 |

所有模式都保留反向必需的路由元数据和 dispatch payload；`none` 不代表删除全部前向状态。MXFP8 是保存激活的格式，矩阵计算仍使用 BF16。Host 在前后向按同一规则解析规范化的 float32 limit；具体舍入、clip mask 和梯度语义见[数值说明](BF16_SWIGLU_NUMERICS.md)。

## Session 所有权与实际通信

构造前，调用方先创建 EP group、选定本地 NPU，并设置各 rank 一致且可达的独立 `DEEPEP_SHMEM_ENDPOINT`。该端口用于 SHMEM bootstrap，不应与训练框架 rendezvous 或测试协调服务争用同一监听端口。

构造时，`_TrainRuntime` 选择根扩展的 `create_megamoe_train_runtime`，协调容量、超时、bootstrap 地址和设备映射，并进行进程级互斥检查。根 Session 初始化 SHMEM、申请并清零对称窗口；模块执行 barrier 后，执行器借用窗口地址，保存各 peer 地址的持久 context。

训练 Session 使用 **MTE/default**。已有 ProcessGroup 或 StoreGroup 提供 rank/size 及初始化协调，实际 token 和梯度搬运由设备内核访问 SHMEM peer 窗口完成。传入 HCCL ProcessGroup不表示数据面改为 HCCL，也不表示跨节点传输已经通过验收。

同一进程同一时刻只能有一个 SHMEM owner，受 Python Runtime 和 C++ Session 的共同互斥约束。训练模块不能与另一个未关闭的 DeepEP buffer、推理 owner 或训练 owner 同时持有独立 Session。多层训练需要按该所有权契约组织复用和串行调用，不能通过创建多个模块绕过约束。

| 状态或内存 | 所有者及生命周期 |
| --- | --- |
| EP group / Store server | 调用方持有；算子关闭后再销毁 |
| SHMEM 对称窗口 | 根 Session 持有，构造至集体 `close()` |
| `context_` | C++ 执行器持有，模块级复用；只借用窗口地址 |
| `y/counts` | Torch 分配；counts 同时由计算图保留 |
| `saved_state/dispatch_output` | 每次前向独立的 Torch Tensor，由对应计算图保留 |
| 前向 transient workspace / backward workspace | 当前调用的临时 Torch Tensor，不作为结果保存 |
| `topk_weights/w1/w2` | 调用方提供，计算图保留引用；所需 backward 完成前不得原地修改 |

`local_mem_size` 只表示通信窗口容量，不是总显存占用。保存状态、结果和 scratch 还需要额外显存；未完成的前向计算图越多，保留的状态越多。

## 流、microbatch 与关闭顺序

前后向都检查模块未关闭、Runtime 有效、当前 NPU 未变，并要求当前流等于构造时的 `_stream`。launcher 在当前流提交内核；Host 对参与计算的 Tensor 调用 `NPUCachingAllocator::recordStream`，包括临时 workspace，以保护异步计算期间的分配器生命周期。

同一模块可按各 rank 一致的顺序排队多个前向，再执行它们对应的反向。每个前向有独立保存状态，但通信窗口和执行器 context 仍共享，因此调用必须在构造流上串行提交。`recordStream` 不会实现任意换流的依赖传递，也不会协调不同 rank 的调用顺序。跨流、并发调用和同时持有多个独立 Session 不属于该接口契约。

`close()` 的顺序为：

```text
检查模块与 Runtime 状态
→ 同步训练模块的构造流
→ Runtime.close() 同步内部流并关闭 Session
→ Session barrier、释放窗口、SHMEM finalize
→ BF16Executor.close() 清空借用地址和 context
→ 标记模块已关闭
```

所有 rank 必须以相同顺序显式关闭；成功后的重复 `close()` 直接返回。调用方应完成所有需要的 backward，或先丢弃不再使用的计算图，再关闭模块。实现不跟踪 pending graph，也不会因尚有计算图引用而阻止关闭；关闭后再前向或反向会被拒绝。若使用 `retain_graph=True`，必须保持模块开启，直到完成所有计划中的反向。

Python GC 不执行集体 SHMEM 清理，不能代替显式 `close()`。发生设备或集体操作失败后应重启分布式任务；不能假设仍可安全复用原 Session。EP group 仍归调用方，须在算子关闭后销毁。

## 构建边界与验收范围

根 `deep_ep._C` 提供 Runtime/Session 和训练窗口工厂；`deep_ep_experimental.megamoe_train._C` 提供训练 Tensor 接口。训练 `_C` 与 `libdeepep_megamoe_train_bf16.so` 安装在同一训练包目录，使用同目录设备库。构建由[根工作流](../../../build_support.py)和[训练 CMake](../CMakeLists.txt)组织，依赖遵循[根锁文件](../../../dependencies.lock.json)。

`DEEPEP_BUILD_MEGAMOE_TRAIN=ON` 控制可选训练组件，默认关闭，与推理组件开关独立。Host 扩展、设备库和根 Runtime 必须来自匹配的完整 wheel；修改构建策略后应完整重建、重新安装，各 EP rank 使用同一构建。不要只替换一个SO。前后向整专家回退及 SwiGLU 构建策略见[就绪策略](BF16_DISPATCH_READY_DEFAULTS.md)。

当前设备验收边界为单节点 Ascend950。API 接受 ProcessGroup/StoreGroup及其代码范围，不构成 HCCL 框架接入、EP 子组、跨节点或大规模 EP 的验收证明；这些需要对实际框架、group、设备映射和拓扑分别验证。单个 kernel/reference case 也不证明训练收敛、长稳或模型整网性能。

| 验证入口 | 能检查的内容 |
| --- | --- |
| [test_training_api.py](../tests/bf16/test_training_api.py) | 公开签名、参数校验和保存模式透传 |
| [test_mega_moe.py](../tests/bf16/test_mega_moe.py) / [前向 golden](../tests/bf16/mega_moe_bf16_golden.py) | 指定shape的前向输出、计数和可见舍入语义 |
| [test_mega_moe_backward.py](../tests/bf16/test_mega_moe_backward.py) | 公共前后向链路、四梯度及计数的reference对照；其[启动辅助](../tests/bf16/device_support.py)使用StoreGroup |
| [bf16_lifecycle.py](../tests/integration/bf16_lifecycle.py) | ProcessGroup/gloo协调下的重复调用、排队前向、流约束、布局和关闭重建；重复结果对照不替代独立golden |

测试文件存在只说明有验证入口；设备通过范围必须以相应源码、完整wheel、环境及实际执行结果为依据。框架接入还应覆盖所用ProcessGroup后端、microbatch/重计算顺序、梯度累积和关闭时机，不能把单节点小shape的精度通过等同于这些场景已经完成验证。
