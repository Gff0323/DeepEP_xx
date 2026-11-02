# BF16反向Dispatch行进度与配套分核

Dispatch producer按source搬运dO并发布已完成行数，dHp计算侧据此消费可读行块。行进度交付与静态任务映射配合，使计算侧能够消费已到达的数据，而不必等待整个专家的Dispatch完成。

## 构建与回退

本页描述`DEEPEP_BF16_FORCE_EXPERT_READY=OFF`的默认行为。通过同名wheel构建环境变量或CMake选项设为`ON`时，Host/Device一致使用C++值`1`（`OFF`为`0`），反向均使用`splitExperts=0`，不分配行进度后缀或准备配套元数据，等待整专家Dispatch完成flag；前向同时不启用tile-ready。该选项优先于`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS`。

更改选项需完整重建、重装同一次构建的Host/Device wheel，所有EP rank必须使用同一构建。SwiGLU交接由独立的`DEEPEP_BF16_SWIGLU_DECOUPLE=0/1/2`控制，设备默认`2`，其作用范围仅为前向。

## 默认交付条件

| 条件 | Dispatch→dHp交付 |
| --- | --- |
| `save=bf16/mxfp8`，每卡专家数≤7且有效专家行数>128 | 全部符合条件的非空专家使用256行进度及配套分核 |
| `save=bf16/mxfp8`，每卡专家数≥8，或专家有效行数≤128 | 整专家ready及`3_1`计算顺序 |
| `save=none` | GMM1重算与UB路径，不启用行进度细分 |

生产者与消费者调用同一个`UseBackwardDispatchRowReady()`，行数使用kernel已获得且按接收容量截断后的实际值，不用均衡估算或额外扫描peer。空专家按空输入处理；非空单M专家也计入非空序号，后续专家不能顶替前k的位置。

排障时可设置`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=0/1/.../all`，按执行顺序选择前k个非空专家，再应用单M保护。默认每卡专家数≤7时为`all`，≥8时为`0`。`0`不启用细分或配套分核，也不申请进度suffix；saved模式的单行DMA搬运独立于该选择。`none`始终使用`0`。

## 执行流程

1. Dispatch producer按实际source分段搬运；每256行或分段尾部，在MTE3写完成后发布本地进度。
2. AIC按source发布阶段组织M128块，用3宽N条带计算dHp，不执行动态ready任务搜索、全peer扫描或排序。
3. 跨source组的M128尾块，其全部N任务排在本组内部任务之后，避免本核等待尾块时阻挡可计算任务。最终专家partial M块无下一组依赖，无需此后置。
4. 每个tile检查实际输入行区间。全部AIC退休该专家的dHp写入后，AIV按调度顺序消费；dW按完整专家归约，dO复用受阶段同步保护。

每组source数最多为AIC数，支持空组定位和有界metadata查询；EP大于AIC数本身不会关闭配套分核。元数据在`RestoreTokenPerExpert()`的source循环内准备，通过AIV同步交付，并在dHp-ready cacheline的expert0 word1发布准备完成标志。AIC首次需要细分时等待此标志，随后按实际行进度等待数据。

静态任务映射受15组上界约束。超出时使用回退计算映射及真实行ready查询，不越界访问元数据。

## 控制区与生命周期

行进度控制区大小为`k × EP × P × 64B`，其中k为选择细分的专家数，P为每source的producer数。该区位于workspace后缀，payload和控制区前缀具有固定偏移。Host初始化控制区及suffix，kernel每次调用使用本次进度。以每卡4专家、P=1为例，EP64/128/256的控制区分别为16/32/64KiB。

producer只能发布已完成写入的行数，consumer据此判断数据是否可读。该同步协议使用的`BackwardDispatchProgressLayout`后缀与saved-state中的1KiB预留区彼此独立：预留区固定Gate/Up等后续偏移，不承担行进度或诊断写入。

## 支持边界

算法容量及离线调度覆盖不能替代设备验证。大EP、跨节点、不同路由分布、权重缓存和通信争用条件下的功能与端到端性能需要独立验收；不能从相同任务数推断相同访存量或所有token分布都有性能收益。

## 主要代码

- `layout/mega_moe_layout.hpp`：单M选择策略、进度区布局。
- `mega_moe_bf16_backward.hpp`：非空专家序号、DMA搬运、进度发布。
- `schedule/backward_dispatch_ready.hpp`：实际行依赖查询、空source处理。
- `schedule/backward_dispatch_schedule.hpp`及`schedule/backward_dispatch_schedule_math.hpp`：发布阶段映射、3宽N顺序、跨组尾块后置。
- `catlass/mega_moe_bf16_backward_catlass_impl.hpp`：dHp消费ready及完整专家交付。
