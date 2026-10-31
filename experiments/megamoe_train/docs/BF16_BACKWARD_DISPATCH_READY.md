# BF16反向Dispatch细分与配套分核

本方案在已有SwiGLU/Gather稳定优化之上，提前交付dO行块供dHp计算。前向逻辑、保存模式、梯度数学和舍入顺序保持不变。

## 默认与回退

| 条件 | Dispatch→dHp交付 |
| --- | --- |
| `save=bf16/mxfp8`，每卡专家数≤7且有效专家行数大于一个M128块 | 默认全部非空专家启用256行进度及配套分核 |
| `save=bf16/mxfp8`，有效专家行数不超过128 | 整专家ready、原`3_1`计算顺序 |
| `save=none` | 保持重算与原UB路径，不启用本细分 |

行数使用kernel中已获得且按接收容量截断后的实际值，不用均衡估算，不扫描peer。生产者与消费者调用同一个`UseBackwardDispatchRowReady()`。空专家沿原路径处理；非空单M专家仍消耗一个非空序号，不让后续专家顶替前k的位置。

正常使用无需额外参数。排障时可设置`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=0/1/.../all`：按原执行顺序选前k个非空专家，之后仍应用单M保护。默认每卡专家数≤7时为`all`，≥8时为`0`。`0`关闭细分及配套分核、不申请进度suffix；saved模式的单行Dispatch继续保留。none强制按0处理。

## 执行流程

1. 原Dispatch producer分工不变，按实际source分段搬运；每256行（或分段尾部）在MTE3写完成后发布本地进度。
2. AIC按source发布阶段组织M块，用原3宽N条带计算dHp；不运行动态找ready任务、全peer扫描或排序。
3. 跨source组的M128尾块，其全部N任务放在本组内部任务之后，避免本核先等尾块而挡住本组可计算任务。最终专家partial M块无下一组依赖，不做此后置。
4. 每个tile继续检查其实际输入行区间。全部AIC退休dHp写入后，AIV沿原序消费；dW仍按完整专家归约，dO复用及阶段同步保留。

每组source数最多为AIC数；原有空组定位和有界metadata查询保留。EP64/128/256不会仅因EP大于AIC核数就关闭配套分核。当前元数据在原RestoreTokenPerExpert source循环内准备，复用原AIV同步，在既有dHp-ready cacheline的expert0 word1发布准备完成。AIC首次需要细分时等待此标志，后续不再逐组等待metadata arrival；实际数据ready不变。原15组支持上界检查保留，超出时保留原计算映射和真实行ready查询，不越界访问。

## 控制区与边界

新增控制区为`k × EP × P × 64B`，P为每source producer数；原payload与控制区前缀偏移不变。Host对控制区及suffix清零，kernel每次调用使用新进度。没有新增大块GM/UB payload。以每卡4专家、P=1为例，EP64/128/256分别增加16/32/64KiB。

本版本不包含查询跳过、N8、跨组N方向交替、row-wave、none-GM等额外候选。保留必要的发布/复用/整专家同步。

大EP目前只有离线功能与小EP强制多组精度覆盖；权重缓存、标量查询、通信争用及端到端性能仍需真实设备验收。不能从相同任务数推断相同访存量，也不宣称所有token分布都一定提速。

## 主要代码

- `layout/mega_moe_layout.hpp`：单M选择策略、进度区布局。
- `mega_moe_bf16_backward.hpp`：非空专家序号、单行搬运、进度发布。
- `schedule/backward_dispatch_ready.hpp`：实际行依赖查询，跳过空source。
- `schedule/backward_dispatch_schedule.hpp`及`schedule/backward_dispatch_schedule_math.hpp`：发布阶段映射、3宽N顺序、跨组尾块后置。
- `catlass/mega_moe_bf16_backward_catlass_impl.hpp`：dHp消费ready、保持完整专家交付。
