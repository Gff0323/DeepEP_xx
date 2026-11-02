# BF16前反向Dispatch就绪策略

普通BF16的默认构建策略为`adaptive-auto`。前向使用M192计算块、Ready384行进度及静态producer元数据；反向saved路径使用M128计算块、256行进度及静态元数据。DMA每次搬运的行数与消费者等待行进度或整专家完成是两个独立选择。

## 统一整专家回退选项

`DEEPEP_BF16_FORCE_EXPERT_READY`是前反向Dispatch就绪回退构建选项，默认`OFF`。wheel构建可设置同名环境变量，直接CMake构建可传`-DDEEPEP_BF16_FORCE_EXPERT_READY=ON/OFF`；构建系统向Host和Device一致传入C++值`0/1`。

| 构建值 | 前向 | 反向 |
| --- | --- | --- |
| `OFF`（默认） | 按下文条件使用M192／Ready384和静态producer元数据 | 按保存模式、专家数和实际行数选择256行进度或整专家交付 |
| `ON` | 不启用tile-ready和配套元数据，等待完整专家Dispatch完成flag | Host与Device均使用`splitExperts=0`，不分配行进度后缀或准备配套元数据，等待完整专家完成flag |

`ON`优先于`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS`，运行时覆盖不能重新开启细分。只有默认`OFF`构建使用专家数选择及显式排障覆盖。整专家回退不控制SwiGLU的UB／GM交接；`DEEPEP_BF16_SWIGLU_DECOUPLE`独立使用`0/1/2`，设备构建默认`2`。

更改构建选项后必须完整重建、重装同一次构建的Host/Device wheel，所有EP rank必须使用同一构建。不要只替换设备SO或混用默认与整专家构建。两种构建使用相同的公开API及设备库名`libdeepep_megamoe_train_bf16.so`；saved-state格式由保存模式定义。以下细分说明均针对`OFF`。

## 前向

静态元数据容量可用时，**所有符合条件的非空专家**分别启用细分。形状条件为每卡专家数≤7、`M × TopK ≥ 每卡专家数 × 768`，并要求该专家按接收容量截断后的实际行数≥768。`none/bf16/mxfp8`共用此选择；SwiGLU交接的hidden阈值属于独立策略。

静态元数据在`GetCumsumForMMAIV()`的source前缀循环内生成，通过pre-Dispatch同步交付。计算侧按producer的实际发布阶段组织M块，AIC与消费GMM1结果的AIV使用相同坐标。消费者无需等待payload相关的metadata arrival，但必须检查实际行ready或整专家完成flag。

静态元数据路径受AIC数（28／32／36）、EP≤256及GM空闲word、UB和staging容量约束。容量不足时，只有首个非空专家进入tile-ready选择，其余专家等待整专家。专家行数按接收容量截断；不能使用静态分段映射时，计算侧使用回退映射并继续检查真实行ready。行数不足768的专家等待整专家。

```text
静态元数据容量可用且满足形状条件：
    每个非空专家独立判断：实际行数>=768用Ready384，否则等待整专家
静态元数据容量不可用且满足形状条件：
    仅首个非空专家进入tile-ready选择，其余等待整专家
不满足形状条件：
    全部等待整专家
```

## 反向

| 保存模式与实际行数 | 默认交付 |
| --- | --- |
| `bf16/mxfp8`，每卡专家数≤7，专家实际有效行数>128 | 全部符合条件的非空专家细分，使用256行进度及N3配套分核 |
| `bf16/mxfp8`，每卡专家数≥8 | 整专家ready，使用`3_1`分核，不分配细分进度后缀 |
| `bf16/mxfp8`，专家实际有效行数≤128 | 整专家ready，应用单M保护 |
| `none` | GMM1重算与UB流程，不启用行进度细分 |

反向元数据在`RestoreTokenPerExpert()`的source循环内分散准备，通过AIV同步交付。AIC首次使用细分时等待一次元数据准备完成标志；数据读取仍检查真实dO行ready，dW按完整专家归约，dO复用受阶段同步保护。

排障可设置`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS=0/1/.../all`，按执行顺序选择前k个非空专家，再逐专家应用单M保护。默认每卡专家数≤7时为`all`，≥8时为`0`；显式覆盖可在≥8专家时选择`all`。`none`始终使用`0`。该专家数边界与前向共用常量，实际行数门槛分别定义。

反向细分使用M128计算块、256行producer进度、固定N3和跨组尾块后置；整专家路径使用`3_1`。默认≥8专家及强制整专家构建不追加细分控制区。

## Saved-state预留区

saved-state包含`256 × int32`的1KiB预留区，用于固定后续Gate/Up、scale和clipping mask偏移。Host和kernel不对该区执行诊断写入或逐次清零。它与`BackwardDispatchProgressLayout`的行进度后缀是不同内存区域；后者用于算法同步，必须按当前就绪策略正确初始化、发布和等待。

## 支持边界

元数据容量上界和可寻址EP范围不等于通过设备验证的部署范围。更大EP、跨节点及不同路由分布需要独立设备验证；离线调度覆盖不能替代设备功能、性能或训练收敛验收。
