# BF16前反向SwiGLU与路由搬运

## 前向：批量读权重与整块计算

`epilogue/block_epilogue_swiglu_forward_bf16.hpp::ComputeSwigluDirect()`将BF16 Gate/Up直接解包到FP32寄存器，以固定小块完成向量计算，按tile批量读取路由权重。计算使用准确Div和clamp，先将SwiGLU结果舍入BF16，再转回FP32乘p，最后舍入BF16。

普通BF16前向使用M192计算块，反向GEMM使用独立的M128计算块；通信fragment与计算tile的粒度分别定义。

## 前向：SwiGLU交接与容量

`DEEPEP_BF16_SWIGLU_DECOUPLE`控制普通BF16前向GMM1与SwiGLU之间的交接方式：

| 编译值 | 前向交接方式 |
| --- | --- |
| `0` | UB握手：AIC将一对Gate/Up tile写入UB，AIV0等待后完成SwiGLU及路由权重乘法，再通知AIC复用UB。 |
| `1` | GM解耦：AIC将原始BF16 Gate/Up写入GM，当前专家的所有AIC发布完成后，AIV0按tile将其读回UB并完成激活。 |
| `2` | auto：按hidden和实际保存模式选择UB握手或GM解耦。 |

两条路径都将加权后的SwiGLU结果写入GM供GMM2读取；GM解耦路径中的激活计算使用UB和寄存器。二者采用相同的clipping、准确Div及BF16舍入边界。

模块`CMakeLists.txt`显式传入`-DDEEPEP_BF16_SWIGLU_DECOUPLE=2`。共享头文件未定义该宏时使用`0`，用于没有设备编译定义的Host/CPU等翻译单元；设备SO的默认值为`2`。

`layout/mega_moe_layout.hpp::ForwardUsesGmSwiglu(hidden,savedMode)`定义auto策略：

| hidden | none / bf16 | mxfp8 |
| --- | --- | --- |
| 小于3584 | GM解耦 | GM解耦 |
| 等于3584 | UB握手 | GM解耦 |
| 大于3584 | UB握手 | UB握手 |

savedMode须先经`ResolveSavedGateUpMode()`解析。公共保存模式`auto`在clipping关闭时解析为`mxfp8`，开启时解析为`bf16`，与交接策略的编译值`2`是独立选项。例如H3584、保存模式auto且clipping关闭时，按mxfp8选择GM解耦。

Host分配前向临时空间前，先解析保存模式，再通过`launcher.cpp::MegaMoeBF16SwigluDecoupled()`调用设备SO的`deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode(hidden,savedMode)`。设备查询使用其实际构建策略，Host据此调用`ForwardSwigluExtraBytes()`计算容量，不使用Host头文件的宏回退值推断设备路径。

UB握手路径无需原始Gate/Up临时GM。GM解耦路径中，`bf16`复用原始Gate/Up持久平面；`none/mxfp8`将Hp空间临时用作原始Gate，并在控制区之后追加原始Up平面，额外容量为`AlignUp(max_recv_tokens * (n / 2) * sizeof(bfloat16_t), 512)`，其中`n=2F`。

设备SO同时提供仅含hidden的查询和带savedMode的查询。前者按保存模式策略的并集返回结果，Host直接依赖后者，不执行查询回退。改变编译选项必须完整重建、重装同一次构建的Host/Device wheel，不能只替换一个SO或混装产物。

该宏只控制前向交接。反向`BackwardUsesGm(savedMode)`按已解析保存模式选择路径：`none`使用UB，`bf16/mxfp8`使用GM。

## Dispatch整专家回退

`DEEPEP_BF16_FORCE_EXPERT_READY=ON/OFF`默认`OFF`，wheel构建可设置同名环境变量，CMake可设置同名选项。Host/Device一致获得C++值`1/0`。`OFF`按条件选择前向Ready384和反向saved模式行进度；`ON`不启用前向tile-ready或配套元数据，并在Host/Device均使用反向`splitExperts=0`，等待整专家Dispatch完成flag。

该选项优先于运行时`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS`，与SwiGLU交接、保存格式和数值边界独立。更改选项需完整重建、重装同一次构建的Host/Device wheel，所有EP rank必须使用同一构建。详细选择条件见[Dispatch就绪策略](BF16_DISPATCH_READY_DEFAULTS.md)。

saved-state中的1KiB预留区用于固定后续Gate/Up、scale和clipping mask偏移，不执行诊断写入。该区与算法使用的Dispatch行进度后缀是不同内存区域。

## InitRouting与路由权重搬运

`moe_v3_gather_out.h`的普通BF16 X Gather按4/2行成组读入，同组同步，连续目标一次写出。双缓冲槽和尾组分别受MTE完成事件保护；UB容量或DMA长度不满足时使用单行路径。

`routing/mega_moe_topk_weight_gather.hpp`对65537～131072条路由使用两个固定65536窗口。每次先生成安全的窗口内偏移，再Gather，并以原始BF16位选择对应窗口结果；短第二窗只读有效GM字节，不先执行越界Gather再掩码。≤65536使用单窗路径，超过131072使用回退路径；UDMA/MGroup默认不启用两窗。临时UB约152KiB，在Dispatch分配UB前释放，无需额外GM或运行时窗口搜索。

## 反向：满tile计算与数值边界

`epilogue/block_epilogue_swiglu_backward_bf16.hpp::RunFusedFullTile()`在固定寄存器小块内读取Gate/Up/dHp，计算Hp、dTopK局部乘积及dZ，并按tile预取路由权重。只有128×256满tile使用此路径，尾块使用对应尾块实现。

反向使用准确Div、clamp导数及MXFP8裁剪掩码；最终BF16张量边界和dTopK四组64元素的归约顺序固定。SiLU及dH×up内部使用FP32中间量，详见[数值契约](BF16_SWIGLU_NUMERICS.md)。GM/UB路径采用双AIV分工、ready发布及等待协议；`none`使用UB握手，`bf16/mxfp8`使用GM解耦。

`mega_moe_bf16_backward.hpp::GatherRoutes()`在P1/P2三个工作tile上方的固定剩余UB中，以两个槽组织dY搬运：K3为4行，H7168为2行，容量不足时为1行。搬运原始dY，不提前乘p，数据复用受跨核及跨卡同步保护。

## 反向Dispatch

符合条件的非空专家使用行进度、配套分核、组内3宽和跨组尾块后置；单M专家等待整专家ready，`none`使用重算路径。详见[反向Dispatch行进度与配套分核](BF16_BACKWARD_DISPATCH_READY.md)。设备功能及性能须按实际shape、路由和配套环境独立验证。
