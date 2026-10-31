# BF16前反向SwiGLU与路由搬运优化

## 前向：批量读权重、直接解包与整块计算

`epilogue/block_epilogue_swiglu_forward_bf16.hpp::ComputeSwigluDirect()`把BF16 Gate/Up直接解包到FP32寄存器，以固定小块完成向量计算，按tile批量读路由权重。原来每行读一个p再同步的管理开销减少。仍保留准确Div、clamp，以及SwiGLU结果先舍入BF16、转回FP32乘p、最后舍入BF16的顺序。不是把数学公式换成低精度近似。

普通BF16使用既有`DEEPEP_BF16_M192_FORWARD`编译范围；共享头文件中的UDMA/MGroup调用保持原路径。

## 前向：解耦选择与容量必须一致

`layout/mega_moe_layout.hpp::ForwardUsesGmSwiglu(hidden,savedMode)`：

| hidden | none / bf16 | mxfp8 |
| --- | --- | --- |
| 小于3584 | 解耦 | 解耦 |
| 等于3584 | 不解耦 | 解耦 |
| 大于3584 | 不解耦 | 不解耦 |

已有构建选项off/on仍可强制选择，auto采用上表。Host调用新的可选查询`deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode`，按同一个savedMode计算所需临时空间，kernel也用同一函数选择流程。

兼容旧SO时Host回退到旧hidden-only查询。新SO保留旧查询并返回各保存模式所需容量的并集，因此旧Host最多多分配，不会因H3584/mxfp8低估空间。没有改kernel launch参数、已有workspace偏移或saved-state格式。非解耦路径不增加Gate/Up临时GM。

## InitRouting：多行成组搬运

`moe_v3_gather_out.h`的普通BF16 X Gather使用4/2行成组读入，按组同步，连续目标一次写出。双缓冲槽和尾组仍分别受MTE完成事件保护；UB容量或DMA长度不满足时回退原路径。

`routing/mega_moe_topk_weight_gather.hpp`对65537～131072条路由提供固定两个65536窗口。每次先生成安全的窗口内偏移，再Gather，以原始BF16位选择对应窗口结果。短第二窗只读有效GM字节。不会把越界地址先Gather后再掩码。≤65536保留原快路径，超过131072回退；UDMA/MGroup默认不启用两窗。临时UB约152KiB，在Dispatch分配UB之前释放，没有新增GM或运行态遍历窗口搜索。

## 反向：满tile单遍计算，保留数值语义

`epilogue/block_epilogue_swiglu_backward_bf16.hpp::RunFusedFullTile()`在固定寄存器小块内读入Gate/Up/dHp，完成Hp、dTopK局部乘积与dZ所需运算。路由权重按tile预取，减少逐行同步和两遍重新读Gate/Up。仅128×256满tile走此路径，尾块保留原正确路径。

保留准确Div、clamp导数及MXFP8剪裁掩码、最终BF16张量边界和dTopK四组64元素的原归约顺序。SiLU及dH×up内部采用FP32中间量，详见[数值契约](BF16_SWIGLU_NUMERICS.md)。原来的GM/UB布局、双AIV分工、ready发布和等待不变。`none`仍走UB握手，`bf16/mxfp8`仍走已有GM解耦；不带旧none-GM实验、实验控制尾部或额外完整GM缓冲。

`mega_moe_bf16_backward.hpp::GatherRoutes()`在P1/P2三个工作tile上方的固定剩余UB中分两槽组织dY搬运：K3为4行，H7168为2行，容量不足回退1行。仍读取原始dY，不提前乘p；所有原有跨核/跨卡同步保持。

## 反向Dispatch稳定方案

在上述优化基础上接入全部专家细分、配套分核、组内3宽和跨组尾块后置；单M专家回退整专家ready，none保持原路径。详见[BF16反向Dispatch细分与配套分核](BF16_BACKWARD_DISPATCH_READY.md)。设备性能需要按实际配套、shape 和路由独立测量。
