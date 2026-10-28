> 源 b8cff148 的算法/历史说明；本次 GitHub 迁移验证以 [首轮迁移验收](../首轮迁移验收.md) 为准。

# BF16前反向SwiGLU与路由搬运优化

> 本文从 `train@6d8f833` 随算法同步。原实验数据属于 train 的历史验证，不能代替本开源分支的验证；本次同步范围与验证见 [2026-10-28 同步记录](BF16_TRAIN_SYNC_20261028.md)。

本分支基于train的92626f8，仅整理普通BF16前后向已验证候选。正式train和独立实验目录保持不变。组合验证结果另存于原始源版本验证附件（未随本模块发布）及Profiling/0-analysis下的原报告；本文解释代码契约。

## 前向：批量读权重、直接解包与整块计算

`block_epilogue_swiglu_bf16.h::ComputeSwigluDirect()`把BF16 Gate/Up直接解包到FP32寄存器，以固定小块完成向量计算，按tile批量读路由权重。原来每行读一个p再同步的管理开销减少。仍保留准确Div、clamp，以及SwiGLU结果先舍入BF16、转回FP32乘p、最后舍入BF16的顺序。不是把数学公式换成低精度近似。

普通BF16使用既有`DEEPEP_BF16_M192_FORWARD`编译范围；共享头文件中的UDMA/MGroup调用保持原路径。

## 前向：解耦选择与容量必须一致

`mega_moe_training_workspace.h::ForwardUsesGmSwiglu(hidden,savedMode)`：

| hidden | none / bf16 | mxfp8 |
| --- | --- | --- |
| 小于3584 | 解耦 | 解耦 |
| 等于3584 | 不解耦 | 解耦 |
| 大于3584 | 不解耦 | 不解耦 |

已有构建选项off/on仍可强制选择，auto采用上表。Host调用新的可选查询`deepep_train_mega_moe_bf16_swiglu_decoupled_for_saved_mode`，按同一个savedMode计算所需临时空间，kernel也用同一函数选择流程。

兼容旧SO时Host回退到旧hidden-only查询。新SO保留旧查询并返回各保存模式所需容量的并集，因此旧Host最多多分配，不会因H3584/mxfp8低估空间。没有改kernel launch参数、已有workspace偏移或saved-state格式。非解耦路径不增加Gate/Up临时GM。

## InitRouting：多行成组搬运

`moe_v3_gather_out.h`的普通BF16 X Gather使用4/2行成组读入，按组同步，连续目标一次写出。双缓冲槽和尾组仍分别受MTE完成事件保护；UB容量或DMA长度不满足时回退原路径。

`mega_moe_topk_weight_gather.h`对65537～131072条路由提供固定两个65536窗口。每次先生成安全的窗口内偏移，再Gather，以原始BF16位选择对应窗口结果。短第二窗只读有效GM字节。不会把越界地址先Gather后再掩码。≤65536保留原快路径，超过131072回退；UDMA/MGroup默认不启用两窗。临时UB约152KiB，在Dispatch分配UB之前释放，没有新增GM或运行态遍历窗口搜索。

## 反向：满tile单遍计算，保留数值语义

`block_epilogue_swiglu_backward_bf16.h::RunFusedFullTile()`在固定寄存器小块内读入Gate/Up/dHp，完成Hp、dTopK局部乘积与dZ所需运算。路由权重按tile预取，减少逐行同步和两遍重新读Gate/Up。仅128×256满tile走此路径，尾块保留原正确路径。

保留准确Div、clamp导数及MXFP8剪裁掩码、最终BF16张量边界和dTopK四组64元素的原归约顺序。SiLU及dH×up内部采用FP32中间量，详见[数值契约](BF16_SWIGLU_NUMERICS.md)。原来的GM/UB布局、双AIV分工、ready发布和等待不变。`none`仍走UB握手，`bf16/mxfp8`仍走已有GM解耦；不带旧none-GM实验、实验控制尾部或额外完整GM缓冲。

`mega_moe_bf16_backward.h::GatherRoutes()`在P1/P2三个工作tile上方的固定剩余UB中分两槽组织dY搬运：K3为4行，H7168为2行，容量不足回退1行。仍读取原始dY，不提前乘p；所有原有跨核/跨卡同步保持。

## 验收边界

先编译正式Host与组合kernel，运行布局/容量/调度测试，再做各保存模式CPU梯度golden、尾块、空专家、旧权重布局与无重复TopK的不均衡用例。大shape逐位比较Y、dX、dTopK、dW1、dW2及计数、finite；与基线一致不等于具备独立大shape梯度golden。

性能以无Timer完整forward/backward Event为准，分别测量；每rank50次p50是中位数，主指标为各rank p50最大值。基线/组合反序复测，不能相加独立候选收益，也不能把分别测得的前后向时间之和称为实测训练step。


## 反向Dispatch稳定方案

在上述优化基础上接入全部专家细分、配套分核、组内3宽和跨组尾块后置；单M专家回退整专家ready，none保持原路径。详见[BF16反向Dispatch细分与配套分核](BF16_BACKWARD_DISPATCH_READY.md)。设备与性能验收记录保存在工作区外。
