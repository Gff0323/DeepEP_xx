# BF16 SwiGLU反向重建与导数的FP32中间量对齐

## 计算契约

`R`表示舍入到BF16。前向一直使用：

```text
g = FP32(gate_BF16)，u = FP32(up_BF16)
按需clamp g、u
silu = g / (1 + exp(-g))             # FP32
H  = R(silu * u)
Hp = R(FP32(H) * FP32(p_BF16))
```

修改前，反向重建H却使用`R(R(silu)*u)`；dUp使用BF16舍入后的SiLU，dGate先把`dH*u`物化为BF16，再计算SiLU导数。这在复刻`F.silu(BF16 gate)*BF16 up`的独立小算子图，与当前前向融合FP32 SiLU/Up的中间量边界不同。

当前反向与前向采用一致的中间量边界：

```text
H  = R(silu * u)
Hp = R(FP32(H) * p)
dH = R(dHp * p)                     # 保留乘权后的BF16张量边界
s  = 1 / (1 + exp(-g))              # FP32
dUp   = R(FP32(dH) * silu * up_mask)
dGate = R((FP32(dH) * u) * (s * (1 + g * (1 - s))) * gate_mask)
```

中间`SiLU`及`dH*u`不再先BF16舍入再转回FP32；最终H、Hp、dH、dGate/dUp仍分别保留原来的BF16边界。这里均为BF16，不是FP16；两版的实际向量乘法本来都是FP32，变化是是否插入中间BF16舍入。

覆盖`ComputeForwardAux`（尾块H）、`ComputeFusedFullTile`（满tile H及导数）、`ComputeSwigluGradImpl`（尾块导数），并同步清理`ComputeSwigluRecompute`的同类旧舍入。后者当前没有调用点，保持其契约一致，未新增调用。

独立CPU参考`reference_packed_intermediates()`也改为前后共享同一H，并明确保留`dH=R(p*dHp)`的边界。高精度数学梯度参考仍独立保留，未放宽已有设备误差阈值。

## 保留的行为

- 前向代码、TopK早乘位置、GMM和Dispatch调度不变。
- GM/UB布局、workspace大小、Host ABI、ready协议、同步不变。
- clamp数值及导数掩码保持；边界点仍按原规则传递梯度，越界为零。
- dTopK逐元素乘积的BF16舍入、FP32 partial及归约顺序保持。
- saved-none、bf16、mxfp8共用该反向epilogue；MXFP8仍因Gate/Up压缩而有独立误差，不能称其H与未压缩前向逐位一致。
- 此变更不把完整算法改成后乘p的外部fused计算图，也不保证与其逐位一致。

## 为什么这样处理导数

导数按当前融合FP32子图计算，在既有BF16张量边界处收口，遵循通常的浮点Cast恒等梯度约定。这里不是对离散BF16舍入函数求严格数学导数，也不能对包含BF16台阶的整体函数直接做微小有限差分。

CPU回归对平滑的内部SiLU/Up子图做有限差分，并单独检查clamp掩码及BF16边界。`g=1, u=1.3125`回归例确认重建H使用`0.9609375`，而非旧路径的`0.95703125`。

同样取`g=1, u=1.3125, p=1`，`dHp=0.28125`时dUp由旧舍入路径的`0.205078125`变为`0.2060546875`；`dHp=0.40625`时dGate由`0.4921875`变为`0.494140625`。新值与平滑子图的标量导数舍入结果一致。这是CPU舍入回归例，不是网络误差统计或loss改善证明。
