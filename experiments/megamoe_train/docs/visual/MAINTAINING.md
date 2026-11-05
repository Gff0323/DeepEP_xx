# 维护源码导览

本目录直接在 `feat/megamoe-train-bf16` 分支随 MegaMoE BF16 训练代码一起维护。图中的文字、节点和连线描述源码行为；生成工具负责更新源码片段、位置与网页。默认阅读场景为 `save=bf16`、clipping 关闭。

## 本地修改与检查

在仓库根目录进入本目录，使用 Node.js 22 或更新版本：

```bash
cd experiments/megamoe_train/docs/visual
npm ci
npm run build
npm run check
npx playwright install chromium
npm run test:browser
```

阅读时直接打开 [index.html](index.html)，无需启动服务。仓库网页入口是 [README.md](README.md)，保留图文说明和源码链接。

在 VS Code 中推荐使用[源码导览扩展](vscode/README.md)。安装一次后，运行 **MegaMoE: 打开训练源码导览**。点击节点、阶段或等待只查看讲解；点击 **打开源码** 或 **查看对应源码**，再在旁边的编辑器中定位到文件和行号。扩展直接读取当前工作区的页面与映射。

本地构建安装包的命令：

```bash
python tools/package-editor.py
code --install-extension .artifacts/megamoe-train-source-guide-0.2.0.vsix
```

普通浏览器继续直接打开离线 HTML；在线页面保留 **复制定位路径**。使用 VS Code 自带内置浏览器时，也可以复制后按 **Ctrl+P** 粘贴并回车。这里与扩展提供的源码导览是两个不同的打开方式。

修改 `data/*.json` 中对应节点的标题、讲解、输入、输出、等待和连线。`sources` 使用文件路径与源码锚点定位函数或控制点；函数移动或改名时同步修改锚点。节点 ID 保持稳定，便于页面链接和跨图下钻。

三个阅读场景由 `data/learning-case.json` 与 `tools/learning-scenes.mjs` 生成。默认 EP8、每源 M4096、TopK6、每 rank 4 个专家，流水总览固定看 rank0，按专家展示。`assets/full-pipeline.js` 维护前反向总览，`assets/gmm2-pipeline.js` 维护独立双槽练习。输入规则、聚合计数和所选 token 的数据共同驱动路由图、保存桥和流水示意；修改案例后同时检查路由守恒和数值结果。时间轴 tick 表示可调的教学耗时，实测时间及性能记录单独维护。

H、dH 和 dp 乘积保持 FP32；Hp、dGate/dUp 写出时转回 BF16，Y、dX、dp 在 FP32 归约后最终舍入。修改数值边界时，同时更新 L5 节点、教学计算和表格；流水详情从 L5 读取同一份公式。

[数值契约](../mega_moe_bf16_design.md#13-数学关系与舍入边界)也用于 NPU BigOp 与契约参考，CPU FP64 真值独立计算。修改测试参考时，核对导览中的公式和源码引用；原始设备报告与临时验证脚本继续存放在仓库外。

浏览器检查覆盖仓内 HTTP 预览、`file://` 离线页面、VS Code 消息桥，以及 `site/` 发布产物的 HTTP 模式。打包站点后仍使用同一份页面和源码映射。

训练代码更新后：

1. 在 `feat/megamoe-train-bf16` 中同时维护训练代码与本目录文档。
2. 核对受影响节点，更新相应的执行流程、张量和同步说明。
3. 运行构建，重新采集源码片段、位置和版本信息。
4. 完成源码引用与浏览器检查，提交修改后的文档、图源和生成文件。

浏览器检查的截图和报告写入 `.artifacts/`；`node_modules/`、`.artifacts/` 和发布暂存目录 `site/` 是本地生成目录。网页阅读和文档构建不需要 NPU 或 CANN。

修改编辑器接入时，运行 `npm run test:editor` 检查消息、源码路径和当前文档锚点定位，再执行浏览器检查。`python tools/package-editor.py --site` 同时把 VSIX 放入站点下载目录。扩展代码更新后需重新打包安装；只更新训练代码、图或讲解时无需重装扩展，重新打开导览即可。

## 自动发布

工作流位于仓库根目录的 `.github/workflows/megamoe-train-visual.yml`。向 `feat/megamoe-train-bf16` 推送相关修改后，自动执行依赖安装、生成、源码检查与浏览器检查，通过后发布 `site/`。也可从 Actions 页手动选择该分支运行。

PR 执行文档检查，并在 Actions 附件中提供 `megamoe-train-visual-preview`，包含站点页面和 VS Code 安装包。下载解压后，可在该目录运行 `python -m http.server 8000`，从浏览器打开 `http://localhost:8000` 预览。正式站点在合入发布分支后部署。检查失败时，Actions 保存 `.artifacts/` 中已有的报告和截图，供查看问题。

首次启用时，仓库维护者在 **Settings → Pages → Source** 选择 **GitHub Actions**，并让 `github-pages` 环境允许 `feat/megamoe-train-bf16` 部署。工作流通过当前仓库上下文生成和发布，未绑定个人账号。

文档正式合入上游后，将工作流的 `push.branches` 和两处发布条件中的分支名一起改为正式发布分支，并同步 `github-pages` 环境的部署分支设置。文档内容及生成脚本继续使用同一套。

源码行为变化时，需要一起审阅图的讲解和连线；重新生成会刷新引用内容，执行语义由代码与文档 review 对齐。
