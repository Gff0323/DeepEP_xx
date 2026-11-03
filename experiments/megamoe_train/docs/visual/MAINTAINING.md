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

修改 `data/*.json` 中对应节点的标题、讲解、输入、输出、等待和连线。`sources` 使用文件路径与源码锚点定位函数或控制点；函数移动或改名时同步修改锚点。节点 ID 保持稳定，便于页面链接和跨图下钻。

训练代码更新后：

1. 在 `feat/megamoe-train-bf16` 中同时维护训练代码与本目录文档。
2. 核对受影响节点，更新相应的执行流程、张量和同步说明。
3. 运行构建，重新采集源码片段、位置和版本信息。
4. 完成源码引用与浏览器检查，提交修改后的文档、图源和生成文件。

浏览器检查的截图和报告写入 `.artifacts/`；`node_modules/`、`.artifacts/` 和发布暂存目录 `site/` 是本地生成目录。网页阅读和文档构建不需要 NPU 或 CANN。

## 自动发布

工作流位于仓库根目录的 `.github/workflows/megamoe-train-visual.yml`。向 `feat/megamoe-train-bf16` 推送相关修改后，自动执行依赖安装、生成、源码检查与浏览器检查，通过后发布 `site/`。也可从 Actions 页手动选择该分支运行。

PR 执行文档检查，合入发布分支后再部署。检查失败时，Actions 保存 `.artifacts/` 中已有的报告和截图，供查看问题。

首次启用时，仓库维护者在 **Settings → Pages → Source** 选择 **GitHub Actions**，并让 `github-pages` 环境允许 `feat/megamoe-train-bf16` 部署。工作流通过当前仓库上下文生成和发布，未绑定个人账号。

文档正式合入上游后，将工作流的 `push.branches` 和两处发布条件中的分支名一起改为正式发布分支，并同步 `github-pages` 环境的部署分支设置。文档内容及生成脚本继续使用同一套。

源码行为变化时，需要一起审阅图的讲解和连线；重新生成会刷新引用内容，执行语义由代码与文档 review 对齐。
