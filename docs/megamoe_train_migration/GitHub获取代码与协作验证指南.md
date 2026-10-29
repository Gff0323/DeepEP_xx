# MegaMoE 训练：GitHub 获取代码与协作验证指南

更新：2026-10-29。面向参与验证的同事，重点说明如何拿到正确代码、更新版本和提交反馈。

**GitHub 保存代码和讨论记录；编译、精度和性能测试在 Ascend950 Linux 服务器上执行。**
上次验证也是先把固定版本代码及依赖放到服务器的独立目录，再构建安装、运行测试，最后把脱敏结果提交到 GitHub。

**蓝区服务器不能访问 GitHub 时，按第 4 节离线传输，不需要让 NPU 服务器直接登录 GitHub。**

## 1. 用哪个仓、哪个分支

| 项目 | 当前使用位置 |
| --- | --- |
| 协作仓 | [xixing-321/DeepEP](https://github.com/xixing-321/DeepEP) |
| 训练开发分支 | [feat/megamoe-train-bf16](https://github.com/xixing-321/DeepEP/tree/feat/megamoe-train-bf16) |
| 训练代码 | [experiments/megamoe_train](https://github.com/xixing-321/DeepEP/tree/feat/megamoe-train-bf16/experiments/megamoe_train) |
| 分工与验证记录 | [docs/megamoe_train_migration](https://github.com/xixing-321/DeepEP/tree/feat/megamoe-train-bf16/docs/megamoe_train_migration) |

**当前训练代码还没合入 main。不要只下载默认 main 分支来验证。**
我们先在这个公开协作仓中复核；正式提交给 `lueelu/DeepEP` 是后续步骤。
同事以自己的 GitHub 账号 `Gff0323` 协作，不需要再 fork 一个仓。

## 2. 先在 GitHub 网页上看

直接打开上面的训练分支链接，即可浏览代码。点开文件后可查看内容和历史修改；
打开协作资料目录，先看 [实施与分工方案](实施与分工方案.md) 和 [首轮迁移验收](首轮迁移验收.md)。

网页主要用于看代码、看文档和审查修改。实际测试需要在服务器上取得代码。
建议使用 Git 克隆，保留版本历史并方便更新；ZIP 下载不适合这次持续协作，也不包含完整子模块源码。

## 3. 第一次把代码拿到 NPU 服务器

登录分配给自己的 Linux 测试环境，在自己的独立工作目录执行：

```bash
git clone --branch feat/megamoe-train-bf16 https://github.com/xixing-321/DeepEP.git DeepEP-train
cd DeepEP-train
git submodule update --init third-party/shmem third-party/catlass

git branch --show-current
git rev-parse HEAD
git remote -v
```

应看到分支 `feat/megamoe-train-bf16`，`origin` 指向 `xixing-321/DeepEP`。
子模块命令会获取固定版本 SHMEM、CATLASS，相关依赖还需要访问 GitCode。

这个仓是公开仓，HTTPS 克隆和拉取本仓不需要 GitHub 登录。推送修改才需要自己的 GitHub 写权限及 Git 认证。
网页上已登录，不代表服务器上的 Git 已认证；服务器推送时应配置自己的 SSH key 或 HTTPS 凭据，不共享账号或令牌。

若服务器不能访问 GitHub/GitCode，使用下面的离线方式。BF16 模块只需要这里指定的两个依赖，
不必递归下载 CATLASS 的 Python DSL、编译器等无关子项目。

## 4. 蓝区无法访问 GitHub 时，怎么做

```text
可联网电脑：从 GitHub 获取训练分支、从 GitCode 获取依赖
    → 打包传入蓝区
蓝区 NPU 服务器：本地恢复代码、编译和验证
    → 把结果或修改带回可联网电脑
可联网电脑：提交结果或修改到 GitHub，发起 PR
```

2026-10-29 对指定蓝区主机做了只读检查：SSH 可用，GitHub HTTPS 请求 12 秒超时，
`git ls-remote` 18 秒超时，GitCode HTTPS 请求也未成功。该结论只描述检查时的主机环境；
尚未收到 GitHub 认证响应，不能把超时归因于仓库写权限。没有改网络配置或占用 NPU。

上次实际交付使用源码压缩包、依赖 Git bundle 和配套产物。为了让同事保留 Git 历史、方便后续改代码，
推荐这次用 **Git bundle（可离线克隆的 Git 文件）** 传递主仓和两个依赖。

**联网侧：** 完成第 3 节克隆及依赖获取后，在仓库目录执行下面的 Bash 命令（Linux 或 Windows Git Bash）：

```bash
mkdir ../deepep-offline
PACK="$(cd ../deepep-offline && pwd)"
git bundle create "$PACK/deepep.bundle" feat/megamoe-train-bf16
git -C third-party/shmem bundle create "$PACK/shmem.bundle" HEAD
git -C third-party/catlass bundle create "$PACK/catlass.bundle" HEAD
git rev-parse feat/megamoe-train-bf16 > "$PACK/commit.txt"
```

三个 bundle 和 `commit.txt` 一起通过现有文件传输渠道送到蓝区。使用新目录存放，避免覆盖上一轮测试。
bundle 只包含已提交代码；本地未提交修改不会被带入。

**蓝区侧：** 假设文件已放到 `/你的目录/deepep-offline`，在自己的工作目录执行：

```bash
PACK=/你的目录/deepep-offline
git clone --branch feat/megamoe-train-bf16 "$PACK/deepep.bundle" DeepEP-train
cd DeepEP-train
for dep in shmem catlass; do
    git init "third-party/$dep"
    git -C "third-party/$dep" fetch "$PACK/$dep.bundle" HEAD
    git -C "third-party/$dep" checkout --detach FETCH_HEAD
    expected="$(git rev-parse "HEAD:third-party/$dep")"
    test "$(git -C "third-party/$dep" rev-parse HEAD)" = "$expected" || exit 1
done
git rev-parse HEAD
cat "$PACK/commit.txt"
```

最后两个提交号应一致。这些操作读取本地 bundle，不访问 GitHub/GitCode；无需再执行联网的 `submodule update`。
这里的 `origin` 指向本地 bundle。蓝区后续更新时由联网侧重新生成文件，恢复到新的验证目录即可。

这一步解决的是 **源码及两个 Git 依赖的获取**。离线编译还要有配套 CANN、Python 构建包，
以及 SHMEM 构建所需的额外依赖，或者直接交接匹配环境的已验证 SDK/wheel；它们不会自动包含在 bundle 中。

## 5. 联网副本后续如何拿最新修改

在自己的验证副本中，先检查本地修改，再更新：

```bash
cd /你的工作目录/DeepEP-train
git status
```

确认当前没有需要保留的未提交修改后，执行：

```bash
git switch feat/megamoe-train-bf16
git pull --ff-only origin feat/megamoe-train-bf16
git submodule update --init third-party/shmem third-party/catlass
git log -1 --oneline
```

`pull --ff-only` 遇到本地与远端历史分叉会停止。此时先核对双方提交，不用 `reset --hard` 覆盖已有工作。
有未完成修改时，先提交到自己的工作分支，或另建干净验证副本。

**开始一轮测试后先固定版本，不在测试过程中继续 pull。** 每份测试结果都写明 `git rev-parse HEAD` 的完整输出。
代码更新后按改动重新构建安装；`git pull` 只更新源码，不会自动更新环境中的 wheel/SO。

## 6. 拿到代码后，到哪里找验证命令

按 [训练模块 README](../../experiments/megamoe_train/README.md) 的 **Build and installation** 安装训练 wheel，
再按 **Tests** 和 [首轮迁移验收](首轮迁移验收.md) 第 6 节执行测试。

| 想做的事情 | 入口 |
| --- | --- |
| 编译、安装、确认公开接口 | [训练模块 README](../../experiments/megamoe_train/README.md) |
| 双卡前反向精度、生命周期验证 | [首轮迁移验收](首轮迁移验收.md) |
| 对照上次 EP4、EP8 性能与方法 | [EP4/EP8 性能对照](EP4_EP8性能对照_20261028.md) |
| 判断 H7168 的已知精度失败 | [H7168 ptrDX 精度归因](H7168_ptrDX精度归因.md) |

上次运行时代码为 `ae530ad`，随后资料整理到 `cf00f1f`；源版本固定为 `b8cff148`。
上次设备环境是 CANN **9.2.0-beta.2**，而仓库正式要求 **9.2.0**。复测需记录实际环境，不能只凭 Git 提交相同就认为环境和结果完全一致。

## 7. 验证后怎样反馈或提交修改

**只报告结果：** 可以先把结果发给席星统一整理，不必修改代码。至少写清楚：

```text
测试提交：完整 commit SHA
环境：Ascend950 / CANN / Torch / torch_npu
测试命令和 shape：
结果：PASS / FAIL / 未执行
失败摘要或性能数据：
原始日志位置：
```

**需要修改代码或补文档：** 建议从已更新的训练分支创建自己的工作分支。例如：

```bash
git switch -c review/gff-train-validation
# 编辑需要修改的文件，并完成对应验证。
git diff
git status
git add <实际修改的文件路径>
git diff --cached
git commit -m "docs: record training validation"
git push -u origin review/gff-train-validation
```

以上 `push` 在能访问 GitHub 且已配置本人凭据的电脑上执行。如果修改是在蓝区完成，
在蓝区提交后用下面命令导出，将生成的文件带回联网电脑：

```bash
git bundle create ../gff-review.bundle review/gff-train-validation
```

联网电脑在自己的 `xixing-321/DeepEP` 克隆中导入并推送（示例工作分支名须尚未在该副本使用）：

```bash
git fetch /收到文件的目录/gff-review.bundle review/gff-train-validation:review/gff-train-validation
git push -u origin review/gff-train-validation
```

示例提交说明适用于验证文档；修改代码时写清实际改动。提交前检查按仓库 [贡献指南](../../CONTRIBUTING.md) 执行。
`git add` 后面替换为实际文件路径，只提交本次需要的文件。使用个人工作分支，方便两人审查，避免直接覆盖共同验证分支。

随后在 GitHub 创建 Pull Request，检查目标：

- **base 仓库：** `xixing-321/DeepEP`，不要误选 fork 上游 `lueelu/DeepEP`。
- **base 分支：** `feat/megamoe-train-bf16`。
- **head 分支：** 自己的 `review/gff-train-validation`。

PR 中写明改了什么、跑了哪些验证、还有哪些未验证。两人讨论和审查后再决定是否合入协作分支。
验证发现问题也可以在 GitHub Issue 或 PR 评论里附上上述信息；不要把密码、令牌、服务器地址或完整内部日志贴进公开仓。

## 8. 当前资料的存放边界

中文分工、验证结果和定位过程继续放在 `docs/megamoe_train_migration/`。
**该目录仅用于协作分支，后续不得带入 main 或 `lueelu/DeepEP` 的正式 PR。**
正式交付要另外整理干净的代码与长期文档，具体见 [本目录 README](README.md)；不能直接把当前迁移草稿 PR 合入 main。
