# 贡献到 VSAG

首先，感谢你愿意花时间为 VSAG 做贡献！正是像你一样的贡献者帮助 VSAG 项目变得更好。🎉

> 如果你是第一次参与开源项目，我们非常推荐你跟着 [这个项目](https://github.com/firstcontributions/first-contributions/blob/main/docs/translations/README.zh-cn.md) 了解开源贡献的基本流程。

以下是为 VSAG 做贡献你可能需要知道的，了解这些有助于你更加轻松地为此项目做出贡献。

## 我可以做哪些贡献

1. 【报告错误】要报告 bug 或者文档问题，请创建 bug issue 并提供问题的详细信息。如果你认为该问题需要被优先关注，请在问题评论中 @ VSAG开发组。

2. 【提议新功能】要提议新功能，请创建 feature request issue。描述预期的功能，并与 VSAG 开发组和社区讨论设计和实现。一旦 VSAG 开发组同意该计划，就可以按照 [贡献流程](#贡献流程) 来实施它。

3. 【开发功能或者修复错误】要开发未实现的功能或者修复错误，请遵循 [贡献流程](#贡献流程) 。如果你需要关于这个问题的更多背景信息，可以在该问题上发表评论并 @VSAG开发组。

## 我该如何贡献

### 贡献代码

如果你有任何改进 VSAG 项目的地方，请创建你的 pull request！记得在你的 pull request 中引用相关 issue，如果有的话。

### 贡献流程

>  我们使用 [GitHub Flow](https://docs.github.com/en/get-started/using-github/github-flow) 来协作开发 VSAG 项目。了解 GitHub Flow 可以帮助你更快地参与到 VSAG 的社区开发中。

1. 在 GitHub 上 fork 一个 VSAG 仓库。
1. 使用 `git clone git@github.com:<yourname>/vsag.git` 命令将你的 fork 仓库下载到本地计算机。
1. 使用 `git checkout -b my-topic-branch` 创建分支。
1. 在本地进行修改，通过本地检查，创建提交并使用 `git push --set-upstream origin my-topic-branch` 推送到 GitHub。
1. 访问 GitHub 网站并创建 pull request。

如果你已有本地仓库，请在开始之前对其进行更新，以最大程度减少产生合并冲突的可能性。

```bash
git remote add upstream git@github.com:antgroup/vsag.git
git checkout main
git pull upstream main
git checkout -b my-topic-branch
```

### 一些准则

在创建 pull request 前，请确保你的修改通过了本地测试，并且符合 VSAG 编码风格。

- 在提交新功能时，pull request 需要包含功能测试，以证明你的代码是正常工作的，还可以避免未来的修改意外地破坏了这个功能。
- 在修复 bug 时，需要添加触发 bug 的测试用例，因为 bug 的存在通常表明测试覆盖不足。
- 在 VSAG 中修改代码时，要保持 API 的兼容性。
- 不要在 VSAG 的公开头文件（`include/` 目录）中引用内部头文件（`src/` 目录）。
- 当你向 VSAG 项目贡献新功能时，维护成本（默认情况下）会转移给 VSAG 开发组。这意味着我们要考虑贡献的好处和维护的成本。

### 签署 DCO（Developer Certificate of Origin）

对于本项目的所有贡献必须同意并附带 [Developer Certificate of Origin（后简称 DCO）](https://developercertificate.org/) 的确认。对 DCO 的确认和同意**必须**包含在每一个 Commit Message 中，并采用 `Signed-off-by: {{Full Name}} <{{email address}}>`（不带 `{}`）的形式。如果贡献者不能或不愿意同意 DCO，其贡献将不会被接收。

贡献者可以通过在 Commit Message 中添加如下 Signed-off-by 行来签署 DCO：

```text
This is my commit message

Signed-off-by: Random J Developer <random@developer.example.org>
```

Git 还有一个 `-s` 命令行选项，可以在提交时自动附加 Signed-off-by 行：

```bash
git commit -s -m "This is my commit message"
```

对于由人类开发者和 AI Coding Agent（如 OpenCode、Claude Code、Codex 等）共同完成的贡献，请为**每一位协作者**各添加一行 `Signed-off-by` trailer，例如：

```text
Signed-off-by: Random J Developer <random@developer.example.org>
Signed-off-by: OpenCode (claude-sonnet-4.5) <noreply@opencode.ai>
```

### Commit 信息与 PR 标签

- Commit 信息请遵循 [Conventional Commits](https://www.conventionalcommits.org/)，常用前缀包括 `feat:`、`fix:`、`docs:`、`chore:`、`refactor:`、`test:`、`ci:` 等；
- 如果该 commit 无需触发 CI，请将 `[skip ci]` 放在 commit subject 的**开头**，例如 `[skip ci] docs: fix typo in README`；
- 每一个 PR 都**必须**至少包含以下两类 label（由 Mergify 强制校验，否则无法合并）：
  - `kind/*`：变更类型，可选值为 `kind/bug`、`kind/feature`、`kind/improvement`、`kind/documentation`；
  - `version/*`：目标版本，例如 `version/1.0`、`version/0.18`。

### 编码风格

VSAG 项目编码风格基于 [Google C++ 风格指南](https://google.github.io/styleguide/cppguide.html) 做了一些修改，包括缩进、命名规则、行宽等，具体可以参考以下两个配置文件：

- clang-format：https://github.com/antgroup/vsag/blob/main/.clang-format
- clang-tidy：https://github.com/antgroup/vsag/blob/main/.clang-tidy

> clang-tidy 是一个静态代码分析的工具，配置文件中不仅定义了函数/变量的命名标准，定义了一些编码风格的检查，例如 Magic Number 使用的检查等。

VSAG 项目通过 Makefile 提供了格式化代码的命令，需要安装 clang-format 和 clang-tidy。

运行命令可以直接格式化代码：

```bash
make fmt
```

运行命令会静态代码检查，需要根据提示手动修复：

```bash
make lint
```

### 本地测试

VSAG 项目使用 Makefile 提供了方便运行所有测试的命令，请执行并确认所有测试通过：

```bash
make test
```

