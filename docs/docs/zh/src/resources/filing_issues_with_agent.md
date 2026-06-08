# 使用 AI Agent 创建 Issue

你可以借助 AI 编码 Agent（Claude Code、OpenCode 或 Codex）与 VSAG 仓库内置的
`/create-issue` 斜杠命令一起，为 VSAG 起草并提交一份高质量的 GitHub Issue。
Agent 会把你的需求映射到项目的 Issue 模板，自动填好必填字段，并通过 GitHub
CLI 提交。

本页面介绍端到端的使用步骤。Agent 内部遵循的规范工作流位于
[`.github/agent-prompts/create-issue.md`](https://github.com/antgroup/vsag/blob/main/.github/agent-prompts/create-issue.md)，
本页只关注用户侧的操作。

## 前置条件

- 一个 GitHub 账号。
- 本地已安装并配置好以下任意一个受支持的 AI 编码 Agent：
  [Claude Code](https://docs.claude.com/en/docs/claude-code/overview)、
  [OpenCode](https://opencode.ai/) 或 Codex。
- 本机可用的 `git`。

## 1. 安装并登录 GitHub CLI（`gh`）

先按官方快速入门在你的平台上安装 `gh`：

<https://docs.github.com/en/github-cli/github-cli/quickstart>

然后在终端登录：

```bash
gh auth login
```

选择 **GitHub.com**，挑选认证协议（HTTPS 即可），并按浏览器提示完成登录。

## 2. 验证 `gh` 登录状态

```bash
gh auth status
```

确认 GitHub.com 已成功认证后再继续。

## 3. 克隆 VSAG 仓库

```bash
git clone https://github.com/antgroup/vsag.git
cd vsag
```

`/create-issue` 命令及其 Prompt 文件都在仓库内，因此 Agent 必须在 `vsag/`
目录下启动，才能识别该命令。

## 4. 在仓库目录中启动 Agent

在 `vsag/` 目录下启动其中一个受支持的 Agent：

```bash
# Claude Code
claude

# OpenCode
opencode

# Codex CLI
codex
```

## 5. 运行 `/create-issue`

在 Agent 对话中调用斜杠命令，并用自然语言描述你的需求。例如：

```
/create-issue HGraph 在 dim=0 时构建会崩溃；希望返回一个明确的错误。
```

Agent 将会：

1. 在
   [`.github/ISSUE_TEMPLATE/`](https://github.com/antgroup/vsag/tree/main/.github/ISSUE_TEMPLATE)
   中选择最合适的模板；
2. 在必填字段缺失时主动追问；
3. 以 `path:line` 形式引用代码或文档，撰写 Issue 正文；
4. 把最终草稿展示给你确认；
5. 你确认后，通过 `gh issue create` 提交 Issue。

整个过程中你可以反复与 Agent 沟通——让它调整措辞、补充复现步骤、切换模板、
附加日志，再决定是否提交。

## 小贴士

- 描述要具体：报 Bug 时附上索引类型、参数、数据集形状、报错信息以及运行平台。
- 提需求时，描述使用场景以及期望的 API 或行为，Agent 会据此填好模板字段。
- Issue **不需要** `Signed-off-by:`——DCO 仅适用于 commit。
- 如果不想通过交互式 Agent 驱动整个流程，可参考仓库提供的 Shell 包装脚本
  [`tools/issue-helper/new-issue.sh`](https://github.com/antgroup/vsag/blob/main/tools/issue-helper/new-issue.sh)。

## 参见

- [开源社区](community.md)
- [贡献到 VSAG](../development/contributing.md)
