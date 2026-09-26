# Hermes Copilot SDK

> **Superseded for the current deployment:** use
> [hermes-copilot-cli](https://github.com/Chibaheit/hermes-copilot-cli).
> Hermes is now the coordinator and dispatches directly to Copilot CLI workers.
> The experimental SDK provider below is retained for existing opt-in users,
> but is no longer part of the active architecture.

An **experimental, opt-in model-provider plugin** for
[Hermes Agent](https://github.com/Chibaheit/hermes-agent), using the official
[`github-copilot-sdk==1.0.14`](https://github.com/github/copilot-sdk/releases/tag/v1.0.14).
It is not Hermes' existing `copilot` direct API provider or `copilot-acp` transport.

The SDK is an agent runtime, not an OpenAI chat API. This plugin adapts its
**text responses** to Hermes' existing completion contract. Hermes retains its
conversation history, outer loop, tool dispatch, and approval checks. Copilot
receives no native tools.

## Install and enable

Use the **same Python 3.11-3.13 environment as your existing Hermes installation**.
The compatibility baseline is Hermes commit
`03fee43ca344ead7245a3b0ae20d38de0ae75642`; later versions have not been certified.
No Hermes core patches are required.

```sh
git clone --depth 1 https://github.com/Chibaheit/hermes-copilot-sdk.git
cd hermes-copilot-sdk
python -m pip install -e .
python -m copilot download-runtime
```

The SDK downloads its matching runtime (1.0.85 for this SDK release), with checksum
verification. Do not substitute an arbitrary interactive Copilot CLI release.
An optional `copilot_sdk.cli_path` must point to a compatible executable.

Merge these entries into the **active Hermes profile's** `config.yaml`, preserving
any existing enabled plugins. Restart Hermes after changing discovery settings:

```yaml
plugins:
  enabled:
    - copilot-sdk

model:
  provider: copilot-sdk
  default: YOUR_COPILOT_MODEL_ID
  base_url: copilot-sdk://runtime
  api_mode: chat_completions

copilot_sdk:
  timeout_seconds: 120
  # cli_path: C:\path\to\compatible\copilot-runtime.exe
```

Put `COPILOT_GITHUB_TOKEN` in that profile's `.env`, using a GitHub credential
with the required Copilot entitlement and permissions. Do not commit credentials.
See the SDK's [authentication guide](https://github.com/github/copilot-sdk/blob/v1.0.14/docs/auth/authenticate.md).
The plugin deliberately does **not** read another profile's token, `gh auth`
credentials, or the interactive Copilot login/keychain.

Discover account-authorized model IDs from the same configured Hermes environment:

```sh
python -c "from hermes_cli.env_loader import load_hermes_dotenv; load_hermes_dotenv(); from hermes_copilot_sdk.client import CopilotSDKClient; c=CopilotSDKClient(); print(c.list_models()); c.close()"
```

Then use `hermes --provider copilot-sdk --model YOUR_COPILOT_MODEL_ID`.
This baseline's interactive `hermes model` picker intentionally excludes third-party
external-process providers; use explicit configuration/flags instead. Executable
availability is not proof of authentication; invalid credentials fail at the SDK.

## Data flow and boundaries

1. Hermes discovers the explicitly enabled pip entry point and resolves an
   external-process provider. The SDK, not the `python` availability check, launches
   the matching runtime.
2. Each completion resolves the active profile's configuration, credential, and
   sanitized child environment. A fresh private request directory is created under
   `$HERMES_HOME/copilot-sdk/`.
3. System/developer messages and the Hermes tool contract form a stable system
   prefix. The remaining messages are serialized as JSON, retaining tool-call
   IDs, arguments, and results. No Hermes history is modified.
4. The SDK starts in `mode="empty"` with `available_tools=[]`, a reject-all permission
   handler, discovery/hooks/skills/memory/session telemetry disabled, and private
   config/working directories. No custom SDK tools or MCP servers are registered.
5. Assistant `<tool_call>...</tool_call>` text is strictly checked and converted
   into OpenAI-shaped tool requests. **Only Hermes executes those requests**, through
   its existing tool/approval path. Malformed or unoffered calls raise errors.
6. The request has an outer deadline covering startup, session creation, and sending.
   On completion, failure, timeout, or cancellation, the session is aborted,
   disconnected/deleted and the runtime stopped; shutdown has bounded force-stop
   fallback. The private temporary directory is then removed.

This is **not an OS sandbox**. The runtime still performs its own configuration,
logging, and session/cache I/O; a crash can leave request directories for the
operator to inspect. The SDK's downloaded executable cache is outside the temporary
request directory. Hermes' existing secret redaction and tool protections are not
changed.

## Deliberate limitations

- **Text only.** Image/audio/other non-text parts fail explicitly; configure a
  separate vision provider. Native SDK tools, MCP, autonomous delegation and
  Copilot skills are unavailable by design.
- **Tool requests are a text protocol**, not native structured tool calling.
  `tool_choice` is validated at the response boundary, but the model may fail to
  comply. Malformed output is an error, not an empty successful answer.
- **Buffered streaming:** sync and async consumers receive chunks only after the
  SDK's final answer. There are no real-time token deltas.
- **No native full-history import or session reuse.** Every request starts fresh
  to avoid appending duplicate transcripts. Prefix ordering is stable, but remote
  prompt-cache hits are not guaranteed; startup overhead and costs can be higher.
- **No structured `response_format` or reliable usage accounting.** Usage is
  reported as unavailable, not fabricated zero-token usage. Temperature, token
  caps and other OpenAI-only sampling controls are not mapped; the SDK controls
  them. `reasoning_effort` is forwarded when supplied.
- Explicit configuration only; this plugin does not rewrite your model choice or
  silently fall back to a different backend.

## Development

Tests exercise real Hermes discovery, profile routing, and SDK types, with model
inference replaced at the SDK boundary. The optional real-runtime smoke test
requires the downloaded matching runtime but **no account credentials or model
request**.

From the Hermes checkout, use its canonical runner with this plugin installed:

```sh
HERMES_PYTHON=/path/to/hermes/python \
  bash scripts/run_tests.sh /absolute/path/to/hermes-copilot-sdk/tests -q
```

Append `-- --run-copilot-runtime-smoke` to also exercise the real, credential-free
SDK runtime and verify its session exposes zero native tools.

On Windows PowerShell:

```powershell
$env:HERMES_PYTHON = 'C:\path\to\hermes\Scripts\python.exe'
& 'C:\Program Files\Git\bin\bash.exe' scripts/run_tests.sh 'C:\path\to\hermes-copilot-sdk\tests' -q
```

## 中文说明

这是独立的、默认关闭的 **Copilot SDK 模型提供方插件**，不是已有的
`copilot` API 或 `copilot-acp`。在 Hermes 使用的同一个 Python 环境安装，
在当前配置档的 `config.yaml` 中启用 `copilot-sdk`，并把具备 Copilot
权限的 `COPILOT_GITHUB_TOKEN` 放入该配置档的 `.env`；不要提交密钥。

每次请求都会创建独立 SDK 会话，禁用 Copilot 原生工具、技能和配置发现。
Hermes 的完整对话以文本传入，返回的工具调用经过校验后仍由 **Hermes
执行和审批**，不会把执行权限交给 Copilot。不同配置档不共用凭据或会话。

当前为实验性文本兼容层：不支持图片和原生结构化工具调用；流式输出是
完整答案生成后的分块；不承诺远端缓存命中，也不伪造 token 用量。
超时会触发中止和清理。运行时仍需要文件读写，因此这些设置不是操作系统沙箱。
此版本需要显式配置或命令行参数，暂不加入交互式模型选择菜单。
