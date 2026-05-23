# 修复当前测试失败（PR1）

## 目标

让 **OpenAI 官方 `gpt-5.4-mini`** 在跑完 `uv run pytest --config -v` 后，所有**适用于该模型的**测试全部通过，不适用的被合法 skip（而不是 FAIL）。

这是 [基于 Profile 的兼容性测试方案](profile-based-compatibility-testing.md) 的前置工作。本 PR **不引入 profile 基础设施**，只利用现有的用户声明 capability 机制，先把测试本身的质量问题和交叉依赖问题解决掉。

## 基线

参照报告：[reports/20260418_124634/summary.md](../reports/20260418_124634/summary.md) —— 5 failed / 26 total on `gpt-5.4-mini`（OpenAI 官方端点）。

## 失败归类

| 测试 | 失败根因 | 分类 |
|------|----------|------|
| `test_max_tokens` | gpt-5.4-mini 不认 `max_tokens` | (b) 模型不支持 |
| `test_stop_sequence` | gpt-5.4-mini 不认 `stop` | (b) 模型不支持 |
| `test_logprobs` | 内部混入了 `max_tokens: 10`（交叉依赖） | (a) 测试写错 |
| `test_image_url_input` | 1×1 PNG 被 OpenAI 视觉端拒绝 | (a) fixture 错 |
| `test_image_url_with_detail` | 同上 | (a) fixture 错 |

- **(a) 测试写错** → 改测试
- **(b) 模型不支持** → 加细粒度 capability marker，靠 [config.yaml](../config.yaml) 的 capabilities 声明驱动 skip

## 工作流约束（重要）

每修完一个问题，**只跑与该问题相关的 test case 做验证**，不要每次都跑全量。目的：缩短反馈循环、减少 API token 消耗、避免重复触发已通过的测试。

单测 CLI 模板：
```bash
uv run pytest tests/openai_compat/<file>.py::<TestClass>::<test_name> --config -v
```

全量只在**最后的验收阶段**跑一次（见文末"验收标准"）。

## 具体改造

### 1. 修复 vision fixture（(a) 类）

**文件**：[tests/openai_compat/test_vision.py](../tests/openai_compat/test_vision.py)

- [test_vision.py:10-13](../tests/openai_compat/test_vision.py#L10-L13) 的 `TINY_PNG_B64` 是 1×1 红色 PNG，OpenAI 视觉端拒绝。换成至少 32×32 的合法 PNG（纯色或简单渐变皆可）
- [test_vision.py:41,76](../tests/openai_compat/test_vision.py#L41) 的 `max_tokens: 100` → 改成 `max_completion_tokens: 100`（gpt-4o 和 gpt-5.4-mini 都接受 `max_completion_tokens`，向前兼容）

**验证**：只跑这两个 vision 测试
```bash
uv run pytest tests/openai_compat/test_vision.py::TestVision::test_image_url_input \
              tests/openai_compat/test_vision.py::TestVision::test_image_url_with_detail \
              --config -v
```
预期：2 passed。

### 2. 修复 test_logprobs 交叉依赖（(a) 类）

**文件**：[tests/openai_compat/test_chat_basic.py:307-335](../tests/openai_compat/test_chat_basic.py#L307-L335)

- 第 317 行 `"max_tokens": 10` → `"max_completion_tokens": 10`
- 这样测试只考察 `logprobs`，不再绑定 `max_tokens` 能力

**验证**：只跑 test_logprobs
```bash
uv run pytest tests/openai_compat/test_chat_basic.py::TestChatBasic::test_logprobs \
              --config -v
```
预期：1 passed。

### 3. 给 test case 加细粒度 capability marker

| 测试 | 新增 marker |
|------|-------------|
| `test_max_tokens` | `@pytest.mark.capability("max_tokens")` |
| `test_stop_sequence` | `@pytest.mark.capability("stop_sequences")` |
| `test_logprobs` | `@pytest.mark.capability("logprobs")` |

**注意**：不动其他已经在跑且通过的测试（如 `test_max_completion_tokens` / `test_n_parameter` / `test_seed` / `test_json_mode` 等），因为它们对 gpt-5.4-mini 已经 PASS。PR1 尽量最小改动。

> 这些 test case 在 PR2（profile 实施）时会继续扩展 marker 覆盖，但本 PR 只加会影响 gpt-5.4-mini skip 决策的那几个。

**验证**：本步骤只加 marker，还没更新 config.yaml，所以三个测试仍会跑且结果保持原样（两个 FAIL、`test_logprobs` 在步骤 2 之后已 PASS）。**跳过本步骤的单独验证**，直接进步骤 4——marker 的真正效果（被 skip）要和步骤 4 的 config 一起验证。

### 4. 更新 config.yaml

**文件**：[config.yaml](../config.yaml)

gpt-5.4-mini 的 `capabilities` 列表显式**不包含** `max_tokens`、`stop_sequences`、`logprobs`：

```yaml
providers:
  - name: "openai-official"
    base_url: "https://api.openai.com"
    api_key_env: "OPENAI_API_KEY"
    api_format: "openai"
    models:
      - name: "gpt-5.4-mini"
        capabilities:
          - chat
          - streaming
          - tools
          - vision
          # 不声明：max_tokens、stop_sequences、logprobs
          # 这三个 test case 会被 _should_skip_for_capability skip
```

其他 capability（`max_completion_tokens`、`n_multi`、`seed` 等）因为对应的测试没加 marker，不会受影响——这是故意的最小改动。

**验证（步骤 3+4 合起来验证 marker + config 的配合）**：只跑那三个带新 marker 的测试
```bash
uv run pytest tests/openai_compat/test_chat_basic.py::TestChatBasic::test_max_tokens \
              tests/openai_compat/test_chat_basic.py::TestChatBasic::test_stop_sequence \
              tests/openai_compat/test_chat_basic.py::TestChatBasic::test_logprobs \
              --config -v
```
预期：**3 skipped**（skip reason 包含 `lacks capability`），0 passed、0 failed。

## 验收标准

PR1 合并后，执行：

```bash
uv run pytest --config -v
```

预期结果（OpenAI 官方 gpt-5.4-mini）：
- **0 failed**
- 3 skipped（`test_max_tokens` / `test_stop_sequence` / `test_logprobs`——均因 capability 未声明）
- 23 passed

报告里的 skipped 应有清晰的 reason：`"Model 'gpt-5.4-mini' lacks capability 'max_tokens'"` 之类。

## 不在本 PR 范围

以下留给 PR2（profile 实施）：
- 引入 `ModelProfile` / `ProfileRegistry`
- 从用户声明 capability 切换到 profile 解析
- 其余 test case 的细粒度 marker 扩展
- `--ignore-profile` CLI 开关
- 报告中标注所用 profile

## 预估工作量

- 代码改动：< 50 行（主要是 3 个 marker + 2 处 `max_tokens` 改名 + 1 个 PNG fixture 替换）
- Review 重点：新 PNG 合法性、capability 名字选择（`stop_sequences` vs `stop`？`n_multi` vs `n`？）
