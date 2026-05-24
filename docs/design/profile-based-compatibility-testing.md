# 基于 Profile 的兼容性测试方案

## 背景与问题

当前设计把"API 兼容性"定义为"是否兼容 OpenAI API"，但 **OpenAI 自身的 API 在不同模型间并不一致**：

- `gpt-4o` 使用 `max_tokens` 控制输出长度
- `gpt-5.x` / o-series 只接受 `max_completion_tokens`，拒绝 `max_tokens`
- 部分模型不支持 `stop`、`logprobs`、`n>1` 等参数

这意味着，按"是否 OpenAI 兼容"作为统一标准，会得出"OpenAI 自己都不兼容自己"的荒谬结论。

**正确的问题**不是"是否兼容 OpenAI API"，而是：

> 在该模型声明的能力剖面下，第三方的行为是否符合 OpenAI 在同模型上的协议规范？

## 核心设计

### 三个关键概念

1. **Capability（能力）**：test case 上的标记，声明它考察的是哪一项参数/行为（如 `max_tokens`、`stop_sequences`、`n_multi`）。已有机制，需要扩展粒度。

2. **Profile（能力剖面）**：**项目内置**的 ground truth，记录"OpenAI 官方某个模型快照支持哪些 capability"。由人工（AI 辅助）维护，不由用户声明。

3. **Model 配置**：用户侧配置，声明"我要测的第三方模型 X 对标 OpenAI 的哪个 profile"。

### 数据流

```
用户 config.yaml
  ├─ 第三方 model.name           （第三方那边的模型标识）
  └─ model.profile              → 指向内置 profile
                                         │
                         model_profiles/openai/gpt-4o/2024-08-06.yaml
                                         │
                                     capabilities: {chat, max_tokens, ...}
                                         │
                                         ▼
                         pytest 过滤：只跑 marker ∈ capabilities 的 test
                                         │
                                         ▼
                            第三方 vs OpenAI 跑同一份子集 → 对比有意义
```

## Profile 设计

### Schema

```yaml
# model_profiles/openai/gpt-4o/2024-08-06.yaml
model: gpt-4o
snapshot: gpt-4o-2024-08-06          # OpenAI 带日期的不可变别名
api_format: openai
created_at: 2026-04-18               # 这份 profile 的创建日期
source_endpoint: https://api.openai.com   # profile 所参照的官方端点
capabilities:
  - chat
  - streaming
  - tools
  - vision
  - max_tokens
  - stop_sequences
  - n_multi
  - logprobs
  - seed
  - json_mode
  # ...
```

### 版本化

Profile 必须带时间维度，因为 OpenAI 会：
- 给现有模型加新参数（例：某天给 gpt-4o 加了 `parallel_tool_calls`）
- 弃用旧参数（`max_tokens` → `max_completion_tokens`）
- 让别名漂移（`gpt-4o` 今天指向 `gpt-4o-2024-08-06`，半年后可能指向 `gpt-4o-2025-03-15`）

**关键实践**：录 profile 时**用带日期的不可变快照名**（如 `gpt-4o-2024-08-06`），别名只在 `model:` 字段留做检索用。

### 目录结构

```
model_profiles/
  openai/
    gpt-4o/
      2024-08-06.yaml
      2025-03-15.yaml
    gpt-5.4-mini/
      {snapshot}.yaml
  anthropic/
    claude-sonnet-4-6/
      ...
```

### 缺失时的行为

用户配置引用了一个不存在的 profile → **直接 `pytest.UsageError`**，列出该 `api_format` 下所有可用 profile。不允许默默 fallback，避免"不知情地跑了错误的对比基准"。

### 测试报告必须标注所用 profile

由于 snapshot 省略时由框架自动推导"最新"，**报告必须显式记录每次跑实际加载的是哪个 profile 文件**，否则结果不可追溯（"两次跑同一 config，对比基准悄悄变了"）。

[reports/{ts}/summary.md](/reports/) 的 Configuration 表至少要新增以下行：

| Key | Value |
|-----|-------|
| Profile | `model_profiles/openai/gpt-4o/2025-03-15.yaml` |
| Profile snapshot | `gpt-4o-2025-03-15` |
| Profile created_at | `2025-03-15` |
| Profile resolution | `auto-latest`（或 `pinned`，标明是用户 pin 的还是自动取最新）|

多 model 时每个 model 独立列出。

## 用户配置侧

### 新的 `config.yaml` 形态

```yaml
providers:
  - name: "open-router"
    base_url: "https://openrouter.ai/api"
    api_key_env: "OPENROUTER_API_KEY"
    api_format: "openai"
    models:
      - name: "openai/gpt-5.4-mini"      # 第三方的模型标识
        profile: "gpt-5.4-mini"           # 对标 OpenAI 的同名模型；省略则等于 name
        # profile_snapshot: 2025-03-15    # 可选，pin 到特定快照；省略 → latest
```

**关键变化**：`ModelConfig` 不再有 `capabilities` 字段。capability 完全由 profile 决定，用户不参与。

### CLI 模式

```bash
uv run pytest --base-url https://... --api-key ... --model X --profile gpt-4o
```

`--profile` 必填（替代原来"硬编码全开 capability"的行为）。可选 `--profile-snapshot` 用于 pin。

### 录制模式（新增）

录制 profile 或调查兼容性时需要**绕过 profile 过滤、跑全量**：

```bash
uv run pytest --config --ignore-profile
```

该模式下所有带 capability marker 的 test case 都会跑，不做 skip，便于产出完整报告供 AI/人分析。

## 建立 Ground Truth 的流程（人工 + AI 协同）

**不做自动化录制**。理由：test case FAIL 可能源于 (a) 测试本身有 bug、(b) fixture 质量差（如 1×1 PNG）、(c) 模型真不支持——这些需要判断，不能简单用"跑过即支持"。

### 流程

1. 用 `--ignore-profile` 对 **OpenAI 官方 API** 跑全量，产出 `reports/{ts}/summary.md` + 日志
2. 将报告交给 AI（或人工）分析，每条 FAIL 归类到：
   - **(a) 测试用例/fixture 有问题** → 回去修测试，不写入 profile
   - **(b) 模型确实不支持该 capability** → 不写入 profile
   - **(c) 歧义/需 debug** → 挂起，不盲猜
3. 手写（AI 辅助生成）profile YAML，`capabilities` 只包含 PASS 的项
4. 人工 review 后 commit 到 `model_profiles/`

### 隐含收益

- **profile 可信度高**：每一条 capability 都经过判断
- **测试质量被反向拉高**：录 profile 的过程会揪出坏 fixture / 错断言（如 1×1 PNG 问题应该在录 gpt-4o profile 时就暴露）
- **同一流程可复用于第三方**：分析第三方 FAIL 时同样区分 (a)(b)(c)——能准确识别"声称兼容但静默丢参"这类隐蔽 bug

## 细粒度 Capability 清单（初版草案）

在现有 `chat / streaming / tools / vision` 基础上新增：

| Capability | 含义 | 相关测试 |
|------------|------|----------|
| `max_tokens` | 接受老字段 `max_tokens` | `test_max_tokens` |
| `max_completion_tokens` | 接受新字段 `max_completion_tokens` | `test_max_completion_tokens` |
| `stop_sequences` | 接受 `stop` 参数 | `test_stop_sequence` |
| `n_multi` | 支持 `n > 1` 返回多 choice | `test_n_parameter` |
| `logprobs` | 支持 `logprobs` / `top_logprobs` | `test_logprobs` |
| `seed` | 支持 `seed` 参数 | `test_seed` |
| `json_mode` | 支持 `response_format: json_object` | `test_json_mode` |
| `system_message` | 支持 `role: system` | `test_system_message` |
| `temperature` | 支持自定义 temperature | `test_temperature` |
| `top_p` | 支持 `top_p` | `test_top_p` |
| `frequency_penalty` | 支持 frequency_penalty | `test_frequency_penalty` |
| `presence_penalty` | 支持 presence_penalty | `test_presence_penalty` |

清单会随 test case 增长而扩展。粒度原则：**一个 capability 对应一个可独立开关的模型行为**。

### 关于"交叉依赖"的测试

如 `test_logprobs` 内部请求 `max_tokens: 10` —— 这会让它同时依赖 `logprobs` 和 `max_tokens` 两个 capability。三种处理方式：

- **方式 A**：挂多个 marker，任一缺失即 skip
- **方式 B**：拆成两个版本 `test_logprobs_with_max_tokens` / `test_logprobs_with_max_completion_tokens`
- **方式 C**（务实派）：直接用向前兼容字段（如 `max_completion_tokens`，老新模型都接受），只标一个 marker 考察主能力

PR1 对 `test_logprobs` 采用了**方式 C**。未来若出现"必须分别验证老/新字段"的需求，再走方式 B。方式 A 不推荐——它让"不支持 A"掩盖"是否支持 B"的事实，削弱了兼容性报告的信息量。

## 前置：PR1 修复现有测试问题（✅ 已完成）

本方案的实施依赖 [PR1：修复当前测试失败](/docs/design/fix-current-test-failures.md) 先完成。PR1 已合入（commit `919b8ba`），做了三件事：

1. 修复 fixture 错误（1×1 PNG → 64×64 合法 PNG）
2. 解决 `test_logprobs` 与 vision 测试的交叉依赖（`max_tokens` → `max_completion_tokens`）
3. 对受影响的 test case 加细粒度 capability marker（`max_tokens` / `stop_sequences` / `logprobs`）

PR1 完成后，OpenAI 官方 `gpt-5.4-mini` 在现有用户声明 capability 机制下 0 failed（23 passed / 3 skipped / 26 total）。本方案（PR2+）再把"用户声明 capability"替换为"profile 解析 capability"。

## 实施步骤（PR2+，TODO + Checkpoint）

按依赖顺序推进，每个 TODO 完成后**停下来让用户 review**，确认方向再进下一步。

### TODO 1：Profile 基础设施（✅ 已完成）

**产出：**
- `src/model_profile.py`：`ModelProfile` Pydantic 模型 + `ProfileRegistry` 加载器（模块名用 `model_profile` 以避开 stdlib `profile` 冲突）
- `model_profiles/` 目录结构（先放一份 `gpt-5.4-mini` 的样板 YAML，作为 schema 示例）

**关键点：**
- `ProfileRegistry.load(name, snapshot=None)`：查找 `model_profiles/{api_format}/{name}/{snapshot}.yaml`；snapshot 省略 → 扫描该目录下所有 YAML，按文件内 `created_at` 排序取最新（不维护 `latest.yaml`）
- Schema 严格 validate，缺字段即报错
- 返回的 `ModelProfile` 对象要携带来源文件路径（后续报告需要）

**✋ Checkpoint 1**：review `ModelProfile` 字段、YAML schema、registry API 设计。确认后才动下面。

---

### TODO 2：ModelConfig 引入 profile 字段（✅ 已完成，过渡 fallback 已在 TODO 7 移除）

**产出：**
- `src/config.py` 的 `ModelConfig`：
  - 新增 `profile: str | None = None`、`profile_snapshot: str | None = None`
  - 保留（**暂不删除**）`capabilities: list[str]` 字段，作为过渡期的 fallback
- 加载 config 后，对每个 model 调用 `ProfileRegistry.load`；加载失败时 fallback 到用户声明的 `capabilities`（打印警告）
- 新建 `ResolvedModel = (config, profile | None)` 容器

**为什么保留过渡期**：让现有 [config.yaml](/config.yaml) 在 profile 文件尚未建立时仍能跑，避免一次性 break 所有使用者。

**✋ Checkpoint 2**：跑 `uv run pytest --config -v`，确认 PR1 的配置仍能正常运行。

---

### TODO 3：capability 解析切到 profile（✅ 已完成）

**产出：**
- 修改 `conftest.py` 的 `_should_skip_for_capability`：优先从 `model.profile.capabilities` 读；profile 不存在时 fallback 到 `model.capabilities`
- `model_config` / `model` fixture 传递 `ResolvedModel`

**✋ Checkpoint 3**：把样板 profile 填入一个完整的 `gpt-5.4-mini.yaml`，把 [config.yaml](/config.yaml) 改成 `profile: gpt-5.4-mini`（去掉 capabilities 字段），跑测试确认结果与 PR1 等价。

---

### TODO 4：扩展细粒度 marker 覆盖（✅ 已完成）

PR1 只标了 3 个与 gpt-5.4-mini 失败相关的 marker。本步骤扩展到"细粒度 Capability 清单"里剩余项：`n_multi` / `seed` / `json_mode` / `system_message` / `temperature` / `top_p` / `frequency_penalty` / `presence_penalty` / `max_completion_tokens`。

**产出：**
- 所有 `tests/openai_compat/` 下的 test case 都标上对应 marker
- 若发现新的交叉依赖，按方式 C（务实）或方式 B（拆分）处理

**✋ Checkpoint 4**：审计 marker 覆盖表，确保每个被测行为都有一个对应 capability。

---

### TODO 5：建立初始 ground truth（✅ `--ignore-profile` 开关 + `gpt-5.4-mini` profile 已完成；其它模型待补）

**产出：**
- 加 `--ignore-profile` CLI 开关（在 `conftest.py` 的 `pytest_addoption`）
- 对 **OpenAI 官方** 跑 `--ignore-profile` 全量，产出 `reports/{ts}/summary.md`
- AI/人分析报告：每条 FAIL 归类 (a) 测试 bug → 回修；(b) 模型不支持 → 不写入 profile；(c) 歧义 → 挂起
- 写 profile 文件：初版至少包含 `gpt-5.4-mini`、`gpt-4o`

**✋ Checkpoint 5**：review 生成的 profile YAML，确认每一条 capability 都经过判断（不是盲目从 PASS 名单抄进来）。

---

### TODO 6：测试报告标注所用 profile（✅ 已完成）

**产出：**
- `src/report.py` 生成的 [summary.md](/reports/) Configuration 段加字段：Profile 文件路径、snapshot、created_at、resolution（auto-latest / pinned）
- 多 model 时每个 model 独立列出

**✋ Checkpoint 6**：跑一次测试看报告，确认"从报告可以反推出这次对比基准"。

---

### TODO 7：CLI 模式调整 + 移除过渡 fallback（✅ 已完成）

**产出：**
- CLI 新增 `--profile <name>` / `--profile-snapshot <date>`；`--profile` 必填
- 取消 [src/config.py](/src/config.py) 的 `from_cli` 中硬编码 capability 列表
- **删除** `ModelConfig.capabilities` 字段和 TODO 2/3 的 fallback 逻辑
- profile 缺失时直接 `pytest.UsageError`，列出可用 profile

**✋ Checkpoint 7**：所有现有 config 必须带 profile 才能跑。确认错误信息清晰。

---

### TODO 8：文档与示例更新（✅ 已完成）

**产出：**
- 更新 [config.yaml](/config.yaml) 示例，用新的 profile 语法
- 更新 [CLAUDE.md](/CLAUDE.md) "Adding a new test" 段落：说明测试必须标具体 capability、profile 由人工维护、不走自动录制
- 在 README / CLAUDE.md 添加"建立新 profile"的流程说明（引用本方案）

**✋ Checkpoint 8**：冷启动测试——新人从 README 入手能否顺利跑通一次对比。

---

## PR 切分建议

按 TODO 粒度切，避免一次动过多文件：

| PR | 范围 | 可独立合并 |
|----|------|-----------|
| **PR2a** | TODO 1~3（基础设施 + 过渡兼容） | ✅ 现有 config 不受影响 |
| **PR2b** | TODO 4（扩展 marker 覆盖） | ✅ 纯测试层 |
| **PR2c** | TODO 5（建立初始 profile） | ✅ 纯数据文件 |
| **PR2d** | TODO 6（报告标注） | ✅ 纯报告层 |
| **PR2e** | TODO 7~8（强制 profile + CLI + 文档） | ⚠️ 破坏性变更，需配合迁移说明 |

## 非目标（明确不做）

1. **不做自动化 profile 录制脚本**——"PASS = 支持"是误判，必须人工/AI 判断
2. **不允许用户 override profile 来"调通"测试**——那会把工具退化成"自欺欺人模式"
3. **不维护"模型 → test case"的显式映射表**——用 capability + profile 两层间接，可读性和可维护性更好
4. **不让 profile 缺失时静默 fallback**（过渡期除外）——避免不知情地用错误基准对比
