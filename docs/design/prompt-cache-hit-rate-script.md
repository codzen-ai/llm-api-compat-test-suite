# Prompt 缓存命中率压测脚本

> 这是**设计文档**（记录为什么这么做、决策与权衡）。如果你只想知道**怎么用**这个脚本，看 [docs/usage/cache-hit-rate.md](/docs/usage/cache-hit-rate.md)。

## 背景与目标

现有 compat 测试套件覆盖功能正确性 + TTFT/TPOT 延迟，但 **没有衡量 prompt cache 命中率** 的入口。Prompt cache 是衡量第三方 OpenAI-兼容网关质量的关键指标 —— 同样的请求被官方 OpenAI 缓存命中、第三方却不命中，对用户的延迟和成本影响极大。

**目标**：提供一个独立可重现的脚本，模拟"长系统提示 + 长会话"工作负载，统计累计的 `cached_tokens / prompt_tokens` 比值，输出可对比的报告。

**测试方法**（用户给定规格）：

- 共发起 500 次多轮对话请求，每次请求 ≥1 秒间隔
- 首轮 prompt 长度：平均 28K tokens，最高 80K tokens（随机生成）
- 后续每轮在上一轮 messages 基础上追加：平均 1.3K，最高 5K
- 每轮输出：平均 0.3K，最高 1.5K
- 命中率公式：`Σ cached_tokens / Σ prompt_tokens`（所有 500 次请求的总和之比）

## 设计决策

### 1. 独立 script，不走 pytest

放在新建的 [scripts/cache_hit_rate.py](/scripts/cache_hit_rate.py)，CLI 入口由 `uv run python scripts/cache_hit_rate.py ...` 触发。

**理由：**

- 跑一次 ~30–60 分钟，与现有秒级 compat 测试节奏不匹配，混在 `pytest` 默认收集里会污染常规反馈循环
- 不需要 capability gating、per-model parametrization、profile resolution 这套基础设施 —— 它只针对单 (provider, model) 跑一次
- 报告语义是"一次压测产生一组数字"，不是"N 个 test case 各自 pass/fail"，套 pytest report 会扭曲
- pytest fixture 链对长时运行 + 严格节奏控制（每秒一次）也不顺手

**复用项**：仍然复用 `src/config.py`（`SuiteConfig` / `ProviderConfig`）解析 `config.yaml`，避免 CLI 参数重复定义鉴权信息。但 `src/http_client.py` 的 `LoggingHttpClient` 不直接复用 —— 它把每次请求都累计到 `records` list，500 次大 payload 会吃光内存。脚本内直接用 `httpx` 客户端，按需写日志。

### 2. 上下文溢出 → 开新会话，继续凑满 500 次

当一次请求失败且失败原因判定为"上下文超长"时：

- 丢弃当前会话（清空 `messages`，重置首轮标志）
- 不把这次请求计入命中率分母（请求都没成功就没有 usage 数据）
- 继续凑到 **成功** 请求数 = 500 为止
- 在最终报告里单独统计：会话数、每个会话坚持了多少轮、因溢出重启了几次

**溢出判定**：

- HTTP 400 / 422 + 响应体 JSON 里 `error.code` 或 `error.message` 含 `context_length` / `maximum context` / `token limit` 关键词
- 其他 4xx/5xx 不算溢出 —— 记为"请求失败"另外统计，**不重置会话**（可能是网络抖动），下一轮在同会话继续
- 连续 3 次非溢出失败仍触发会话重置，避免死循环

### 3. Prompt 文本生成：~4 chars/token 粗略估算

不引入 tiktoken / 任何特定分词器。

- 维护一个固定的英文 lorem-ipsum 语料字符串（约 10K 字符，硬编码或读 `tests/fixtures/`）
- 需要 N tokens → 拼接 `lorem_text * ceil(N*4 / len(lorem_text))`，截取前 `N*4` 字符
- 服务端返回的 `usage.prompt_tokens` 是真实分词结果 —— **命中率计算只依赖这个真值**，本地估算只是用来"造出大致够长的输入"，不参与统计

**首轮 prompt 结构**：放在 `messages[0]` 的 `user` 角色里，前缀加一句 `"Summarize the following text:\n\n"` 后接生成的长文本。不放 `system` 角色 —— OpenAI 缓存对 system/user 都生效，但用 user 简化跨格式适配（如果将来扩到 anthropic）。

**后续轮追加**：每轮在历史 messages 末尾追加新的 `user` 消息（短）+ 服务端返回的 `assistant` 消息。"追加 1.3K tokens"指 user 那条新消息的长度。

### 4. 长度采样分布

需要"平均 X，最高 Y"两个参数。选 **截断正态分布**：

```python
def sample_length(avg: int, max_value: int, rng: random.Random) -> int:
    # std = avg/3 → 大部分样本落在 [0, 2*avg]，再硬截到 [1, max_value]
    while True:
        s = int(rng.gauss(avg, avg / 3))
        if 1 <= s <= max_value:
            return s
```

- 不要 lognormal —— 长尾会让"max"经常爆掉，需要丢弃重采样
- 不要 triangular —— `mode = 3*avg - max` 在 (1.3K avg, 5K max) 场景下算出负数
- 截断正态简单、可调（std 比例可改）、500 个样本的经验均值与 28K/1.3K/0.3K 误差 <5%

### 5. 节奏控制：sequential，间隔 ≥ 1 秒

多轮对话内在串行（turn N+1 的 messages 包含 turn N 的回复），不可能真并发"每秒一次"。

实际语义：**记录每轮请求的发起时刻 `t_start`，下一轮在 `max(now, t_start + 1.0)` 时刻发起**。如果上一轮响应耗时 > 1 秒（大概率，28K prompt 通常要十几秒），就直接发；否则 sleep 补齐到 1 秒。

### 6. 数据采集

每次请求成功后立即提取（不缓存大 response body）：

```python
{
    "session_id": int,           # 第几个会话（溢出重启后 +1）
    "turn": int,                 # 该会话内的第几轮
    "prompt_tokens": int,        # usage.prompt_tokens
    "cached_tokens": int,        # usage.prompt_tokens_details.cached_tokens（缺失则 0）
    "completion_tokens": int,
    "elapsed_ms": float,
    "wall_time_iso": str,
}
```

- `prompt_tokens_details.cached_tokens` 是 OpenAI 标准字段；非 OpenAI 兼容网关可能不返回 → 记 0
- 脚本在第一次请求后检测：如果 `prompt_tokens_details` 整个键缺失，打印一次 WARNING 但不中止（命中率会显示为 0%，这本身就是个有意义的结果）

### 7. 配置入口

CLI 形态：

```bash
uv run python scripts/cache_hit_rate.py \
    --config config.yaml \
    --provider mgtv \
    --model glm-5.1 \
    [--total-requests 500] \
    [--interval-sec 1.0] \
    [--initial-avg-tokens 28000 --initial-max-tokens 80000] \
    [--append-avg-tokens 1300 --append-max-tokens 5000] \
    [--output-avg-tokens 300 --output-max-tokens 1500] \
    [--seed 42] \
    [--output-dir reports/cache_hit_rate]
```

- `--config` 复用 `SuiteConfig.from_yaml`，从中按 `--provider` / `--model` 定位目标
- 如果 config 只有一个 provider/model，对应 flag 可省略
- `--api-format` 默认 openai；初版只支持 openai 格式（anthropic 的 `cache_read_input_tokens` 字段不同，留作后续）
- 所有调参都给默认值 = 题目规格，开箱即用

### 8. 输出

每次跑产出一个目录：`reports/cache_hit_rate/{timestamp}_{provider}_{model_slug}/`

```
20260524_153012_mgtv_glm-5.1/
├── summary.md           ← 人类可读总结
├── per_request.csv      ← 500 行原始数据，便于后续 plot / 复算
└── config.json          ← 本次跑的全部入参 + git commit hash，可重现性凭据
```

**summary.md 模板**：

```markdown
# Prompt Cache Hit Rate Report

Run: 20260524_153012
Provider: mgtv (https://aigc-llm.mgtv.com)
Model: glm-5.1
Git commit: <sha>

## Headline

- Cache hit rate: **42.3%** (Σ cached_tokens 12,345,678 / Σ prompt_tokens 29,178,234)
- Successful requests: 500 / 503 attempted
- Sessions: 4 (3 restarts due to context overflow)
- Wall-clock duration: 47m 12s

## Per-session breakdown

| Session | Turns | Σ prompt_tokens | Σ cached_tokens | Hit rate |
|---|---|---|---|---|
| 1 | 167 | 9,234,123 | 4,123,456 | 44.7% |
| 2 | 158 | 8,891,002 | 3,890,221 | 43.8% |
| ...

## Sampling fidelity (sanity check)

| Stat | Spec | Empirical mean | Empirical max |
|---|---|---|---|
| Initial prompt tokens | avg 28K / max 80K | 27,841 | 79,212 |
| Append tokens / turn | avg 1.3K / max 5K | 1,287 | 4,891 |
| Output tokens / turn | avg 0.3K / max 1.5K | 295 | 1,448 |

## Failures

| Reason | Count | Notes |
|---|---|---|
| context_length_exceeded | 3 | Triggered session restart |
| timeout | 0 | |
| other 5xx | 0 | |
```

终端只打 headline + summary 路径，不刷屏 500 行。

### 9. 可重现性：fixed seed

`--seed` 默认 42。脚本里所有随机来源（长度采样、文本拼接的起始 offset 如果有）都从同一个 `random.Random(seed)` 实例派生。这样同一 `--seed --total-requests --avg/max` 组合下，**发给服务端的 prompt 长度序列完全一致**，不同 provider / 不同时间跑的命中率结果可直接对比。

服务端响应不可控（命中率本来就受服务端缓存状态影响），但"输入差异"被消除。

### 10. 错误处理与中断

- `KeyboardInterrupt` 优雅退出：把已收集的数据写成 partial report（在 `summary.md` 顶部标 `**INCOMPLETE: X/500**`）
- 单次请求 timeout 设为 120 秒（28K prompt 可能慢但不应永挂）
- httpx 网络异常（连接重置等）算作"非溢出失败"，按 §2 处理

## 实现步骤

### Step 1: 脚手架
新建 [scripts/cache_hit_rate.py](/scripts/cache_hit_rate.py)，把 CLI 解析、配置加载、输出目录创建、`Sampler` 类、`LoremGenerator` 类骨架先撸出来；不发请求，跑 `--total-requests 5` 验证采样分布合理。

### Step 2: 请求循环 + usage 提取
接入 httpx，把单次 POST `/v1/chat/completions` 跑通；验证 `usage.prompt_tokens_details.cached_tokens` 能取到（用 mgtv glm-5.1 真发一次，确认字段存在）。

### Step 3: 多轮会话 + 溢出重启
把 messages 累加、追加 assistant 回复、检测溢出、重启会话的逻辑闭环；用 `--total-requests 20 --initial-avg-tokens 5000` 之类小规格跑通。

### Step 4: 节奏控制 + 完整 500 次
加入 §5 的 `max(now, t_start + 1.0)` 时序、SIGINT 处理、120s timeout。这一步要小心：尝试用 dry-run（mock httpx 返回固定 usage）验证时序逻辑，再真跑。

### Step 5: 报告产出
实现 §8 的 summary.md / per_request.csv / config.json 三件套。

### Step 6: 文档与示例
- 在 [CLAUDE.md](/CLAUDE.md) 的 Commands 段加一行示例
- 本设计文档保持作为唯一的"为什么这么做"参考

## 非目标（明确不做）

1. **不支持 anthropic / gemini 格式** —— usage 字段名/结构都不同，初版只做 OpenAI 格式；将来若要扩，加一层 `extract_cache_usage(api_format, response)` 适配函数
2. **不并发** —— 多轮对话内在串行；如果将来要测"并发会话各自的缓存命中"是另一个完全不同的测试，单独写
3. **不集成进 pytest 报告** —— 见 §1
4. **不引入 tokenizer 依赖** —— ~4 chars/token 估算 + 信任服务端返回，足够；本测试不是在测分词器
5. **不做命中率断言/阈值** —— 这是测量工具，不是 pass/fail 测试。如果将来要"低于 X% 失败"，再考虑包一层 CI gate

## 待确认

- 语料：硬编码一段 lorem ipsum vs 读 `tests/fixtures/` 下的文件？倾向硬编码（脚本自足，不增加 fixture 依赖）
- `reports/cache_hit_rate/` 命名是否和现有 `reports/{ts}/` 区分清楚？或者放 `cache_reports/`？倾向前者，统一在 `reports/` 下便于 .gitignore 管理
