# Prompt Cache Hit-Rate 压测使用手册

[scripts/cache_hit_rate.py](/scripts/cache_hit_rate.py) 是一个独立脚本，模拟"长系统提示 + 长会话"工作负载，统计第三方 OpenAI-兼容网关的累计 `Σ cached_tokens / Σ prompt_tokens` 命中率。设计动机与决策见 [docs/design/prompt-cache-hit-rate-script.md](/docs/design/prompt-cache-hit-rate-script.md)。

## Quick Start

```bash
uv run python scripts/cache_hit_rate.py --config config.mgtv.yaml
```

跑完后看：

```
reports/cache_hit_rate/{时间戳}_{provider}_{model}/summary.md
```

`summary.md` 顶部的 **Cache hit rate** 行就是结果。

## 常见场景

### 1. 快速 sanity（5-10 分钟）

第一次跑、或换 provider 后先试一下，确认能拿到 `cached_tokens` 字段、报告写出正常：

```bash
uv run python scripts/cache_hit_rate.py \
    --config config.mgtv.yaml \
    --total-requests 30 \
    --initial-avg-tokens 5000 --initial-max-tokens 15000
```

30 次请求 + 较小 prompt，几分钟出结果。

### 2. 完整 500 次压测（默认负载）

```bash
uv run python scripts/cache_hit_rate.py --config config.mgtv.yaml
```

默认参数严格对应 spec：500 次、每隔 1 秒、首轮平均 28K / 最高 80K、追加平均 1.3K / 最高 5K、输出平均 0.3K / 最高 1.5K。实际耗时取决于服务端响应速度（28K prompt 通常 10-30 秒/次），**整跑大约 1.5-3 小时**。

### 3. 对比两个 provider

同 seed、同参数跑两次，输入序列完全一致，命中率可直接对比：

```bash
uv run python scripts/cache_hit_rate.py --config config.mgtv.yaml --seed 42
uv run python scripts/cache_hit_rate.py --config config.other.yaml --seed 42
```

报告目录的时间戳前缀保证不会覆盖。

### 4. 自定义负载

把首轮做大、追加做小，更贴近 RAG 场景：

```bash
uv run python scripts/cache_hit_rate.py \
    --config config.mgtv.yaml \
    --initial-avg-tokens 60000 --initial-max-tokens 100000 \
    --append-avg-tokens 500 --append-max-tokens 2000 \
    --output-avg-tokens 800 --output-max-tokens 3000
```

## CLI 参数速查

完整列表用 `uv run python scripts/cache_hit_rate.py --help`。下面是会经常用到的：

| Flag | 默认 | 何时要改 |
|---|---|---|
| `--config` | `config.yaml` | 多个 config 文件时切换目标 |
| `--provider` | (单 provider 时自动) | config 里有多个 provider 必须指定 |
| `--model` | (单 model 时自动) | provider 里有多个 model 必须指定 |
| `--total-requests` | 500 | 加大压力 / 降到几十做 sanity |
| `--interval-sec` | 1.0 | 限速要求严的网关调大；本地测试调到 0 全速 |
| `--initial-avg-tokens` / `--initial-max-tokens` | 28000 / 80000 | 模拟更长/更短系统提示 |
| `--append-avg-tokens` / `--append-max-tokens` | 1300 / 5000 | 模拟更长/更短轮次追加 |
| `--output-avg-tokens` / `--output-max-tokens` | 300 / 1500 | 模拟输出长度 |
| `--seed` | 42 | 想要不同 prompt 长度序列时换；想可重现就保持 |
| `--request-timeout-sec` | 120 | 超长 prompt 还是超时就调大（gateway 一般 <120s） |
| `--output-dir` | `reports/cache_hit_rate` | 想分离不同次压测的目录 |

## 输出文件解读

每次跑产出一个目录：`reports/cache_hit_rate/{时间戳}_{provider}_{model}/`，里面 3 个文件：

### `summary.md` — 人类阅读

- **Headline** 段：核心命中率 + 总分母分子 + 成功/尝试比 + 会话重启次数 + 总耗时
- **Per-session breakdown**：每个会话各自的 token 总量与命中率（会话因 context overflow 重启时分隔）
- **Sampling fidelity**：采样器实际生成的长度 vs spec，验证负载是否符合预期
- **Failures**：按 reason 聚合的失败计数（overflow / http_error / timeout / network）

### `per_request.csv` — 进 Excel / 复算

500 行原始数据，每行一次成功请求。列：`request_idx`、`session_id`、`turn`、`prompt_tokens`、`cached_tokens`、`completion_tokens`、`elapsed_ms`、`wall_time_iso`、3 个 target token 字段。

想自己算分段命中率、画时间序列、或核对脚本算的总数，全看这个文件。

### `config.json` — 可重现性凭据

本次跑的全部入参 + provider/model 信息 + git commit hash + 起止时间。两次跑结果对不上时，先 diff 这两个 config.json 看是不是参数变了。

## 怎么读结果

### 命中率的合理范围

| 命中率 | 含义 |
|---|---|
| **70-95%** | 正常表现 —— 长会话的 prefix 高度复用，缓存系统工作良好 |
| **30-70%** | 缓存命中部分有效，可能是 TTL 短/容量小/分片策略不友好 |
| **0-30%** | 缓存基本没起作用 —— 要么没开，要么缓存键策略有问题 |
| **正好 0%** | 见 troubleshooting 第 1 条，**先排除字段缺失** |

### 看哪几个数字

1. **Headline 的 cache hit rate** —— 一句话结论
2. **Per-session 各会话的 hit rate 是否随 turn 数上升** —— 正常应该 session 越长命中率越高（prefix 越多），如果反过来或者忽高忽低，可能是缓存 TTL 太短
3. **Sessions 数量** —— 多次 context overflow 重启会拉低整体命中率（因为每次重启第一轮必然 0% 命中），看是不是 prompt 设得太大
4. **Sampling fidelity 的 empirical mean** —— 应该和 spec 接近（±5%），偏太多说明随机分布有问题

## Troubleshooting

### `cached_tokens` 总是 0 / 命中率显示 0%

**先看 summary.md 有没有这一行**：

> ⚠ `usage.prompt_tokens_details` was missing on at least one response → hit rate above may be 0 even if caching is working

如果有，说明服务端响应根本没返回 `prompt_tokens_details.cached_tokens` 字段 —— 不是缓存没生效，是网关没透传。需要 provider 那边修。

如果没这条警告但命中率仍然 0%，那是缓存真的没起作用 —— 检查 provider 是否开启了 prompt cache。

### ETA 显示 `--:--`

rich 默认用 30 秒滑动窗口估速度。本脚本默认设到 3600 秒（1 小时），通常没问题。如果 ETA 卡在 `--:--`：

- 看 elapsed —— 如果还没跑满 30 秒（窗口里只有 1 个样本），正常，再等等就好
- 如果跑了几分钟还是 `--:--`，说明每次请求都耗时极长（超过 1 小时）—— 这本身就异常，先排查超时

### context overflow 频繁触发

正常 —— spec 默认是"打满即停，开新会话"。500 次 × (1.3K + 0.3K) ≈ 800K tokens，加上 28K 初始 prompt，会很快超出大多数模型的 context window。Per-session breakdown 表里会看到多个 session，每个 ~几十到一百多 turn。

**如果完全不想触发 overflow**，调小负载：

```bash
--initial-avg-tokens 5000 --initial-max-tokens 15000 \
--append-avg-tokens 500 --append-max-tokens 1500
```

### 中途想停

`Ctrl+C` 一次。脚本会跑完当前请求后写出标了 `**INCOMPLETE: N/500**` 的部分报告。再按一次强退则没有报告。

### 连续 3 次非 overflow 失败后会话重置

设计如此 —— 避免 prompt 被服务端搞坏（比如内部错误把 messages 状态弄乱）后无限重试。如果是网络抖动，重试一两次往往能恢复，不会触发重置。

### 测试很长但只想看进度

进度条 + 实时命中率会刷新在终端底部（rich Live）。Log 消息（overflow notice、WARN）从 Live 区域上方滚动，不会打断进度展示。如果终端不支持 ANSI（如 CI），rich 会自动退化为单次最终输出。

## 已知限制

- **只支持 OpenAI 格式**：Anthropic 的 `cache_read_input_tokens` 和 Gemini 的字段名都不一样，初版不做适配
- **粗略 token 估算**：用 ~4 chars/token 生成 prompt，分布会和真实分词器略有偏差；但命中率统计完全基于服务端返回的 `usage.prompt_tokens` 真值，估算不影响结果
- **不并发**：多轮对话内在串行；"并发会话各自的缓存命中"是另一个测试场景，这个脚本不覆盖

详见 [design doc 的"非目标"段](/docs/design/prompt-cache-hit-rate-script.md)。
