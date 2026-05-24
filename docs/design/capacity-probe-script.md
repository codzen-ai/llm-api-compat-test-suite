# 业务容量压测脚本(RPM/TPM)

> 这是**设计文档**(记录为什么这么做、决策与权衡)。如果你只想知道**怎么用**这个脚本,看 `docs/usage/capacity-probe.md`(待编写)。

## 背景与目标

供应商通常同时下发两条速率限额:**RPM**(requests/min) 与 **TPM**(tokens/min)。账号实际能撑的业务量是这两条的耦合结果:

```
有效容量 (tpm) = min(TPM_limit, RPM_limit × 业务平均每请求 token 数)
```

举例:RPM=60, TPM=100k,业务平均 1200 tok/请求 → `min(100k, 72k) = 72k tpm`,瓶颈是 RPM 而非 TPM。

现有 compat 套件覆盖**单请求**的功能正确性 + TTFT/TPOT,完全没有触及**速率限额**这个维度。本脚本填这个空。

**目标**:针对一个 (provider, model) 二元组,在两种模式下产出可重现的容量报告:

1. **probe 模式**:摸出账号的 `RPM_limit` 与 `TPM_limit` 数值
2. **capacity 模式**:在给定的业务请求形态(平均 input/output token 数)下,跑出该账号实际能撑的稳态 RPM 与 TPM,并标注瓶颈是哪一条

## 设计决策

### 1. 独立 script,不走 pytest

放在 `scripts/capacity_probe.py`,CLI 入口 `uv run python scripts/capacity_probe.py ...`。

**理由**(与 [prompt-cache-hit-rate-script.md](prompt-cache-hit-rate-script.md) 第 1 节完全一致):

- 单次运行 ≥ 几分钟到几十分钟,与秒级 compat 测试节奏不匹配
- 不需要 capability gating / per-model parametrization / profile resolution
- 产出"一组统计量",不是 N 个 pass/fail,套 pytest report 会扭曲
- 主动制造 429,会污染常规测试的 reports

**复用项**:沿用 `SuiteConfig` / `ProviderConfig` 解析 `config.yaml`(参见 [cache_hit_rate.py](/scripts/cache_hit_rate.py) 中 `select_target`、`build_headers` 的实现,直接搬过来即可)。**不复用** `LoggingHttpClient` —— 它每次请求累加到 `records`,容量压测会跑几千上万次请求,内存会爆。

### 2. 两个独立子命令:`probe` 与 `capacity`

CLI 形态:

```bash
# 摸限额上限(--max-context 必填:Stage 2 需要构造接近上下文窗口上限的请求)
uv run python scripts/capacity_probe.py probe \
    --config config.yaml --provider mgtv --model glm-5.1 \
    --max-context 32000

# 在业务请求形态下测稳态容量
uv run python scripts/capacity_probe.py capacity \
    --config config.yaml --provider mgtv --model glm-5.1 \
    --avg-input-tokens 1000 --avg-output-tokens 200

# 通用安全开关(两种模式都有):
#   --max-duration-minutes N   默认 10;到点后即使未收敛也优雅停止
#   --allow-long-run           解锁 --max-duration-minutes 上限,需显式传
```

**理由**:两种模式的工作负载、收敛条件、输出语义都不同,塞同一条命令里靠 flag 切换会让参数表混乱。子命令把"摸墙"和"测吞吐"在认知上清晰分开。

### 3. probe 模式:三段式探测

```
Stage 0 ── Cheap discovery ───────────────────────────────────────
  发 1 次最小请求,扫描响应 header:
    - x-litellm-key-rpm-limit   → 直接拿到 RPM_limit(litellm 网关)
    - x-ratelimit-*-limit-requests → 拿到 RPM_limit(OpenAI 风格)
    - x-ratelimit-*-limit-tokens   → 拿到 TPM_limit
  若两者都拿到 → 跳过 Stage 1/2,直接进入瓶颈分析
  若只拿到部分 → 仅跑缺失维度的 Stage
  若一个都没有 → 走完整 Stage 1+2
  mgtv 实测:RPM 能从 header 拿到,TPM 不能 → 跳 Stage 1,跑 Stage 2

Stage 1 ── RPM probe ─────────────────────────────────────────────
  请求形态:input ~50 tok, max_tokens=10(极小,确保 token 不是瓶颈)
  策略:从 concurrency=2 开始倍增,直到出现持续 429
  收敛判定:在某并发下,连续 60s 滚动窗口的 429 率 ∈ [5%, 50%]
           → 取该窗口内的成功请求/分钟作为 RPM_limit
  超时:任一并发持续 3 分钟未撞 429 则放弃,记 RPM_limit=∞ + 已尝试的最高并发

Stage 2 ── TPM probe ─────────────────────────────────────────────
  请求形态:input 顶到 context 上限(从 model_profiles 读 max_context,
           或 CLI --max-context 指定),max_tokens 也拉满
  策略:用 Stage 1 的 RPM_limit 作为发送速率上限(避免又撞 RPM)
  收敛判定:同上 60s 窗口 429 率落入区间
           → 取该窗口的 Σ(prompt_tokens + completion_tokens) / 分钟作为 TPM_limit
  特殊情况:若以 RPM_limit × max_tokens_per_request 都不足以撞到 TPM,
           说明 TPM 高到 RPM 永远先打死,记 TPM_limit=`>=` 这个下界值
```

**为什么不是单段式 ramp-up**:RPM 和 TPM 是两条不同的墙,单段只能撞到先到的那一条。要分别拿到数值,必须用不同的请求形态分两次打。

**为什么用滚动窗口而不是固定时长**:供应商常允许短时 burst(比如令牌桶 + 突发额度),刚启动的第一分钟测出来的"通过率"偏高。要等系统进入稳态。

### 4. capacity 模式:用业务形态算实际吞吐

输入:`--avg-input-tokens` / `--avg-output-tokens` (从用户业务日志的 P50/P90 取)。

```
1. 用截断正态采样每个请求的 input/output token 数(沿用 cache_hit_rate 的 Sampler)
2. 起 N 个并发 worker,从一个目标 RPM 开始(初值 = probe 测到的 RPM_limit;
   若没 probe 数据,从 CLI --start-rpm 起,默认 10)
3. 用 token bucket 控制全局发送速率到目标 RPM
4. 每分钟做一次决策:
     - 若 429 率 < 1%   → 目标 RPM × 1.5 继续探
     - 若 429 率 ∈ 1-10% → 维持当前 RPM,累计稳态数据(连续 3 分钟)
     - 若 429 率 > 10%  → 目标 RPM × 0.7 回退
5. 终止:稳态 3 分钟数据收集完成 / 累计运行 30 分钟 / SIGINT
6. 报告:稳态期间的 mean RPM、mean TPM(分 input/output)、瓶颈判定
```

**瓶颈判定逻辑**:

- 拿稳态数据里的 429 错误体逐条分类(见 §6),统计 RPM-triggered vs TPM-triggered 比例
- 占比 > 70% 的那条 = 瓶颈;否则 = "RPM/TPM 接近平衡"

### 5. 并发模型:asyncio + httpx.AsyncClient

cache_hit_rate 是串行(多轮对话天然顺序),容量测试必须并发。

- 单个 `httpx.AsyncClient(limits=Limits(max_connections=N*2))`
- N 个 worker 协程从共享 `asyncio.Queue` 取请求(请求由 dispatcher 按目标 RPM 投递)
- Dispatcher 用 token bucket(`asyncio.sleep` 控制投递间隔 = `60/target_rpm`)而非每个 worker 自带间隔 —— 全局速率精度更高
- N 的取值:`max(target_rpm × p99_latency_sec / 60 × 2, 10)`,留 2× 余量避免 worker 不够时被 sleep 拖慢实际 RPM

**为什么不用 threading**:几千请求并发,GIL + httpx 同步版本性能上不去;asyncio 单进程能轻松撑住 1000+ 并发。

### 6. 429 区分:RPM 限制 vs TPM 限制

经对 mgtv(litellm 网关,路由到 dashscope)实测,识别优先级如下:

**Tier 1 — `rate_limit_type` header(litellm 风格,首选)**

```
rate_limit_type: requests   → RPM 触发
rate_limit_type: tokens     → TPM 触发
```

litellm 给出的干净机读字段。mgtv 实测确认 `requests` 取值;`tokens` 取值待真实触发 TPM 时验证。

**Tier 2 — `x-ratelimit-*-remaining-*` header**

```
x-ratelimit-remaining-requests=0           → RPM(OpenAI 原版)
x-ratelimit-api_key-remaining-requests=0   → RPM(litellm,带 api_key- 中缀)
x-ratelimit-remaining-tokens=0             → TPM
x-ratelimit-api_key-remaining-tokens=0     → TPM
```

匹配时对 header 名做 case-insensitive 包含判断:`"remaining-requests" in name` / `"remaining-tokens" in name`,自动覆盖 OpenAI 与 litellm 两种命名。

**Tier 3 — body `error.message` 关键词(兜底)**

```python
RPM_KEYWORDS = (
    "limit type: requests",      # litellm
    "requests per minute", "rpm", "request rate", "too many requests",
)
TPM_KEYWORDS = (
    "limit type: tokens",        # litellm
    "tokens per minute", "tpm", "token rate", "token quota",
    "token limit per",
)
```

都识别不出 → 记为 `429_unclassified`,在报告里单列;首次出现时打 WARNING 并保存一份原始响应到 `reports/.../unclassified_429_sample.json` 供人工核对。

**这一步是脚本最容易出兼容性问题的环节**,设计上必须做到"识别不出也不崩,如实记录原文"。

**Retry-After**:mgtv 实测会返回(60s),严格遵守。详见 §7。

**`reset_at` header**:litellm 还会返回 ISO 时间戳的 `reset_at`,可作为 `Retry-After` 缺失时的回退,但不强求支持。

### 7. 退避策略

- 429 响应若带 `Retry-After`(秒)→ 严格遵守
- 否则:`backoff = min(2^attempt × 0.5, 30)` 秒,最多重试 2 次
- 重试请求**计入** RPM 消耗(因为服务端通常这样算),但**不计入** TPM 消耗(失败请求未产生 token 计费)
- 5xx:同样指数退避重试 2 次;仍失败记为 `transient_failure`,不计入容量统计
- 网络异常(connection reset / timeout):同 5xx 处理

### 8. 请求形态生成

复用 cache_hit_rate 已有的:

- `LoremGenerator` / `make_text_for_tokens` 生成长文本
- `Sampler`(截断正态)采样每请求的 token 数

新增:`--max-tokens` flag 控制 `max_tokens` 参数(probe Stage 2 必须拉满,capacity 模式由 `--avg-output-tokens` × 1.2 上限自动设定)。

不引入 tokenizer,沿用 ~4 chars/token 估算;最终所有统计都用服务端返回的 `usage.prompt_tokens` / `completion_tokens` 真值。

### 9. 数据采集

每次请求(成功或失败)落一行:

```python
{
    "ts": float,                 # epoch seconds, dispatcher 投递时刻
    "elapsed_ms": float,
    "status": int,               # http status; 0 = network error
    "input_tokens": int,         # usage.prompt_tokens(失败=0)
    "output_tokens": int,        # usage.completion_tokens(失败=0)
    "classification": str,       # "ok" / "429_rpm" / "429_tpm" /
                                 # "429_unclassified" / "5xx" / "timeout" / "network"
    "retry_after_sec": float | None,
}
```

另起一份 per-second 聚合(滚动窗口算 429 率必需),内存里维护 deque,最大保留 600 秒(10 分钟)即可。

### 10. 输出

每次跑产出一个目录:`reports/capacity_probe/{timestamp}_{mode}_{provider}_{model_slug}/`

```
20260524_153012_probe_mgtv_glm-5.1/
├── summary.md           ← 人类可读总结(headline + 决策依据)
├── per_request.csv      ← 所有请求原始数据
├── per_minute.csv       ← 按分钟聚合:req_count, ok_count, 429_rpm,
│                          429_tpm, input_tokens, output_tokens
└── config.json          ← 入参 + git commit + probe 模式的两段中间结论
```

**summary.md headline**(probe 模式):

```markdown
# Capacity Probe Report (probe mode)

Run: 20260524_153012
Provider: mgtv | Model: glm-5.1
Git commit: <sha>

## Headline
- **RPM_limit: 60** (measured at concurrency=8, sustained 60s window, 23% 429 rate)
- **TPM_limit: ≥ 72,000** (could not exceed RPM bottleneck even at max prompt size;
  true TPM ceiling likely higher)
- Effective capacity for typical business request (1200 tok avg):
  `min(72k, 60×1200) = 72,000 tpm`,**RPM-bound**
```

**summary.md headline**(capacity 模式):

```markdown
## Headline
- Sustained throughput: **48 RPM / 57,600 TPM** (input 48k + output 9.6k)
  over 3-minute steady-state window
- 429 distribution: RPM-triggered 82%, TPM-triggered 4%, unclassified 14%
- **Bottleneck: RPM**
- Headroom: input tokens at 67% of TPM_limit;若把单请求平均 input 提到 2000 tok,
  TPM 也会成为瓶颈
```

终端只打 headline + 报告路径,不刷屏。运行过程中用 `rich.Live` 显示一个滚动窗口面板(当前 RPM / 当前 TPM / 当前 429 率 / 已运行时长)。

### 11. 可重现性

- `--seed` 默认 42,所有随机源(Sampler、Lorem 偏移)从同一 `random.Random(seed)` 派生
- **重要**:即使 seed 相同,实际测出的 RPM/TPM 仍会受服务端负载、时段波动影响。可重现性只保证**输入分布完全一致**,便于跨时间/跨供应商对比"在相同请求形态下,谁更扛得住"
- `config.json` 写入完整入参 + git commit + 起止时间戳

### 12. 错误处理与中断

- `KeyboardInterrupt`:停止 dispatcher,等所有 in-flight 请求结束(最多 30 秒),写 partial report,顶部标 `**INCOMPLETE: aborted at minute X**`
- 单请求 timeout:120 秒(同 cache_hit_rate)
- 启动前自检:如果 5 分钟内已有同 (provider, model) 的 probe 报告,提示"账号可能仍在 429 冷却期"并等待用户确认(避免连续猛打把账号锁掉)

## 实现步骤

### Step 1: 共享脚手架抽取
把 cache_hit_rate.py 里的 `select_target`、`build_headers`、`Sampler`、`LoremGenerator`、`_git_commit`、`_slugify` 等公共部分抽到 `scripts/_common.py`,两个脚本同时 import。同时迁移 cache_hit_rate 的 import。**这一步必须先做,否则 capacity_probe 会复制大量代码。**

### Step 2: capacity_probe 骨架
`probe` 和 `capacity` 两个子命令的 CLI 解析、配置加载、输出目录创建。先不发请求,验证 `--help` 和参数校验。

### Step 3: 并发请求核心
asyncio dispatcher + worker pool + token-bucket 限速 + per-request 数据采集。用本地 mock server(`asyncio.start_server` 起一个固定延迟返回假 usage 的玩具服务端)跑通,验证目标 RPM 与实测 RPM 误差 < 5%。

### Step 4: 429 分类器
实现 §6 的 header + body 双通道识别。针对 mgtv 真账号刻意打 429,采集几条 error 原文,补全 keyword 列表。

### Step 5: probe 模式收敛逻辑
Stage 1 ramp-up + Stage 2 大请求探 TPM + 60s 滚动窗口判定。先用 mock server 模拟 RPM=10/TPM=10k 验证逻辑收敛到正确值,再跑真账号。

### Step 6: capacity 模式自适应控速
§4 的 1.5×/0.7× 调节 + 3 分钟稳态收集。同样先 mock 再真跑。

### Step 7: 报告产出
summary.md / per_request.csv / per_minute.csv / config.json 四件套 + rich live 面板。

### Step 8: 文档
- 写 `docs/usage/capacity-probe.md`(参考 `docs/usage/cache-hit-rate.md` 的结构)
- 在 [CLAUDE.md](/CLAUDE.md) Commands 段加示例
- 本文档作为唯一的"为什么这么做"参考

## 非目标(明确不做)

1. **不做多模型对比报告** —— 一次跑一个 (provider, model),跨模型对比留给外部脚本/人工
2. **不做长期监控** —— 单次压测产出快照,不是常驻 collector;若要做账号容量随时间漂移,另起方案
3. **不支持 anthropic / gemini 格式**(初版) —— 429 响应结构、ratelimit header 命名都不同,留到第二阶段;接口预留 `extract_ratelimit_info(api_format, response)` 适配点
4. **不引入 tokenizer 依赖** —— 同 cache_hit_rate
5. **不做 SLO 断言** —— 测量工具,非 pass/fail
6. **不并发多账号** —— 一次只压一个 API key;并发多账号是"多租户容量规划",超出本脚本范围

## 已确认的决策(原"待确认")

以下四点在动手前已通过对 mgtv 真账号的探测(`/tmp/probe_mgtv.py`,一次性脚本,未入库)和取舍决策落定:

1. **鉴权**:`build_headers()` 现状已支持 mgtv(标准 `Authorization: Bearer ...`),无需特殊 header
2. **mgtv 的 429 格式**:litellm 网关,带 `rate_limit_type` header 与 `Retry-After: 60`,error body 英文且结构化("Limit type: requests. Current limit: 60, Remaining: 0...")—— §6 已据此细化为三层识别
3. **`max_context` 来源**:**CLI 必填 `--max-context`**,不读 profile —— probe 是一次性运维操作,显式传比依赖可能滞后的 profile 数据更安全
4. **安全开关**:**默认 `--max-duration-minutes 10`**;要长跑必须显式传 `--allow-long-run`。理由:连续打 30+ 分钟在部分供应商上可能触发 abuse detection,新手运行不该默认踩雷

## 实测发现(写入设计的额外约束)

probe mgtv 时顺带得到的事实,影响实现细节:

- **`x-litellm-key-rpm-limit` 在每次成功响应里都直接给出 RPM 上限** —— §3 因此新增 Stage 0,先廉价探测一次,能拿到的指标就不再用打压方式测
- **TPM 无对应 header**(`x-litellm-key-tpm-limit` 未出现)—— 不确定是 mgtv 未启用 TPM 限制,还是启用但不暴露。脚本必须按"可能不存在"处理:Stage 2 跑完若全程无 `rate_limit_type: tokens`,报告里写 `TPM_limit: not enforced or >= <下界>`
- **底层是阿里 dashscope**(`x-litellm-model-api-base: https://dashscope.aliyuncs.com/...`)—— 真实并发能力受 dashscope 限制,不是 mgtv 自有
- **litellm 是国内中转网关的常见技术栈** —— §6 的识别规则对其他 litellm-based 供应商应同样适用,设计有迁移价值
