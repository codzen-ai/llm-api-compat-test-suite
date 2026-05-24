# 业务容量压测使用手册(RPM/TPM)

[scripts/capacity_probe.py](/scripts/capacity_probe.py) 是一个独立脚本,用来回答两个相关但不同的问题:

1. **"这个账号能撑多少 RPM / TPM?"** —— `probe` 子命令,摸账号上限
2. **"我的业务能撑多少?"** —— `capacity` 子命令,在给定请求形态下找稳态吞吐 + 瓶颈

设计动机与决策见 [docs/design/capacity-probe-script.md](/docs/design/capacity-probe-script.md)。

## 为什么不是 cache_hit_rate 那样的单命令?

`probe` 摸的是**服务端硬上限**(账号配额墙在哪);`capacity` 测的是**给定业务请求形态下能跑多快**(可能远低于墙,因为 RPM × 单请求 token 数已经够大)。两种模式的工作负载、收敛条件、报告语义都不同,塞同一命令会乱。

正确的顺序:**先 `probe` 摸墙,再 `capacity` 验证实际可用容量。**

## Quick Start

### 一、`probe` —— 摸账号上限

```bash
uv run python scripts/capacity_probe.py probe \
    --config config.mgtv.yaml --provider mgtv --model glm-5 \
    --max-context 32000
```

跑完后看:

```
reports/capacity_probe/{时间戳}_probe_{provider}_{model}/summary.md
```

`summary.md` 顶部的 **Headline** 直接给出 RPM / TPM 数值与来源(`header` / `measured` / `bounded_by_rpm`)。

### 二、`capacity` —— 测业务容量

```bash
uv run python scripts/capacity_probe.py capacity \
    --config config.mgtv.yaml --provider mgtv --model glm-5 \
    --avg-input-tokens 1000 --avg-output-tokens 200
```

`--avg-input-tokens` / `--avg-output-tokens` 从**你真实业务日志的 P50** 取,不要拍脑袋。报告给出稳态期间的 RPM / TPM(分 input/output)与瓶颈判定(`RPM` / `TPM` / `MIXED` / `NO_429`)。

## 常见场景

### 1. 第一次摸新账号

```bash
uv run python scripts/capacity_probe.py probe \
    --config config.mgtv.yaml --provider mgtv --model glm-5 \
    --max-context 4000
```

`--max-context 4000` 用小一点的上下文窗口可以让 Stage 2 的请求更轻、不浪费 token 配额。如果运气好(像 mgtv 一样),Stage 0 就能从响应 header 里直接读到 RPM 上限,Stage 1 自动跳过,只跑 Stage 2 验证 TPM。

### 2. 验证业务可以撑住目标流量

业务 P50 是 1500 input + 300 output:

```bash
uv run python scripts/capacity_probe.py capacity \
    --config config.mgtv.yaml --provider mgtv --model glm-5 \
    --avg-input-tokens 1500 --avg-output-tokens 300 \
    --start-rpm 20
```

控制器会从 `--start-rpm` 起步,1.5× 加速 / 0.7× 减速,直到 3 分钟稳定停在 1-10% 的 429 率,然后报告稳态吞吐。

### 3. 跨 provider 对比

同 seed、同业务形态跑两次 `capacity`:

```bash
uv run python scripts/capacity_probe.py capacity \
    --config config.mgtv.yaml --provider mgtv --model glm-5 \
    --avg-input-tokens 1000 --avg-output-tokens 200 --seed 42

uv run python scripts/capacity_probe.py capacity \
    --config config.other.yaml --provider other --model X \
    --avg-input-tokens 1000 --avg-output-tokens 200 --seed 42
```

输入分布完全一致,稳态 RPM/TPM 可直接对比。**注意**:对比的是"在同样请求形态下哪家撑得住更高吞吐",不是"哪家配额上限更高"——后者用 `probe` 测。

### 4. 长跑(超过默认 10 分钟上限)

默认 `--max-duration-minutes 10` 是安全阀,防止连续 429 触发上游 abuse detection。需要更长必须显式 `--allow-long-run`:

```bash
uv run python scripts/capacity_probe.py capacity \
    --config ... --avg-input-tokens 1000 --avg-output-tokens 200 \
    --max-duration-minutes 30 --allow-long-run
```

硬上限 120 分钟。

## 报告内容速览

每次跑产出一个目录,例如 `reports/capacity_probe/20260524_153012_probe_mgtv_glm-5/`:

| 文件 | 用途 |
|---|---|
| `summary.md` | 人类可读总结(headline + 瓶颈判定 + 逐分钟/逐阶段表) |
| `per_request.csv` | 每次请求一行:ts / 耗时 / status / token 数 / 分类 / Retry-After |
| `per_minute.csv` | (capacity only)逐分钟聚合,用于绘图/反算 |
| `config.json` | 全部入参 + git commit + 起止时间戳,可重现性凭据 |

## 怎么读结果

### probe 模式 headline 例子

```
- RPM limit: **60** (source: header)
- TPM limit: **61,002** (source: bounded_by_rpm)
```

- `source: header`:从 `x-litellm-key-rpm-limit` 等响应头直接读到,**精确可信**
- `source: measured`:压力测试时撞到 429 测出来的,**精确可信**
- `source: bounded_by_rpm`:Stage 2 没撞到 TPM 429,这只是**下界**——真实 TPM 上限可能高很多,或者 provider 根本没启用 TPM 限制
- `source: unknown`:壹个都没拿到,大概率是 stage 没跑完或全程失败

如果 summary.md 出现"Caveats"小节,**一定要看**——通常是"Stage 没撞到 429,客户端并发是瓶颈",需要你重跑或调整。

### capacity 模式 headline 例子

```
- Sustained throughput: **48.0 RPM** / **57,600 TPM** (input 48,000 + output 9,600)
- Bottleneck: **RPM** (RPM-triggered 82% · TPM-triggered 4% · Unclassified 14%)
- Converged: **yes**
```

- 稳态 throughput 是过去 N 分钟(由 `--steady-state-minutes` 控制,默认 3)的均值
- Bottleneck 看 429 来源占比:≥70% 一边 → 该边瓶颈;否则 `MIXED`
- `NO_429` 表示稳态期间根本没 429 —— 说明你给的目标 RPM 太保守,可以加大 `--start-rpm` 重跑

### `Converged: no` 怎么办

意味着 `--max-duration-minutes` 用完时控制器还在抖动,没找到稳态。最后一分钟的数据会被当作"最佳近似"报告出来,但**不要把它当稳态数据用**。处理:

1. 调大 `--max-duration-minutes`(配合 `--allow-long-run`)
2. 检查是否是某分钟 Retry-After 太长导致下一分钟空跑 → 看 `per_minute.csv` 的 `n_requests` 列

## 常见坑

### 1. probe 测出的 TPM 比 RPM × tokens 还小

**这是 client 并发不够,不是服务端限额**。典型场景:模型单请求 latency 20+ 秒,Stage 2 默认 10 个 worker 60 秒内顶多发 30 个,撞不到 RPM=60 的墙,自然也撞不到 TPM。

修复:启用更大的 max-context 让单请求更"贵",或者(将来要做)加 `--workers` 把并发拉上去。

### 2. probe 跑完账号被锁了一阵子

正常。Stage 2 通常会触发若干次 429,每次带 `Retry-After: 60`(或类似)。脚本严格遵守,但你下次再跑要等冷却结束。

### 3. capacity 模式的 `--start-rpm` 选什么

- 不知道账号上限 → 默认 `10`,控制器自己往上爬
- 知道 `probe` 给的 RPM 上限,且想节省时间 → 设为该值的 80%(避免一上来就撞墙、被 Retry-After 拖一分钟)

### 4. 跨时间对比不稳定

服务端负载、时段、灰度有波动。可重现性只保证**输入分布完全一致**(同 seed → 同 token 长度序列),不保证测出来的数字稳定。要做严肃对比,**同一时间段连续跑**。

### 5. 中文 prompt 的 token 数算不准

脚本内部用 `~4 chars/token` 估算来**生成**填充文本,但所有报告统计都用**服务端返回的真值**(`usage.prompt_tokens` / `completion_tokens`)。所以即使估算偏差,报告数字仍然精确。中文场景估算会偏短(中文 ~1.5 chars/token),实际请求的 token 数会比 `--avg-input-tokens` 偏小,但报告里都是真值,不影响结论。

## 与 cache_hit_rate 的关系

两个脚本共用 [scripts/_common.py](/scripts/_common.py)(`Sampler` / `make_text_for_tokens` / `select_target` / `build_headers` 等),都是独立于 pytest 的长跑工具。但它们测的是**完全不同的维度**:

| 脚本 | 测什么 | 单 (provider, model) 一次跑多久 |
|---|---|---|
| `cache_hit_rate.py` | prompt cache 命中率 | 1.5-3 小时 |
| `capacity_probe.py probe` | 账号 RPM/TPM 配额上限 | 2-10 分钟 |
| `capacity_probe.py capacity` | 业务形态下的稳态吞吐 + 瓶颈 | 5-30 分钟 |

互补,不重叠。
