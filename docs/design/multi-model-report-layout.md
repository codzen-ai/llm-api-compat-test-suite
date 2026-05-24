# 多 Model 报告分层与拆分

## 背景与问题

[config.yaml](/config.yaml) 单 provider 下已经允许声明多个 model；[conftest.py](/conftest.py) 的 `pytest_generate_tests` 也确实会对每个 model 参数化测试。但**测试结果的产物完全没有按 model 分层**——多 model 跑下来的目录长这样：

```
reports/20260523_142817/
├── summary.md
└── logs/
    ├── tests__openai_compat__test_chat_basic.py__TestChatBasic__test_simple_message[glm-5.1].log
    ├── tests__openai_compat__test_chat_basic.py__TestChatBasic__test_simple_message[glm-4.5].log
    ├── tests__openai_compat__test_chat_stream.py__TestChatStream__test_stream[glm-5.1].log
    └── …
```

model 名只通过参数化 id 的 `[glm-5.1]` / `[glm-4.5]` 后缀体现在文件名里，没有 `logs/glm-5.1/`、`logs/glm-4.5/` 这种分层。

更严重的问题在 `summary.md`：[report.py:130-134](/src/report.py#L130-L134) 按 `node_id.split("/")[1]`（即 `openai_compat`/`anthropic_compat`/`gemini_compat`）分组，**不按 model 分组**——两个 model 的结果搅在同一张表里，只能靠 `[xxx]` 后缀区分。

如果只是排序问题倒还好处理，但还有一个更本质的问题：**不同 model 的 capability 集合通常不同**。一个 model 支持 streaming + tools + vision，另一个只支持 streaming——`pytest_collection` 阶段对每个 model 各自过滤 capability，两个 model 实际跑的 test 集合就是两个不同的子集。这时候把它们放在同一份 summary 里：

- 一张 "By Model" 聚合表会出现 pass rate 跨 model 横比，但分母不同，比较是误导的
- 同一个 test 在 model A 的表里出现、在 model B 缺席（被 capability filter 掉），单文件里这种"缺席"无法可视化，只能让读者自己去对比行数

**目标**：让多 model run 的报告产物**结构上承认**"每个 model 跑的是不同子集"这个事实——每个 model 一份独立 summary，反映该 model 自己的测试范围；顶层 index 概览模型清单，但不强行做跨 model 通过率对比。

## 设计决策

### 1. 每个 model 一份 summary

```
reports/20260523_142817/
├── index.md                 ← 顶层索引：model 清单 + 各 model 链接
├── glm-5.1/
│   ├── summary.md           ← 该 model 自己的测试报告（结构同当前 summary.md）
│   └── logs/
│       ├── tests__openai_compat__test_chat_basic.py__TestChatBasic__test_simple_message.log
│       └── …
└── glm-4.5/
    ├── summary.md
    └── logs/
        └── …
```

**理由：**

- 每个 model 的 summary 只展示该 model 的真实测试集合——pass/fail/skip 都基于自己的分母，不会被另一个 model 的 capability 集合污染
- "model A 没跑的 test"自然消失在 A 的报告里，不需要在表里塞"missing"或"N/A"占位
- model 间没有任何报告耦合——单独打开 `glm-5.1/summary.md` 就能完整理解该 model 的结果，不需要知道这次 run 还跑了哪些其它 model
- 与 [multi-provider-support.md](/docs/design/multi-provider-support.md) 的"每 provider 一份 summary"思路一致——形成 `{provider}/{model}/summary.md` 的统一两级分层

### 2. 顶层 `index.md` 是跳板，不是 dashboard

`index.md` 只列出"这次 run 跑了哪些 model + 各自的 passed/failed/skipped 计数 + 链接"，**不计算 pass rate、不做横向对比**。

```markdown
# Run 20260523_142817

Provider: mgtv (https://aigc-llm.mgtv.com, openai)

| Model | Profile | Capabilities | Passed | Failed | Skipped | Error | Summary |
|-------|---------|--------------|--------|--------|---------|-------|---------|
| glm-5.1 | gpt-5.4-mini | 12 | 42 | 3 | 5 | 0 | [summary](glm-5.1/summary.md) |
| glm-4.5 | gpt-5.4-mini | 8  | 28 | 2 | 0 | 0 | [summary](glm-4.5/summary.md) |
```

- "Capabilities" 列展示**该 model profile 声明的 capability 数量**——读者能直接看出"两个 model 跑的不是一个量级的测试集"，自然就不会去做"42/50 vs 28/30 谁更好"的误读
- 不出 pass rate 列：刻意省去，避免诱导跨 model 横比
- 想看具体细节 → 点 summary 链接

**理由**：index 的职责就是"告诉你这次 run 涉及哪些 model"以及"提示它们不在同一个测试集上"。聚合做得越多越像 dashboard，越容易让读者忽略 capability 差异。

### 3. 每个 model 的 `summary.md` 结构与当前一致

单 model summary 的内容**不需要新增 Model 列**——文件路径已经体现了 model 维度。结构沿用当前 [src/report.py](/src/report.py) 的 `generate_summary` 输出：

- Configuration 表（base_url / api_format / auth_type / verify_ssl）
- Models 表（**只剩这一行**——其实可以省略，但保留有助于读者快速看到该 model 用的是哪个 profile snapshot）
- Summary 总数
- 按 api_format 分组的明细表（| Test | Status | Duration | Log |，不变）
- Test Details
- Failure Details

**理由**：单 model 单文件意味着所有现有的 summary 渲染逻辑可以**原样复用**，只是 `ReportCollector` 实例从"全局一个"变成"每 model 一个"。改动成本最小。

### 4. Logs 直接放在 model 子目录下

`reports/{ts}/{model}/logs/{slug}.log`，**没有额外的 `model` 中间层**——因为外层已经是 model 目录。`slug` 沿用现有的 `nodeid.replace("/", "__").replace("::", "__")`，但**去掉末尾 `[model-name]` 段**（参数化 id 后缀，现在多余）。

### 5. `TestResult` 加 `model_name`；多个 `ReportCollector` 实例

[src/report.py:60-73](/src/report.py#L60-L73) 的 `TestResult` 加 `model_name: str`，方便 collector 反查（虽然每个 collector 只装一个 model 的数据，但留这个字段使 collector 不需要在"加入 result"时先校验 model 是不是匹配）。

[conftest.py](/conftest.py) 的 `_collector` 全局单实例 → `_collectors: dict[str, ReportCollector]`（key = model name）。`pytest_configure` 时按配置中的 model 顺序预创建空 collector 与子目录；`pytest_runtest_makereport` 从 `item.callspec.params["resolved_model"]` 取 model name 路由到对应 collector；`pytest_sessionfinish` 遍历 collectors 各自调 `generate_summary` 写入自己的目录，再生成顶层 `index.md`。

### 6. 配置中存在但未跑出任何 result 的 model 也要在 index 出现

某个 model 的所有 capability marker 都被 profile 拒掉 → 0 个 test 跑过 → 该 model 在 `index.md` 中仍要出现（全 0），并且仍要写一份 `{model}/summary.md`（即使表是空的）。

**理由**：配置里写了的 model 在产物里完全消失会让人怀疑"是不是没跑"。让"配了但没跑出东西"在报告里可见，是个明显的"配置/profile 不匹配"信号。

### 7. 与 multi-provider 设计的组合

[multi-provider-support.md](/docs/design/multi-provider-support.md) 的产物形如 `reports/{ts}/{provider}/{summary.md + logs/}`。本设计落地后 + multi-provider 落地后的组合产物是：

```
reports/{ts}/
├── index.md                          ← 顶层：列 provider × model 全表
├── mgtv/
│   ├── index.md                      ← provider 内：该 provider 下的 model 清单
│   ├── glm-5.1/
│   │   ├── summary.md
│   │   └── logs/
│   └── glm-4.5/
│       ├── summary.md
│       └── logs/
└── deepseek/
    ├── index.md
    └── deepseek-chat/
        ├── summary.md
        └── logs/
```

每一层 `index.md` 都是"列下一级 + 链接"，不做跨层聚合。两份方案**正交**：

- multi-provider 单独落地 → `reports/{ts}/{provider}/{summary.md + logs/}` （现在的方案）
- 本方案单独落地 → `reports/{ts}/{model}/{summary.md + logs/}` + 顶层 `index.md`
- 都落地 → 上面两级嵌套

实施顺序无强依赖；同时在审时建议 multi-provider 先合（改动 conftest 范围更大），本 PR 在其上叠加。

## 影响范围

| 文件 | 改动性质 |
|------|----------|
| [conftest.py](/conftest.py) | 单 `_collector` → 多 collector（按 model 路由）；log 路径加 `{model}/` 段；`pytest_runtest_makereport` 把 model 写入 `TestResult`；`pytest_sessionfinish` 写每 model summary + 顶层 index.md |
| [src/report.py](/src/report.py) | `TestResult` 加 `model_name`；新增 `IndexCollector`（或函数）生成顶层 index.md；`generate_summary` 接口基本不动 |
| [unit_tests/test_report.py](/unit_tests/test_report.py) | 现有断言基于"一个 collector 一份 summary"，多 model 时为多个独立 collector，原断言对单 model 用例继续生效；新增多 model 渲染与 index.md 内容的断言 |
| [CLAUDE.md](/CLAUDE.md) | "Data flow" 段落更新 reports 目录示意；新增 "Multi-model" 小节 |

## 实施步骤（TODO + Checkpoint）

每步完成后停下来 review。

### TODO 1：`TestResult` 加 `model_name`

**产出：**

- [src/report.py](/src/report.py) 的 `TestResult` 加 `model_name: str` 字段
- [unit_tests/test_report.py](/unit_tests/test_report.py) 现有用例补 `model_name=` 参数（pre-release，无 back-compat 包袱，直接改）

**✋ Checkpoint 1**：`uv run pytest unit_tests/test_report.py -v` 全绿；`uv run pyright` 无新增告警。

---

### TODO 2：conftest 单 collector → 多 collector

**产出：**

- 删除 `_collector` 单实例，引入 `_collectors: dict[str, ReportCollector]`（key = model name）
- `pytest_configure` 末尾遍历 `_resolved_models` 预创建 collector：每个 collector 的 `report_dir` 直接指向 `_report_dir / model.name`；同时 mkdir `{model}/logs/`
- model name 做 slug 化：`re.sub(r"[^a-zA-Z0-9._-]", "_", name)`；空字符串或全非法 → `pytest.UsageError`
- collector 路由辅助函数：`_collector_for(item) -> ReportCollector`，从 `item.callspec.params["resolved_model"]` 取 name

**关键点：**

- `item.callspec` 在参数化 item 上才存在；非参数化的 item（如 unit_tests）不会进入 compat 流程，但兜底 `if not hasattr(item, "callspec"): return`

**✋ Checkpoint 2**：单 model config 在这一步应当仍能跑通——只是产物路径从 `reports/{ts}/{summary.md + logs/}` 变成 `reports/{ts}/{model}/{summary.md + logs/}`。

---

### TODO 3：log 写盘路径分层 + 文件名去后缀

**产出：**

- [conftest.py:322-344](/conftest.py#L322-L344) 的 log 写盘路径改为 `_report_dir / model_name / "logs" / safe_name + ".log"`
- `safe_name` 去掉末尾 `[model-name]` 段——既然路径已经分目录，文件名里不需要再带 model（参数化 id 末段去除可以用 `re.sub(r"\[[^\]]+\]$", "", safe_name)`）
- log 内部 header 的 `Model:` 字段保留——脚本不依赖路径解析也能拿到 model

**✋ Checkpoint 3**：跑一遍真实的多 model config，确认 `reports/{ts}/{model}/logs/` 子目录被创建、文件名不再带 `[...]` 后缀、log 内 header 仍含 Model 字段。

---

### TODO 4：`pytest_runtest_makereport` 路由到正确 collector

**产出：**

- [conftest.py:350-392](/conftest.py#L350-L392) 不再写入全局 `_collector`，而是先用 `_collector_for(item)` 找到对应 collector
- 取出 `resolved_model.name` 写入 `TestResult.model_name`

**✋ Checkpoint 4**：双 model run 后，两个 collector 各自只包含自己 model 的 result。可以加一个 assertion log（debug 期间）验证。

---

### TODO 5：`pytest_sessionfinish` 输出多 summary + 顶层 index.md

**产出：**

- `pytest_sessionfinish` 遍历 `_collectors`，每个调 `generate_summary(provider, resolved_models=[该 model])` 输出到 `reports/{ts}/{model}/summary.md`
- 即便 collector.results 为空也照样输出 summary（不要"results 空就跳过"）——配合决策 §6
- 新增 `_write_index(report_dir, provider, resolved_models, collectors)`：
  - Provider 行（base_url、api_format）
  - Models 表：| Model | Profile | Capabilities | Passed | Failed | Skipped | Error | Summary |
  - "Capabilities" 列 = `len(resolved_model.capabilities)`
  - 不出 pass rate 列
- 终端打印调整：列出顶层 index.md 路径 + 各 model summary 路径（多行）

**✋ Checkpoint 5**：双 model run 完，`reports/{ts}/index.md` 存在且链接全部有效；每个 model 子目录有自己的 summary.md；点链接能跳到对应 logs。

---

### TODO 6：单元测试

**产出：**

- [unit_tests/test_report.py](/unit_tests/test_report.py) 加：
  - 多 collector 多 summary 的写入断言（用 tmp_path）
  - `_write_index` 的内容断言（不含 pass rate 字样、Capabilities 列存在）
  - 空 collector（model 配了但无 result）也写出 summary
- 确认现有单 model 测试不回归

**✋ Checkpoint 6**：`uv run pytest unit_tests/ -v` 全绿；`uv run pyright`、`uv run ruff check .` 无新增告警。

---

### TODO 7：文档更新

**产出：**

- [CLAUDE.md](/CLAUDE.md) 的 "Data flow" 段落更新 reports 目录示意（多 model 形态）
- 新增 "Multi-model reports" 小节，简述"每 model 一份 summary、capability 集合不同所以不做横比"，链接本文件

**✋ Checkpoint 7**：让一个不熟悉的人读 CLAUDE.md + 跑一次双 model run 后能理解为什么 index.md 没有 pass rate 列。

## 非目标（明确不做）

1. **不做单一聚合 summary**——决策 §1 的核心是承认 model 间测试集合不同；任何"合在一起的 By-Model 表"都会重新引入"不同分母的横比"问题
2. **不做跨 model 语义对比**（"glm-5.1 vs glm-4.5 在 streaming 上响应差异"）——这是 evaluation 任务，超出 compat 测试套件范围
3. **不在 `index.md` 出 pass rate**——刻意避免诱导跨 model 横比；想看通过率就点进单 model summary
4. **不引入 `--model` 多值过滤**（如 `--model glm-5.1 --model glm-4.5`）——当前单值 `--model` 已能选一个；多 model 全跑就是 YAML 模式的默认行为
5. **不动 profile resolution / capability filter 逻辑**——本 PR 只改 report 产物结构

## 设计权衡记录

- **为什么不保留"单文件 + By-Model 聚合表"作为可选模式**：选项越多，读者越容易选错（默认看哪个？）。承认"每 model 独立"是本设计的核心立场，混合模式会模糊这个立场。

- **为什么 `index.md` 出 Capabilities 数量而不是 capability 列表**：列表会很长且大部分名字对非作者读者意义不大；数量能直观传达"测试集大小不同"这个最关键的信息。想看完整 capability → 点进 model 的 `summary.md`（其中 Models 小节会展示 profile 路径，可进一步追溯）。

- **为什么 `model_name` 字段仍要加（即使 collector 已按 model 分隔）**：collector 本身可能在测试或调试代码里被合并/dump，留一个显式字段让 result 自描述更稳健。一行 `model_name: str` 的代价远小于"以后要做跨 collector 比较时再加"。

- **为什么空 collector 也写 summary**：配置里写了的 model 在产物中完全消失是"silent failure"——典型的"配置错了但没人发现"陷阱。强制写出空 summary 让"配了但 0 个 test 跑"成为一个能被审计的事实。

- **为什么不区分 `pass rate 含 skip 与否`**：本设计在 `index.md` 索性不出 pass rate；如果 model 内 summary 未来要补 pass rate（不在本 PR 范围），同样建议 `passed / (passed + failed + error)`、skip 单列不进分母，理由是 skip 来自 capability mismatch，本质上"不该跑"，不应稀释能力评估。
