# 多 Provider 同跑支持

## 背景与问题

[config.yaml](/config.yaml) 的 schema 已经把 `providers` 写成数组（`SuiteConfig.providers: list[ProviderConfig]`），但 [conftest.py](/conftest.py) 实际只用 `providers[0]`：

```python
if _suite_config and _suite_config.providers:
    _active_provider = _suite_config.providers[0]
    _active_models = _active_provider.models
```

之后 `pytest_collection_modifyitems`、`client` fixture、auth headers、`ReportCollector` 都基于这个单值 `_active_provider`。结果是用户即便在 `config.yaml` 里堆三个 provider，也只有第一个会被跑——其余被静默丢弃。

**目标**：把这个"配置允许多个、运行只取一个"的暗坑修掉，让一次 `pytest --config` 真正能跑完所有声明的 provider × model 组合，并产出可追溯的报告。

## 设计决策

### 1. 一次运行 = 一个 timestamp，多个 provider 子目录

```
reports/
  20260523_142817/                  ← 整个 run 共用同一个 timestamp
    mgtv/
      summary.md
      logs/
        tests__openai_compat__test_chat_basic.py__TestChatBasic__test_simple_message.log
        …
    deepseek/
      summary.md
      logs/
        …
    index.md                        ← 顶层索引，链接到各 provider summary
```

**理由**：

- 同一次 `pytest` 启动产生的所有结果在同一棵目录下，便于打包/对比/归档（"这一次 run 跑了哪些 provider"是一眼能看出来的）
- 每个 provider 一份独立 `summary.md`，避免把多个无关 provider 的 PASS/FAIL 表搅在一起；同时 provider 间没有报告耦合
- 不引入"主 summary 聚合所有 provider"的复杂度——`index.md` 只是个跳板，不重复表达每个 provider 的结果

### 2. Provider 名称必须唯一

`provider.name` 直接被用作目录名。两个同名 provider → `pytest.UsageError`。

目录名做 slug 化（`/`, 空格, 等替换为 `_`），但仍要求 `name` 字段本身不要塞奇怪字符——错误提示直接报"provider name must be a non-empty slug"。

### 3. Test 参数化维度：(provider, model) 联合

当前 `pytest_generate_tests` 只对 `resolved_model` 一个维度参数化（id = model name）。改造后：

- 新增一个 **`provider_model`** 间接 fixture，参数是 `(ProviderConfig, ResolvedModel)` 对
- Test id 形如 `[mgtv::glm-5.1]`、`[deepseek::deepseek-chat]`（用 `::` 分隔，避免和 model name 里可能有的 `-` 混淆）
- `provider_config` 和 `resolved_model` fixture 改为从 `provider_model` 派生

**为什么是联合维度而不是两个独立维度**：模型属于特定 provider，不存在"跨 provider 共用 model"的概念。两个独立 `params` 会笛卡尔积出大量无效组合（provider A 的 test 跑 provider B 的 model）。

### 4. api_format 过滤改在参数化阶段做

目前 `pytest_collection_modifyitems` 整体丢弃 api_format 不匹配的 test item。多 provider 下不能这样——一个 `tests/openai_compat/` 下的 test 在 openai provider 上要跑，在 anthropic provider 上不应该跑，但 item 本身不能丢。

**新做法**：在 `pytest_generate_tests` 里，根据 test 所在目录（`openai_compat` / `anthropic_compat` / `gemini_compat`）只把 api_format 匹配的 (provider, model) 对纳入参数化。如果一个 test 没有任何匹配 provider → 用空 params 让它被 deselected（pytest 内置行为）。

副作用：`pytest_collection_modifyitems` 里的 api_format 过滤完全删除；它的职责退化到"没有 provider 配置时全 skip"。

### 5. Provider 间相互独立、互不影响

- 一个 provider 的 HTTP 请求失败、auth 错误、网络抖动**不会**让其它 provider 的 test 失败
- 每个 (provider, model) 组合的 test 通过/失败/跳过都独立计入 *该 provider* 的 `ReportCollector`
- `pytest_sessionfinish` 时遍历 provider，逐个调 `generate_summary` 产出每个 provider 的 `summary.md`

### 6. Log 文件按 provider 分目录

当前路径：`reports/{ts}/logs/<safe_node_id>.log`

新路径：`reports/{ts}/{provider}/logs/<safe_node_id>.log`

`safe_node_id` 沿用现有的 `nodeid.replace("/", "__").replace("::", "__")` 逻辑；由于参数化 id 现在带 provider 前缀，文件名也自然带上 `<provider>__<model>` 段，不会跨 provider 撞名。

### 7. 不引入并发

第一版顺序执行所有 (provider, model)。pytest-xdist 之类的并发交给后续 PR——本 PR 的边界是"正确性 + 报告隔离"，不是"性能"。

## 影响范围

| 文件 | 改动性质 |
|------|----------|
| [conftest.py](/conftest.py) | 核心改造：global state、fixtures、参数化、report 钩子 |
| [src/report.py](/src/report.py) | `ReportCollector` 用法改为"每 provider 一个实例"；`generate_summary` 接口基本不动 |
| [src/config.py](/src/config.py) | 加 provider 名唯一性校验（`SuiteConfig` 上 model_validator） |
| [config.example.yaml](/config.example.yaml) | 加多 provider 示例（如果存在；否则在 [config.yaml](/config.yaml) 注释里说明） |
| [unit_tests/test_cli_config.py](/unit_tests/test_cli_config.py) | 加多 provider 配置的集成测试 |
| [unit_tests/test_report.py](/unit_tests/test_report.py) | 已有断言基于单 provider；多 provider 报告路径独立，原断言对单 provider 用例继续生效，**应无需改动**——确认即可 |
| [CLAUDE.md](/CLAUDE.md) | "Data flow" 段落更新；新增 "Multi-provider" 小节 |

## 实施步骤（TODO + Checkpoint）

每步完成后停下来 review。

### TODO 1：Config 层加唯一性校验

**产出：**
- `SuiteConfig` 加 `model_validator(mode="after")`：检查 `providers` 里 `name` 唯一且 slug 合法（`^[a-zA-Z0-9_-]+$`）
- `unit_tests/test_cli_config.py` 加一个 duplicate-name 报错的测试

**✋ Checkpoint 1**：跑 `uv run pytest unit_tests/ -v`，确认新校验生效；原有测试不受影响。

---

### TODO 2：conftest global state 改造

**产出：**
- 删除 `_active_provider`、`_active_models`、`_resolved_models` 这三个单值 global
- 引入：
  - `_resolved_run: list[tuple[ProviderConfig, list[ResolvedModel]]]`——一次 run 涉及的所有 provider 及其解析后的 model
  - `_collectors: dict[str, ReportCollector]`——按 provider name 索引
- `_report_dir` 仍是 `reports/{ts}/`；各 provider 子目录在创建 collector 时按需 `mkdir`

**关键点：**
- profile 解析（`resolve_models`）对每个 provider 各自调一次；任一 provider 解析失败 → 立即 `pytest.UsageError`（不让"一个错配的 provider 静默拖累整轮 run"）
- `pytest_configure` 末尾遍历 provider 创建子目录和 collector

**✋ Checkpoint 2**：单 provider config 在这一步应当仍能跑通——本 TODO 只是把"单 provider"变成"长度为 1 的列表"。

---

### TODO 3：Fixtures 改造（`provider_model` / `provider_config` / `resolved_model` / `client`）

**产出：**
- 新增 `provider_model` 间接 fixture，参数是 `(ProviderConfig, ResolvedModel)`
- `provider_config`、`resolved_model` 改为派生：
  ```python
  @pytest.fixture
  def provider_config(provider_model): return provider_model[0]
  @pytest.fixture
  def resolved_model(provider_model): return provider_model[1]
  ```
- `client` fixture 的 base_url / auth / verify_ssl 全部从 `provider_config` 来——逻辑不动，只是 provider 现在是参数化得到的
- log 路径改为 `_report_dir / provider_config.name / "logs" / safe_name + ".log"`
- log header 里 `Provider:` 行直接来自参数化的 provider，不再依赖全局

---

### TODO 4：参数化 + api_format 过滤

**产出：**
- `pytest_generate_tests`：
  - 若 fixture 名包含 `provider_model`：拼出所有 `(provider, model)` 对，按 test 所在路径过滤 api_format，给出 ids 形如 `provider::model`
  - 空 params 时让 pytest 自然 deselect（注意：`metafunc.parametrize` 不接受空列表，这种情况要先标 `pytest.mark.skip` 或在 `pytest_collection_modifyitems` 里 deselect）
- 简化 `pytest_collection_modifyitems`：只保留"没有 provider 配置时全 skip"分支，删 api_format 字符串匹配段

**✋ Checkpoint 4**：
- 单 provider config 行为不变（test id 多了 provider 前缀，否则等价）
- 双 provider（一个 openai、一个 anthropic）配置：`openai_compat/` test 只在 openai provider 上跑，`anthropic_compat/` 只在 anthropic 上跑
- `--collect-only` 输出能清楚反映这一点

---

### TODO 5：每 provider 一份 report

**产出：**
- `pytest_runtest_makereport` 改为：根据 item 的参数化值找出当前 provider，把 `TestResult` 加到对应 collector
  - 实现思路：从 `item.callspec.params["provider_model"]` 拿到 `(provider, resolved_model)`；callspec 在参数化 item 上存在
  - log_path 也要用 provider 子目录
- `pytest_sessionfinish` 遍历 `_collectors`，每个调 `generate_summary(provider, resolved_models)`，输出到 `reports/{ts}/{provider}/summary.md`
- 末尾顺手写一份 `reports/{ts}/index.md`：

  ```markdown
  # Run {timestamp}

  | Provider | Summary |
  |----------|---------|
  | mgtv | [summary.md](mgtv/summary.md) |
  | deepseek | [summary.md](deepseek/summary.md) |
  ```

- 终端打印调整为多行（每 provider 一行 summary path）

**✋ Checkpoint 5**：双 provider 跑一遍，确认两份独立 summary、index.md 链接正确、log 文件不串号。

---

### TODO 6：单元测试与示例配置

**产出：**
- `unit_tests/test_cli_config.py` 加：
  - 多 provider 配置加载与参数化验证（用 pytester 起子进程跑）
  - 两个 provider 的 api_format 不同 → 各自只跑匹配的 test
- 更新 [config.example.yaml](/config.example.yaml)（若文件不存在则在 [config.yaml](/config.yaml) 注释里加多 provider 示例段）
- `unit_tests/test_report.py` 跑一遍确认无回归

**✋ Checkpoint 6**：`uv run pytest unit_tests/ -v` 全绿；`uv run pyright`、`uv run ruff check .` 无新增告警。

---

### TODO 7：文档更新

**产出：**
- [CLAUDE.md](/CLAUDE.md) 的 "Data flow" 与 "Adding a new test" 段落更新：
  - test 参数化维度现在是 `(provider, model)`
  - log/report 路径改为 `reports/{ts}/{provider}/...`
- 简短追加一段"Multi-provider"说明，链接到本文件

**✋ Checkpoint 7**：让一个不熟悉的人读 CLAUDE.md 后能理解为什么 test id 有 `provider::model` 前缀。

## 非目标（明确不做）

1. **不做并发执行**——pytest-xdist 与 provider 分组协作是单独的话题，留给后续 PR
2. **不做跨 provider 的对比报告**（"provider A vs provider B 谁更兼容"）——每 provider 独立 summary 已经能让人手工对比，自动化对比会引入"哪个 model 与哪个 model 配对"这类设计问题，超出本 PR 范围
3. **不做 provider 级 setup/teardown 钩子**——目前每个 test 独立握 client，provider 共享状态需求暂未出现
4. **不引入 `--provider` CLI 过滤**——多 provider 的全跑就是默认行为，如果以后要选子集再加；本 PR 不预先设计 CLI 形态

## 设计权衡记录

- **为什么用 `provider::model` 而不是 `provider/model` 作 test id**：`/` 在 pytest nodeid 里有特殊含义（路径分隔），混进 param 段会让 `safe_name = nodeid.replace("/", "__")` 把它误转。`::` 在 nodeid 里也是分隔符，但出现在 `[...]` 内时 pytest 不解析——已经验证过这是相对安全的选择。

- **为什么不让每个 provider 一个独立 `_run_timestamp`**：用户明确要求"一次 run 一个 timestamp"。多个 timestamp 会让"这一次 run 的全貌"难以聚合，归档脚本（如 `tar reports/20260523_142817`）也会失效。

- **为什么 `--collect-only` 输出会变长**：参数化维度加了一个，是设计的必然结果。如果 collect 输出过长成为问题，可以在 `pytest_collection_modifyitems` 后做一个 provider-grouped 打印，但这是 nice-to-have。
