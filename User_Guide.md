# AI Scheduler 用户指南

AI Scheduler 是一套针对多任务神经网络部署的场景级调度，仿真与优化工具，它以NPU为调度资源核心，能够在NPU、DSP、ISP(开发中)、IP(如VCR，开发中)等异构资源之间协调任务执行，并输出可视化的时间线与统计数据。它非常轻便，简洁，快速，可帮助你分析当前NN场景流水仿真的一些潜在痛点，可在规划阶段帮助排查极限场景的NN风险以及在交付阶段与实际上板抓的流水图相互配合——协助挖掘NN场景的优化方向，痛点定位，最终提高整体任务完成质量和NPU吞吐量。

我们的愿景是每个人都可以看到自己的NN场景级调度，让场景级调度仿真变得轻而易举！

欢迎来到 AI Scheduler 用户指南！本文面向首次接触该工具的用户，帮助你从 wheel 包快速安装、运行预置 JSON 配置、理解输入格式，并正确选择各类发射（launch）模式。

## 1. 环境与安装

### 1.1 环境要求

- Python 3.8 及以上版本，Ubuntu OS。
- 推荐在虚拟环境（`venv`、`conda`）中安装。

### 1.2 安装 wheel 并验证

```bash
pip install ai_scheduler-<version number>-py3-none-any.whl
```

安装完成后，`ai-scheduler` 命令行工具与 `ai_scheduler` Python 包会同时就绪。可通过以下命令快速验证：

```bash
ai-scheduler --version
```

### 2.1 浏览与导出样例

wheel 已内置常用 JSON 场景，可通过 CLI 直接查看或导出：

```bash
ai-scheduler list-samples
ai-scheduler copy-sample config_2npu_2dsp.json --dest ./configs/
```

命令支持省略扩展名（例如 `copy-sample config_2npu_2dsp --dest ...`），`--dest` 既可指定目录，也可指定目标文件名。

### 2.2 使用 CLI 运行样例

```bash
ai-scheduler run sample:config_1npu_1dsp.json \
  --output artifacts_demo/config_1npu_1dsp \
  --verbose
```

命令会将结果写入 `--output` 指定目录（未填写时默认 `./artifacts_sim`），并在控制台输出最佳满意率及产物路径。

### 2.3 运行自定义 JSON

```bash
ai-scheduler run ./configs/my_scenario.json --output artifacts_demo/my_scenario
```

如需关闭 Banner，可增加 `--no-banner`；加上 `--verbose` 能看到更详细的执行信息。

### 2.4 样例配置速查

| 文件名 | 场景描述 | 推荐用途 |
| --- | --- | --- |
| `config_1npu_1dsp.json` | 1×NPU + 1×DSP，基础任务组合 | 新手入门、快速验证安装 |
| `config_2npu_1dsp.json` | 双 NPU 场景，带 DSP 负载均衡 | 观察不同发射策略的资源占用 |
| `dnr_4k30_tk_eager.json` | 4K/30fps 视频降噪，`eager` 策略 | 高吞吐场景的端到端延迟评估 |
| `dnr_4k30_tk_balance.json` | 与上例相同场景，`balanced` 策略 | 对比不同发射策略满意率 |
| `config_fixed_launch_example.json` | 自定义发射相位 + 依赖声明 | 研究 `fixed` / `sync` 行为 |
| `dnr_4k30_tk_eager_launch_profile.json` | 在 4K30 DNR 场景中演示 `launch_profile` 延迟发射 | 观察 `eager` 策略下的相位调节 |

> 小贴士：直接使用 `sample:<name>.json` 前缀即可运行内置样例，无需手动复制。

## 3. JSON 输入格式详解

所有配置文件均为 UTF-8 编码的 JSON，顶层包含三个核心部分：

```json
{
  "optimization": { ... },
  "resources": { ... },
  "scenario": { ... }
}
```

### 3.1 `optimization`

- `max_iterations` (int)：最大迭代次数，决定搜索深度。
- `max_time_seconds` (float)：总运行时限，单位秒。
- `time_window` (float)：时间线窗口长度，单位毫秒，影响可视化范围。
- `launch_strategy` (string)：任务发射模式，详见第 4 章。
- `search_priority` (bool)：是否执行优先级搜索；设为 `false` 时直接使用 `scenario.tasks[*].priority`。
- `enable_random_slack` (bool)：首段随机扰动开关，默认为 true。
- `slack` (float)：扰动标准差（毫秒），默认 0.2。
- `random_slack_seed` (int)：可选随机种子，保证结果可复现。
- 可选目标项：`target_satisfaction`、`acceptance_threshold` 等，用于提前收敛。

### 3.2 `resources`

```json
"resources": {
  "resources": [
    {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 80.0},
    {"resource_id": "DSP_0", "resource_type": "DSP", "bandwidth": 80.0}
  ]
}
```

- 必须列出所有可调度资源；`resource_id` 在同一配置中唯一。
- `bandwidth` 单位可自定义（通常是 TOPS 或归一化吞吐量）。
- 当资源存在多条调度队列，可扩展字段 `queue_count`、`latency_budget` 等自定义属性。

### 3.3 `scenario`

- `scenario_name`：便于日志与产物命名。
- `tasks`：任务数组，每个任务包含：
  - `task_id`：唯一任务标识。
  - `priority`：`HIGH` / `MEDIUM` / `LOW` 等等级，影响全局排序。
  - `fps` 与 `latency`：输入帧率与延迟预算。
- `launch_profile`：可选发射相位设置，支持：
  - `offset_ms`：首帧偏移（毫秒）。
  - `respect_dependencies`：是否等待依赖任务完成。
  - 示例：`dnr_4k30_tk_eager_launch_profile.json` 中的 `parsing` 任务将 `offset_ms` 设置为 12ms，以模拟后移发射。
  - `model.segments`：任务片段序列，按执行资源划分：
    ```json
    {
      "segment_id": "npu_s0",
      "resource_type": "NPU",
      "duration_table": {"80": 2.1}
    }
    ```
    - `duration_table` 使用带宽与推理耗时映射（毫秒）。
    - 可选 `cut_points` 字段可进一步拆分长片段。
- `dependencies`：声明跨任务依赖关系，例如：
  ```json
  "dependencies": [
    {"upstream_task": "T1", "downstream_task": "T2", "type": "finish_start"}
  ]
  ```

## 4. 发射模式（Launch Strategies）对比

| 模式 | 行为特性 | 适用场景 | 随机扰动支持 |
| --- | --- | --- | --- |
| `eager` | 新帧尽早入队，优先保证帧率 | 超高吞吐业务、低延迟优先 | 支持 (`enable_random_slack` 控制) |
| `lazy` | 拉长间隔，缓解抖动，偏静态节奏 | 带宽紧张、延迟预算宽松 | 支持 |
| `balanced` | 在 `eager` 与 `lazy` 间权衡 | 资源多样、需兼顾利用率与稳定性 | 支持 |
| `sync`* | 严格按照依赖同步发射 | 强依赖链路、需要确定性 | 不支持随机扰动 |
| `fixed` | 完全遵守 `launch_profile` 自定义偏移 | 多子系统协同、需手工调相 | 支持 |

> 小贴士：
> - 仅修改 `launch_strategy` 即可快速对比不同策略。
> - 加上 `--verbose` 可在控制台查看发射细节。
> - sync仍在开发中

## 5. 输出产物与可视化

默认输出目录会生成以下文件：

- `optimization_result_*.json`：包含搜索过程统计与满意率。
- `optimized_priority_config_*.json`：记录最优优先级配置，可直接复用。
- `optimized_schedule_chrome_trace_*.json`：可导入 Chrome Tracing (`chrome://tracing`或者`https://ui.perfetto.dev/`)。
- `optimized_schedule_timeline_*.png`：时间线可视化，依赖 `matplotlib`。

如需自定义产物根目录，可在运行前设置环境变量 `AI_SCHEDULER_ARTIFACTS_DIR=<path>`。不需要图像输出时，可设置 `AI_SCHEDULER_DISABLE_VISUALS=1`。

## 6. Python API 调用示例

```python
from ai_scheduler import OptimizationAPI, load_sample_config

api = OptimizationAPI(artifacts_dir="artifacts_demo/programmatic")

# 1) 直接使用内置样例
config = load_sample_config("config_1npu_1dsp.json")
result = api.optimize_from_config(config, verbose=True)
print("最佳满意率:", result["best_configuration"]["satisfaction_rate"])

# 2) 从文件加载
result = api.optimize_from_json("./configs/my_scenario.json")
print("结果文件存放于:", result["output_file"])

# 3) 仅做配置体检
summary = api.validate_config(config)
print("任务数量:", summary["task_count"], "发射策略:", summary["launch_strategy"])
```

常用 API 速记：

- `OptimizationAPI.optimize_from_json(path, output_dir=None, verbose=False)`
- `OptimizationAPI.optimize_from_config(config, output_dir=None, verbose=False)`
- `OptimizationAPI.validate_config(config_or_path)`
- `list_sample_configs()` / `load_sample_config(name)` / `copy_sample_config(name, dest, overwrite=False)`

## 7. 常见问题排查

- **命令找不到 (`command not found`)**：确认虚拟环境已激活，或使用 `python -m ai_scheduler.cli` 运行。
- **输出目录未生成**：检查是否拥有写权限，或在命令中显式指定 `--output` / `AI_SCHEDULER_ARTIFACTS_DIR`。
- **缺少绘图库**：如需要输出 PNG，请先 `pip install matplotlib`。
- **配置校验失败**：使用 `OptimizationAPI.validate_config(...)` 或 `ai-scheduler run --verbose` 观察报错定位。
- **满意率过低**：尝试提高 `max_iterations`、调整 `launch_strategy`、增加资源带宽或放宽 `latency`。

## 8. 下一步行动

- 根据样例 JSON 调整参数，观察产物与满意率变化。
- 将自定义场景纳入自动化回归：`pytest test/NNScheduler/test_optimization_comparison.py -k <scenario>`。
- 发布前检查 `dist/` 内的 wheel 是否包含 `ai_scheduler`、`NNScheduler` 目录及 `sample_config` 数据。

如需更多背景与未来需求信息，请联系AIC开发者Tristan.Qiu, Xiong.Guo, Neal.Nie。
