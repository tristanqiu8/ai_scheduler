# AI Scheduler

AI Scheduler 是一套针对多任务神经网络场景的调度与优化引擎，能够在 NPU、DSP、ISP、IP等异构资源之间协调任务执行，并输出可视化的时间线与统计数据。

## ✨ 核心能力

- 🚀 **多资源调度**：支持 NPU / DSP / ISP / CPU / IP等协同执行与资源负载均衡。
- 🎯 **优先级优化**：提供基于满足率的优先级搜索与评估（目前仅DSP和NPU支持）。
- 📊 **可视化输出**：生成 Chrome Tracing JSON 与时间线 PNG（需本地安装 `matplotlib`）。
- 🧪 **场景复现**：内置多份 JSON 配置用于快速回放及回归测试。

👉 面向最终用户的完整使用说明，请参阅最新版《[User_Guide.md](User_Guide.md)》。

## 📦 安装与构建

### 1. 开发环境安装（推荐）
```bash
pip install -e .
```

> 安装完成后，可在虚拟环境中直接导入 `NNScheduler` 与 `ai_scheduler` 模块；面向最终用户的详细操作说明请参阅《User_Guide.md》。

### 2. 面向用户的 wheel 安装
```bash
pip install xxxxx
ai-scheduler --version
ai-scheduler list-samples
```

常用样例 JSON 已随 wheel 一并打包，可通过 `ai-scheduler copy-sample <name> --dest ./configs/` 导出。

### 3. 构建 wheel 包
```bash
# 可选：清理旧产物
rm -rf build *.egg-info

# 方式一：使用 build 模块
python -m build --wheel

# 方式二：沿用 setup.py
python setup.py bdist_wheel
```

生成的文件位于 `dist/ai_scheduler-<版本>-py3-none-any.whl`。

### 4. 安装本地 wheel 包
```bash
pip install dist/ai_scheduler-<版本>-py3-none-any.whl
```

## 🚀 使用指南

### 通过 CLI 运行（发行包）
```bash
# 浏览样例
ai-scheduler list-samples

# 运行打包样例
ai-scheduler run sample:config_1npu_1dsp.json --output ./artifacts_sim --verbose

# 运行自定义配置
ai-scheduler run ./configs/custom_scenario.json --output ./artifacts_custom
```

### 通过源码入口（开发模式）
```bash
# 运行预置场景
python main_api.py test/sample_config/config_1npu_1dsp.json --output ./artifacts_sim

# 显示更多细节
python main_api.py test/sample_config/dnr_4k30_tk_eager.json --verbose --output ./artifacts_debug
```

### Python 中调用核心接口
```python
from ai_scheduler import OptimizationAPI, load_sample_config

api = OptimizationAPI(artifacts_dir="artifacts_sim/python")

# 使用内置样例
config = load_sample_config("config_1npu_1dsp.json")
result = api.optimize_from_config(config, verbose=True)
print(result["best_configuration"]["satisfaction_rate"])

# 直接读取文件
result = api.optimize_from_json("test/sample_config/dnr_4k30_tk_balance.json")
print(result["output_file"])
```

### 测试 & 验证
```bash
pytest                    # 全量回归
pytest test/NNScheduler/test_simple_optimization.py -k priority  # 定点用例
```

## 📁 仓库结构

```
├── NNScheduler/                 # 核心调度引擎
│   ├── core/                    # 调度执行、资源队列、评估等
│   ├── interface/               # JSON 接口、可视化、Web API
│   └── viz/                     # 时序可视化实现
├── artifacts_sim/               # 运行产物示例（Chrome Trace / PNG / JSON）
├── dist/                        # 已构建的 wheel 包
├── docs/                        # 额外文档
├── main_api.py                  # 推荐的命令行入口
├── setup.py                     # 打包脚本
└── test/                        # Pytest 套件与样例配置
```

## ⚙️ JSON 配置概览

```json
{
  "optimization": {
    "max_iterations": 30,
    "max_time_seconds": 120,
    "time_window": 200.0,
    "segment_mode": true,
    "launch_strategy": "balanced"
  },
  "resources": {
    "resources": [
      {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 80.0},
      {"resource_id": "DSP_0", "resource_type": "DSP", "bandwidth": 80.0}
    ]
  },
  "scenario": {
    "scenario_name": "示例场景",
    "tasks": [
      {
        "task_id": "T1",
        "priority": "HIGH",
        "fps": 30.0,
        "latency": 33.3,
        "model": {
          "segments": [
            {"resource_type": "NPU", "duration_table": {"80": 2.1}, "segment_id": "npu_s0"}
          ]
        }
      }
    ]
  }
}
```

关键字段说明：

- `optimization.launch_strategy`：`eager` / `lazy` / `balanced` / `sync` / `fixed`，同时写入生成文件名。
- `optimization.enable_random_slack`：首段高斯扰动开关，默认开启；`fixed` 模式同样支持扰动，`sync` 始终关闭。
- `optimization.slack`：首段扰动的标准差，单位毫秒，默认值 `0.2`。
- `optimization.random_slack_seed`：可选整数种子，设置后可复现扰动序列。
- `scenario.tasks[*].launch_profile`：可选自定义发射相位，支持 `offset_ms` 与 `respect_dependencies`（详见下节）。
- `scenario.tasks[*].model.segments`：描述任务在各资源上的序列执行片段，可配合 `cut_points` 进行细粒度分段。
- `dependencies`：声明任务间的执行依赖，执行器会在依赖完成后立即入队下一段。

### 发射策略与 launch_profile

- **自定义偏移**：`launch_profile.offset_ms` 允许任务在 `eager` / `balanced` / `fixed` 模式下按照固定相位周期性发射；未配置的任务仍由调度器自动推导发射时刻。
- **依赖感知**：当 `launch_profile.respect_dependencies` 为 `true` 时，调度器会在保证偏移的同时推迟到依赖任务完成；默认保持严格固定相位。
- **Sync vs Fixed**：`sync` 策略仍根据 ISP 时长自动推导偏移且禁用扰动，适用于自适应流水线；`fixed` 策略通过 `launch_profile` 显式传参锁定相位，并可叠加随机 slack。
- **示例配置**：
  - `test/sample_config/config_fixed_launch_example.json`：展示 `fixed` 策略与依赖对齐。
  - `test/sample_config/config_eager_launch_profile.json`：展示 `eager` 策略在多个任务间应用自定义偏移。

## ❗ 已知限制

- 生成 PNG 时间线依赖 `matplotlib`，默认不随仓库安装，必要时需自行 `pip install matplotlib`。
- 可视化功能可通过环境变量 `AI_SCHEDULER_DISABLE_VISUALS=1` 关闭，以便在无绘图库环境运行。

欢迎在 `test/sample_config` 基础上扩展场景，也可使用 `artifacts_sim` 目录下的产物做复现与排错。
## 🔧 开发相关

### 项目依赖
- Python >= 3.7
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- plotly >= 4.14.0
- python-dateutil >= 2.8.0

### 构建包
```bash
# 安装构建工具
pip install build twine

# 构建包
python -m build

# 本地安装测试
pip install dist/ai_scheduler-*.whl
```

## 👥 维护团队

- **维护者**: Tristan.Qiu
- **团队**: AIC (AI Computing)
- **版本**: 1.3

---

**AI Scheduler** - 让神经网络任务调度变得简单高效！ 🚀
