# AI Scheduler

AI Scheduler 是一套针对多任务神经网络场景的调度与优化引擎，能够在 NPU、DSP 等异构资源之间协调任务执行，并输出可视化的时间线与统计数据。

## ✨ 核心能力

- 🚀 **多资源调度**：支持 NPU / DSP 协同执行与资源负载均衡。
- 🎯 **优先级优化**：提供基于满足率的优先级搜索与评估。
- 📊 **可视化输出**：生成 Chrome Tracing JSON 与时间线 PNG（需本地安装 `matplotlib`）。
- 🧪 **场景复现**：内置多份 JSON 配置用于快速回放及回归测试。

## 📦 安装与构建

### 1. 开发环境安装（推荐）
```bash
pip install -e .
```

> 安装完成后，包内暴露的是 `NNScheduler` 模块及相关接口。当前工程未随 wheel 一同发布完善的 `ai_scheduler` 包装层，命令行脚本 `ai-scheduler` 在现有代码中不可用。

### 2. 构建 wheel 包
```bash
# 可选：清理旧产物
rm -rf build dist *.egg-info

# 方式一：使用 build 模块
python -m build --wheel

# 方式二：沿用 setup.py
python setup.py bdist_wheel
```

生成的文件位于 `dist/ai_scheduler-<版本>-py3-none-any.whl`。

### 3. 安装 wheel 包
```bash
pip install dist/ai_scheduler-<版本>-py3-none-any.whl
```

> 当前 wheel 仅包含 `NNScheduler` 命名空间；若需命令行入口，请继续使用仓库根目录的 `main_api.py`。

## 🚀 使用指南

### 直接运行调度（推荐）
```bash
# 运行预置场景
python main_api.py test/sample_config/config_1npu_1dsp.json --output ./artifacts_sim

# 显示更多细节
python main_api.py test/sample_config/dnr_4k30_tk_eager.json --verbose --output ./artifacts_debug
```

### Python 中调用核心接口
```python
from NNScheduler.interface.optimization_interface import OptimizationInterface

api = OptimizationInterface()
result = api.optimize_from_json("test/sample_config/config_1npu_1dsp.json")
print(result["best_configuration"]["satisfaction_rate"])
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

- 现有 wheel 入口文件仍指向未实现的 `ai_scheduler.cli:main`，安装后请直接使用仓库内的 `main_api.py` 或导入 `NNScheduler` 接口。
- 生成 PNG 时间线依赖 `matplotlib`，默认不随仓库安装，必要时需自行 `pip install matplotlib`。
- 可视化功能可通过环境变量 `AI_SCHEDULER_DISABLE_VISUALS=1` 关闭，以便在无绘图库环境运行。

欢迎在 `test/sample_config` 基础上扩展场景，也可使用 `artifacts_sim` 目录下的产物做复现与排错。
from ai_scheduler import OptimizationAPI

api = OptimizationAPI()

# 从JSON文件优化
result = api.optimize_from_json(config_file, output_dir)

# 从配置字典优化
result = api.optimize_from_dict(config_dict, output_dir)

# 验证配置文件
validation = api.validate_config(config_file)

# 列出样本配置
configs = api.list_sample_configs()
```

## 📊 输出文件

优化完成后，会在指定的输出目录生成以下文件：

- **甘特图** (`optimized_schedule_timeline_*.png`): 任务执行时间线可视化
- **Chrome Trace** (`optimized_schedule_chrome_trace_*.json`): Chrome浏览器可加载的跟踪文件
- **优化结果** (`optimization_result_*.json`): 详细的优化结果和统计信息
- **最优配置** (`optimized_priority_config_*.json`): 找到的最优优先级配置

## 🔍 样本配置

包内提供了多份样本配置文件，涵盖不同的硬件配置、调度策略与 launch profile 演示：

1. **config_1npu_1dsp.json**: 1个NPU + 1个DSP配置
2. **config_1npu_1dsp_alt.json**: 1个NPU + 1个DSP替代配置
3. **config_2npu_1dsp.json**: 2个NPU + 1个DSP配置
4. **config_2npu_1dsp_alt.json**: 2个NPU + 1个DSP替代配置
5. **config_2npu_2dsp.json**: 2个NPU + 2个DSP配置
6. **config_fixed_launch_example.json**: `fixed` 策略 + 依赖感知偏移示例
7. **config_eager_launch_profile.json**: `eager` 策略 + 多任务 launch_profile 示例

## 💡 高级用法

### 批量处理
```python
import ai_scheduler
import glob

def batch_optimize(config_pattern, output_base):
    config_files = glob.glob(config_pattern)
    results = []

    for config_file in config_files:
        output_dir = f"{output_base}/{Path(config_file).stem}"
        try:
            result = ai_scheduler.optimize_from_json(config_file, output_dir)
            results.append({
                'config': config_file,
                'satisfaction_rate': result['best_configuration']['satisfaction_rate']
            })
        except Exception as e:
            print(f"处理 {config_file} 时出错: {e}")

    return results

# 批量处理所有配置文件
results = batch_optimize("configs/*.json", "batch_results")
```

### 参数扫描
```python
import ai_scheduler
import itertools

# 定义参数范围
max_iterations_values = [10, 25, 50]
target_satisfaction_values = [0.8, 0.9, 0.95]

api = ai_scheduler.create_optimizer()
base_config_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')

for max_iter, target_sat in itertools.product(max_iterations_values, target_satisfaction_values):
    validation = api.validate_config(base_config_path)
    config = validation['config']

    # 修改参数
    config['optimization']['max_iterations'] = max_iter
    config['optimization']['target_satisfaction'] = target_sat

    # 运行优化
    result = api.optimize_from_dict(config, f"sweep_results/{max_iter}_{target_sat}")
    print(f"参数 ({max_iter}, {target_sat}): 满足率 {result['best_configuration']['satisfaction_rate']:.1%}")
```

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
- **版本**: 1.0.0

---

**AI Scheduler** - 让神经网络任务调度变得简单高效！ 🚀
