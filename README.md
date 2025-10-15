# AI Scheduler

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/your-org/ai-scheduler)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

AI Scheduler是一个专业的神经网络任务调度器，具有优先级优化功能。它能够智能地在NPU（神经处理单元）和DSP（数字信号处理器）资源上调度和优化神经网络任务的执行。

## ✨ 主要特性

- 🚀 **智能任务调度**: 基于优先级和资源约束的自动任务调度
- ⚡ **多资源支持**: 支持NPU和DSP混合资源调度
- 🎯 **优化算法**: 内置优先级搜索和满足率优化算法
- 📊 **可视化输出**: 生成甘特图和Chrome Tracing文件
- 🛠️ **双接口支持**: 提供命令行和Python API两种使用方式
- 📦 **开箱即用**: 包含多种预配置的样本场景
- 🔧 **高度可配置**: 支持JSON配置文件和程序化配置

## 📦 安装

### 从PyPI安装（推荐）
```bash
pip install ai-scheduler
```

### 从源码安装
```bash
git clone <repository-url>
cd ai-scheduler
pip install -e .
```

## 🚀 快速开始

### 命令行使用

```bash
# 查看所有可用的样本配置
ai-scheduler --list-samples

# 使用内置样本配置运行优化
ai-scheduler sample:config_1npu_1dsp.json

# 使用自定义配置文件
ai-scheduler my_config.json --output ./results

# 验证配置文件格式
ai-scheduler --validate my_config.json

# 启用详细输出
ai-scheduler my_config.json --verbose
```

### Python API使用

```python
import ai_scheduler

# 最简单的使用方式
result = ai_scheduler.optimize_from_json('config.json')
print(f"满足率: {result['best_configuration']['satisfaction_rate']:.1%}")

# 使用内置样本配置
sample_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')
result = ai_scheduler.optimize_from_json(sample_path, output_dir='./output')

# 创建优化器实例进行高级操作
api = ai_scheduler.create_optimizer()
validation = api.validate_config('config.json')
if validation['valid']:
    result = api.optimize_from_json('config.json', 'output')
```

## 📁 项目结构

```
ai-scheduler/
├── ai_scheduler/              # 主包目录
│   ├── __init__.py           # 包初始化和便捷函数
│   ├── cli.py                # 命令行接口
│   ├── core/                 # 核心功能模块
│   │   ├── optimization_api.py  # 优化API
│   │   └── __init__.py
│   ├── NNScheduler/          # 底层调度器模块
│   │   ├── core/             # 核心调度逻辑
│   │   ├── interface/        # 接口模块
│   │   └── ...
│   └── sample_config/        # 样本配置文件
│       ├── config_1npu_1dsp.json
│       ├── config_2npu_1dsp.json
│       └── ...
├── example_test.py           # 使用示例
├── setup.py                  # 安装配置
├── requirements.txt          # 依赖文件
└── README.md                # 本文件
```

## 🎯 核心概念

### 任务类型
- **NPU任务**: 在神经处理单元上执行的推理任务
- **DSP任务**: 在数字信号处理器上执行的信号处理任务
- **混合任务**: 需要在多种资源上顺序执行的复杂任务

### 优先级系统
- **CRITICAL**: 最高优先级，优先调度
- **HIGH**: 高优先级
- **NORMAL**: 普通优先级
- **LOW**: 低优先级

### 调度策略
- **搜索优化模式** (`search_priority: true`): 系统自动搜索和调整任务优先级
- **固定优先级模式** (`search_priority: false`): 使用用户配置的固定优先级

## ⚙️ 配置文件格式

AI Scheduler使用JSON格式的配置文件，基本结构如下：

```json
{
  "optimization": {
    "max_iterations": 25,
    "target_satisfaction": 0.95,
    "search_priority": true,
    "log_level": "normal"
  },
  "resources": {
    "resources": [
      {
        "resource_id": "NPU_0",
        "resource_type": "NPU",
        "bandwidth": 160.0
      }
    ]
  },
  "scenario": {
    "scenario_name": "示例场景",
    "description": "场景描述",
    "tasks": [
      {
        "task_id": "TASK_1",
        "name": "Task1",
        "priority": "HIGH",
        "fps": 30.0,
        "latency": 20.0,
        "model": {
          "segments": [...]
        }
      }
    ]
  }
}
```

### 主要配置参数

#### optimization段
- `max_iterations`: 最大优化迭代次数
- `max_time_seconds`: 最大优化时间（秒）
- `target_satisfaction`: 目标满足率 (0.0-1.0)
- `search_priority`: 是否启用优先级搜索优化
- `log_level`: 日志级别 ("normal" 或 "detailed")

#### resources段
- `resource_id`: 资源唯一标识
- `resource_type`: 资源类型 ("NPU" 或 "DSP")
- `bandwidth`: 资源带宽

#### tasks段
- `task_id`: 任务唯一标识
- `priority`: 任务优先级
- `fps`: 期望帧率
- `latency`: 延迟要求（毫秒）
- `segmentation_strategy`: 分段策略
- `dependencies`: 任务依赖关系

## 🔧 API参考

### 便捷函数

```python
# 从JSON文件运行优化
ai_scheduler.optimize_from_json(config_file, output_dir="./artifacts")

# 创建优化器实例
ai_scheduler.create_optimizer(config_dict=None)

# 获取样本配置路径
ai_scheduler.get_sample_config_path(name)

# 列出所有样本配置
ai_scheduler.get_sample_configs()

# 获取版本信息
ai_scheduler.version_info()
```

### OptimizationAPI类

```python
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

包内提供了5个样本配置文件，涵盖不同的硬件配置和任务类型：

1. **config_1npu_1dsp.json**: 1个NPU + 1个DSP配置
2. **config_1npu_1dsp_alt.json**: 1个NPU + 1个DSP替代配置
3. **config_2npu_1dsp.json**: 2个NPU + 1个DSP配置
4. **config_2npu_1dsp_alt.json**: 2个NPU + 1个DSP替代配置
5. **config_2npu_2dsp.json**: 2个NPU + 2个DSP配置

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

## 📄 许可证

本项目采用MIT许可证。详细信息请参见 [LICENSE](LICENSE) 文件。

## 👥 维护团队

- **维护者**: Tristan.Qiu
- **团队**: AIC (AI Computing)
- **版本**: 1.0.0

## 🤝 贡献

欢迎贡献代码！请参考以下步骤：

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📞 支持

如果您遇到问题或有任何建议，请：

1. 查看 [示例文件](example_test.py)
2. 阅读文档
3. 提交Issue到GitHub仓库

---

**AI Scheduler** - 让神经网络任务调度变得简单高效！ 🚀