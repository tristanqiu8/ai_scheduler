# 优化API使用指南

## 概述

优化API提供与`test_cam_auto_priority_optimization.py`相同的自动化优先级优化功能，支持JSON配置文件格式输入，实现一模一样的优化效果。

## 核心功能

- ✅ **自动任务特征分析** - 分析被依赖次数、FPS要求、延迟严格度等
- ✅ **智能优先级分配** - 基于任务特征自动生成初始优先级配置
- ✅ **迭代优化过程** - 自动调整优先级直到满足FPS和延迟要求
- ✅ **详细性能分析** - FPS分析、功耗分析、DDR分析、系统利用率等
- ✅ **可视化输出** - 自动生成Chrome Tracing和时间线图片
- ✅ **结果保存** - 保存最佳配置和优化历史到JSON文件

## 使用方式

### 方式一：直接Python调用

```python
from NNScheduler.interface.optimization_interface import OptimizationInterface

# 创建优化接口
optimizer = OptimizationInterface()

# 从JSON配置文件运行优化
result = optimizer.optimize_from_json("optimization_config.json")

print(f"优化完成！满足率: {result['best_configuration']['satisfaction_rate']:.1%}")
print(f"结果保存到: {result['output_file']}")
```

### 方式二：使用配置字典

```python
from NNScheduler.interface.optimization_interface import OptimizationInterface

# 创建配置
config = {
    "optimization": {
        "max_iterations": 50,
        "max_time_seconds": 300,
        "target_satisfaction": 0.95,
        "time_window": 1000.0,
        "segment_mode": True
    },
    "scenario": {
        "use_camera_tasks": True  # 使用预定义相机任务
    },
    "resources": {
        "resources": [
            {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 160.0},
            {"resource_id": "DSP_0", "resource_type": "DSP", "bandwidth": 160.0}
        ]
    }
}

# 运行优化
optimizer = OptimizationInterface()
result = optimizer.optimize_from_config(config)
```

## JSON配置格式规范

### 完整配置示例

```json
{
  "optimization": {
    "max_iterations": 50,
    "max_time_seconds": 300,
    "target_satisfaction": 0.95,
    "time_window": 1000.0,
    "segment_mode": true,
    "enable_detailed_analysis": true
  },
  "resources": {
    "resources": [
      {
        "resource_id": "NPU_0",
        "resource_type": "NPU",
        "bandwidth": 160.0
      },
      {
        "resource_id": "DSP_0",
        "resource_type": "DSP",
        "bandwidth": 160.0
      }
    ]
  },
  "scenario": {
    "use_camera_tasks": true
  }
}
```

### 自定义任务配置示例

```json
{
  "optimization": {
    "max_iterations": 30,
    "max_time_seconds": 180,
    "target_satisfaction": 0.90,
    "time_window": 1000.0,
    "segment_mode": true
  },
  "resources": {
    "resources": [
      {
        "resource_id": "NPU_0",
        "resource_type": "NPU",
        "bandwidth": 160.0
      },
      {
        "resource_id": "DSP_0",
        "resource_type": "DSP",
        "bandwidth": 160.0
      }
    ]
  },
  "scenario": {
    "use_camera_tasks": false,
    "scenario_name": "Custom Optimization Scenario",
    "description": "自定义优化场景",
    "tasks": [
      {
        "task_id": "T1",
        "name": "HighFPSTask",
        "priority": "HIGH",
        "runtime_type": "ACPU_RUNTIME",
        "segmentation_strategy": "FORCED_SEGMENTATION",
        "fps": 30.0,
        "latency": 33.3,
        "model": {
          "model_name": "AimetlitePlus"
        }
      },
      {
        "task_id": "T2",
        "name": "PostProcessTask",
        "priority": "NORMAL",
        "runtime_type": "ACPU_RUNTIME",
        "segmentation_strategy": "NO_SEGMENTATION",
        "fps": 15.0,
        "latency": 66.7,
        "dependencies": ["T1"],
        "model": {
          "segments": [
            {
              "resource_type": "DSP",
              "duration_table": {"40": 5.0, "80": 3.0, "160": 2.0},
              "segment_id": "dsp_post",
              "power": 100.0,
              "ddr": 4.0
            }
          ]
        }
      }
    ]
  }
}
```

## 配置字段说明

### optimization 字段

| 字段名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_iterations` | int | 50 | 最大迭代次数 |
| `max_time_seconds` | int | 300 | 最大运行时间（秒） |
| `target_satisfaction` | float | 0.95 | 目标满足率（0.0-1.0） |
| `time_window` | float | 1000.0 | 仿真时间窗口（毫秒） |
| `segment_mode` | bool | true | 是否启用段级调度模式 |
| `enable_detailed_analysis` | bool | true | 是否启用详细分析 |

### resources 字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `resource_id` | string | 资源ID |
| `resource_type` | string | 资源类型: "NPU", "DSP", "ISP", "CPU", "GPU", "VPU", "FPGA" |
| `bandwidth` | float | 资源带宽（GB/s） |

### scenario 字段

| 字段名 | 类型 | 说明 |
|--------|------|------|
| `use_camera_tasks` | bool | 是否使用预定义相机任务（推荐设为true） |
| `scenario_name` | string | 场景名称（自定义任务时使用） |
| `description` | string | 场景描述（自定义任务时使用） |
| `tasks` | array | 任务配置列表（自定义任务时使用） |

### task 字段（自定义任务时）

| 字段名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `task_id` | string | 必填 | 任务唯一标识 |
| `name` | string | 必填 | 任务名称 |
| `priority` | string | "NORMAL" | 优先级: "CRITICAL", "HIGH", "NORMAL", "LOW" |
| `runtime_type` | string | "ACPU_RUNTIME" | 运行时类型 |
| `segmentation_strategy` | string | "NO_SEGMENTATION" | 分段策略 |
| `fps` | float | 30.0 | FPS要求 |
| `latency` | float | 33.3 | 延迟要求（毫秒） |
| `dependencies` | array | [] | 依赖任务ID列表 |
| `model` | object | 必填 | 模型配置 |

## 输出结果格式

优化API返回包含以下信息的详细结果：

```json
{
  "timestamp": "2024-01-15 14:30:25",
  "optimization_config": { ... },
  "best_configuration": {
    "priority_config": {
      "T1": "HIGH",
      "T2": "NORMAL",
      "T3": "CRITICAL"
    },
    "satisfaction_rate": 0.95,
    "avg_latency": 32.5,
    "resource_utilization": {
      "NPU": 85.2,
      "DSP": 62.8
    },
    "fps_analysis": {
      "T1": 30.0,
      "T2": 15.0,
      "total_fps": 45.0,
      "total_segment_executions": 150
    },
    "power_analysis": {
      "T1": 450.0,
      "T2": 150.0,
      "total_power_mw": 600.0,
      "total_power_w": 0.6
    },
    "ddr_analysis": {
      "T1": 240.0,
      "T2": 60.0,
      "total_ddr_mb": 300.0,
      "total_ddr_gb": 0.293
    },
    "system_utilization": 92.1,
    "fps_satisfaction": {
      "T1": true,
      "T2": true
    },
    "latency_satisfaction": {
      "T1": true,
      "T2": true
    }
  },
  "optimization_history": [
    {
      "iteration": 0,
      "priority_config": { ... },
      "total_satisfaction_rate": 0.8,
      "avg_latency": 38.2,
      "resource_utilization": { ... },
      "fps_analysis": { ... },
      "power_analysis": { ... },
      "ddr_analysis": { ... },
      "system_utilization": 88.5
    }
  ],
  "task_features": {
    "T1": {
      "name": "TaskName",
      "fps_requirement": 30.0,
      "latency_requirement": 33.3,
      "dependency_count": 2,
      "has_dependencies": false,
      "num_segments": 3,
      "uses_npu": true,
      "uses_dsp": true,
      "latency_strictness": 0.75,
      "fps_strictness": 30.0
    }
  },
  "visualization_files": {
    "chrome_trace": "optimized_schedule_chrome_trace_20240115_143025.json",
    "timeline_png": "optimized_schedule_timeline_20240115_143025.png"
  },
  "output_file": "optimization_result_20240115_143025.json"
}
```

## 与test_cam_auto_priority_optimization.py的对比

| 功能 | 原始测试程序 | JSON API |
|------|-------------|----------|
| 任务特征分析 | ✅ | ✅ |
| 智能优先级分配 | ✅ | ✅ |
| 迭代优化过程 | ✅ | ✅ |
| FPS分析 | ✅ | ✅ |
| 功耗分析 | ✅ | ✅ |
| DDR分析 | ✅ | ✅ |
| 系统利用率分析 | ✅ | ✅ |
| Chrome Tracing生成 | ✅ | ✅ |
| 时间线图片生成 | ✅ | ✅ |
| 结果保存 | ✅ | ✅ |
| JSON配置输入 | ❌ | ✅ |
| 批量处理 | ❌ | ✅ |
| API接口 | ❌ | ✅ |

## 快速开始模板

### 生成配置模板

```python
from NNScheduler.interface.optimization_interface import OptimizationInterface

optimizer = OptimizationInterface()
template = optimizer.create_optimization_template()

# 保存模板
from NNScheduler.interface.json_interface import JsonInterface
JsonInterface.save_to_file(template, "optimization_template.json")
```

### 运行优化

```bash
# 1. 编辑配置文件 optimization_config.json
# 2. 运行优化
python -c "
from NNScheduler.interface.optimization_interface import OptimizationInterface
optimizer = OptimizationInterface()
result = optimizer.optimize_from_json('optimization_config.json')
print(f'优化完成！满足率: {result[\"best_configuration\"][\"satisfaction_rate\"]:.1%}')
"
```

## 高级用法

### 批量优化多个配置

```python
import os
from NNScheduler.interface.optimization_interface import OptimizationInterface

optimizer = OptimizationInterface()
config_dir = "optimization_configs"

for config_file in os.listdir(config_dir):
    if config_file.endswith(".json"):
        print(f"优化配置: {config_file}")
        result = optimizer.optimize_from_json(
            os.path.join(config_dir, config_file),
            f"result_{config_file}"
        )
        print(f"满足率: {result['best_configuration']['satisfaction_rate']:.1%}")
```

### 自定义优化参数

```python
from NNScheduler.interface.optimization_interface import OptimizationInterface

# 加载基础配置
config = JsonInterface.load_from_file("base_config.json")

# 自定义优化参数
config["optimization"].update({
    "max_iterations": 100,      # 增加迭代次数
    "target_satisfaction": 0.98, # 提高目标满足率
    "time_window": 2000.0       # 增加仿真时间
})

# 运行优化
optimizer = OptimizationInterface()
result = optimizer.optimize_from_config(config)
```

## 错误处理

API会返回详细的错误信息：

```python
try:
    optimizer = OptimizationInterface()
    result = optimizer.optimize_from_json("config.json")
except FileNotFoundError:
    print("配置文件不存在")
except KeyError as e:
    print(f"配置文件缺少必要字段: {e}")
except Exception as e:
    print(f"优化过程中出现错误: {e}")
```

## 性能建议

1. **使用预定义相机任务**：设置 `"use_camera_tasks": true` 可获得最佳优化效果
2. **合理设置迭代次数**：通常20-50次迭代足够找到较好的解决方案
3. **调整目标满足率**：对于复杂场景，可以将目标满足率设为0.85-0.95
4. **启用段级模式**：`"segment_mode": true` 通常能获得更好的调度效果
5. **适当的资源带宽**：NPU和DSP建议设置为160.0 GB/s以获得最佳性能

## 文件说明

- `optimization_interface.py` - 优化API实现
- `OPTIMIZATION_API_GUIDE.md` - 本文档
- `optimization_template.json` - 配置模板
- `optimization_result_*.json` - 优化结果文件
- `optimized_schedule_chrome_trace_*.json` - Chrome Tracing文件
- `optimized_schedule_timeline_*.png` - 时间线图片文件