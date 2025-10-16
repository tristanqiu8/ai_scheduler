# JSON接口使用指南

## 概述

根据最近的提交（commit 2efc21f），Interface模块已经完成重构，实现了完整的JSON接口层。本文档总结了JSON接口的调用方式和使用方法。

## 接口架构

```
用户配置（JSON格式）
    ↓
JsonInterface（解析器）
    ↓
NNTask对象（任务实例）
    ↓
调度器核心
```

## 核心接口方法

JsonInterface类提供三个主要的解析方法：

### 1. parse_model_config()
解析模型配置，返回ResourceSegment列表和可选的CutPoint字典

### 2. parse_task_config()  
解析单个任务配置，返回NNTask实例

### 3. parse_scenario_config()
解析完整场景配置（包含多个任务），返回NNTask列表

## 使用方式

### 方式一：直接Python调用

```python
from interface.json_interface import JsonInterface

# 创建场景配置
config = {
    "scenario_name": "Camera Scenario via JSON",
    "description": "使用JSON接口创建的相机场景",
    "resources": [
        {
            "resource_id": "NPU_0",
            "resource_type": "NPU",
            "bandwidth": 40.0
        },
        {
            "resource_id": "DSP_0", 
            "resource_type": "DSP",
            "bandwidth": 40.0
        }
    ],
    "tasks": [
        # 任务配置列表（见下文）
    ]
}

# 解析并创建任务
interface = JsonInterface()
tasks = interface.parse_scenario_config(config)
```

### 方式二：任务配置格式

每个任务配置包含以下字段：

```json
{
    "task_id": "T1",                              // 任务唯一标识
    "name": "AimetlitePlus",                      // 任务名称
    "priority": "HIGH",                           // 优先级：HIGH/NORMAL/LOW
    "runtime_type": "ACPU_RUNTIME",               // 运行时类型
    "segmentation_strategy": "FORCED_SEGMENTATION", // 分段策略
    "fps": 30.0,                                  // 帧率要求
    "latency": 33.3,                              // 延迟要求（毫秒）
    "dependencies": ["T0"],                       // 依赖任务ID列表（可选）
    "model": {                                    // 模型配置
        // 见下文模型配置部分
    }
}
```

### 优化配置字段

在同一份 JSON 中，还可以通过 `optimization` 小节控制求解行为：

- `max_iterations` / `max_time_seconds` / `target_satisfaction`：迭代次数、耗时与目标满足率门限。
- `launch_strategy`：`eager` / `balanced` / `lazy` / `sync`，决定任务实例的发射模式。
- `segment_mode`：布尔值，设为 `true` 时在执行器中按子段调度。
- `slack`：首段扰动的方差（毫秒），默认 `0.2`，设为 `0` 表示无幅度。
- `enable_random_slack`：是否启用首段随机扰动，默认 `true`，可显式关闭。
- `random_slack_seed`：可选整数种子，设置后可复现同一组扰动样本。

## 模型配置

### 使用预定义模型

最简单的方式是使用模型库中的预定义模型：

```json
"model": {
    "model_name": "AimetlitePlus"  // 直接引用预定义模型名称
}
```

可用的预定义模型包括：
- AimetlitePlus
- FaceEhnsLite
- 其他（通过list_models()查看）

### 自定义模型

当需要自定义模型时，可以详细定义segments和cut_points：

```json
"model": {
    "segments": [
        {
            "resource_type": "NPU",                     // 资源类型：NPU/DSP
            "duration_table": {40: 3.0, 80: 2.0, 120: 1.5}, // 带宽-时长映射表
            "segment_id": "npu_main",                   // 段标识
            "power": 150.0,                             // 功耗（mW）
            "ddr": 8.0                                  // DDR带宽（GB/s）
        },
        {
            "resource_type": "DSP",
            "duration_table": {40: 1.5, 80: 1.0, 120: 0.8},
            "segment_id": "dsp_post",
            "power": 0.0,
            "ddr": 0.0
        }
    ],
    "cut_points": {                                     // 可选：切分点配置
        "npu_main": [
            {
                "op_id": "op1",
                "perf_lut": {40: 1.5, 80: 1.0},
                "overhead_ms": 0.0
            }
        ]
    }
}
```

## 完整示例

### 示例1：简单场景

```python
from interface.json_interface import JsonInterface

# 使用预定义模型的简单配置
simple_config = {
    "scenario_name": "Simple Camera",
    "tasks": [
        {
            "task_id": "T1",
            "name": "AimetlitePlus",
            "priority": "HIGH",
            "fps": 30.0,
            "model": {
                "model_name": "AimetlitePlus"
            }
        }
    ]
}

interface = JsonInterface()
tasks = interface.parse_scenario_config(simple_config)
```

### 示例2：复杂场景（带依赖和自定义模型）

```python
complex_config = {
    "scenario_name": "Complex Pipeline",
    "tasks": [
        {
            "task_id": "T1",
            "name": "Detection",
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
            "name": "PostProcess",
            "priority": "NORMAL",
            "dependencies": ["T1"],  // 依赖T1任务
            "fps": 30.0,
            "model": {
                "segments": [
                    {
                        "resource_type": "DSP",
                        "duration_table": {40: 2.0, 80: 1.0},
                        "segment_id": "dsp_process",
                        "power": 100.0,
                        "ddr": 4.0
                    }
                ]
            }
        }
    ]
}

tasks = interface.parse_scenario_config(complex_config)
```

## 运行演示

执行以下命令查看完整的使用示例：

```bash
python demo/demo_json_interface.py
```

该演示文件展示了：
1. 创建资源和任务配置
2. 使用预定义模型
3. 创建自定义模型
4. 设置任务依赖关系
5. 运行调度并生成可视化结果

## 支持的枚举值

### Priority（优先级）
- HIGH
- NORMAL  
- LOW

### RuntimeType（运行时类型）
- ACPU_RUNTIME
- KERNEL_RUNTIME
- DSP_RUNTIME

### SegmentationStrategy（分段策略）
- NO_SEGMENTATION
- ADAPTIVE_SEGMENTATION
- FORCED_SEGMENTATION

### ResourceType（资源类型）
- NPU
- DSP

## 注意事项

1. **带宽键值类型**：在JSON中，duration_table和perf_lut的键会自动转换为float类型
2. **依赖关系**：dependencies字段是可选的，用于定义任务间的依赖
3. **默认值**：
   - priority默认为"NORMAL"
   - runtime_type默认为"ACPU_RUNTIME"  
   - segmentation_strategy默认为"NO_SEGMENTATION"
   - fps默认为30.0
   - latency默认为33.3ms（1000.0/30.0）

## 相关文件

- `interface/json_interface.py` - JSON接口实现
- `demo/demo_json_interface.py` - 完整使用示例
- `interface/README.md` - 接口模块总体说明
- `interface/web_api.py` - Web API实现（基于JSON接口）
- `interface/api_client_example.py` - API客户端示例
