# AI Scheduler Interface Module

本模块提供了两层接口来操作AI调度器：
1. **JSON接口层** - 直接的Python接口，使用JSON格式配置
2. **Web API层** - 基于Flask的RESTful API，构建在JSON接口之上

## 架构

```
用户输入
    ↓
Web API (HTTP/REST)
    ↓
JSON Interface (Python)
    ↓
Task/Model Creation
    ↓
Scheduler Core
```

## 安装依赖

```bash
pip install -r interface/requirements.txt
```

## JSON接口使用

### 基本用法

```python
from interface.json_interface import JsonInterface

# 1. 创建任务配置
task_config = {
    "task_id": "T1",
    "name": "MyTask",
    "priority": "HIGH",
    "fps": 30.0,
    "latency": 33.3,
    "model": {
        "model_name": "AimetlitePlus"  # 使用预定义模型
    }
}

# 2. 解析任务
task = JsonInterface.parse_task_config(task_config)

# 3. 创建场景配置
scenario_config = {
    "scenario_name": "My Scenario",
    "tasks": [task_config]
}

# 4. 解析场景
tasks = JsonInterface.parse_scenario_config(scenario_config)
```

### 自定义模型

```python
custom_model = {
    "segments": [
        {
            "resource_type": "NPU",
            "duration_table": {40: 3.0, 80: 2.0},
            "segment_id": "main",
            "power": 200.0,
            "ddr": 10.0
        }
    ],
    "cut_points": {
        "main": [
            {
                "op_id": "op1",
                "perf_lut": {40: 1.5, 80: 1.0},
                "overhead_ms": 0.0
            }
        ]
    }
}
```

### 运行示例

```bash
python demo/demo_json_interface.py
```

## Web API使用

### 启动服务器

```bash
python interface/web_api.py
```

服务器将在 http://localhost:5000 启动

### API端点

| 方法 | 端点 | 描述 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| GET | `/api/models` | 获取可用模型列表 |
| GET | `/api/example` | 获取示例配置 |
| POST | `/api/tasks` | 创建任务 |
| GET | `/api/tasks` | 获取当前任务 |
| POST | `/api/resources` | 设置资源 |
| GET | `/api/resources` | 获取资源配置 |
| POST | `/api/schedule` | 运行调度 |
| GET | `/api/schedule/trace` | 获取调度跟踪 |
| GET | `/api/schedule/metrics` | 获取性能指标 |
| POST | `/api/schedule/visualization` | 生成可视化 |
| POST | `/api/config/load` | 加载配置 |
| POST | `/api/config/save` | 保存配置 |
| POST | `/api/reset` | 重置调度器 |

### Python客户端示例

```python
from interface.api_client_example import SchedulerAPIClient

client = SchedulerAPIClient("http://localhost:5000")

# 1. 健康检查
print(client.check_health())

# 2. 获取可用模型
models = client.get_models()

# 3. 设置资源
resources_config = {
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
    ]
}
client.set_resources(resources_config)

# 4. 创建任务
tasks_config = {
    "tasks": [
        {
            "task_id": "T1",
            "name": "Task1",
            "priority": "HIGH",
            "fps": 30.0,
            "latency": 33.3,
            "model": {"model_name": "AimetlitePlus"}
        }
    ]
}
client.create_tasks(tasks_config)

# 5. 运行调度
result = client.run_schedule(time_window=1000.0, segment_mode=True)

# 6. 获取性能指标
metrics = client.get_metrics()

# 7. 生成可视化
client.generate_visualization("schedule.png")
```

### cURL示例

```bash
# 健康检查
curl http://localhost:5000/api/health

# 获取可用模型
curl http://localhost:5000/api/models

# 创建任务
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "tasks": [{
      "task_id": "T1",
      "name": "MyTask",
      "priority": "HIGH",
      "fps": 30.0,
      "latency": 33.3,
      "model": {"model_name": "AimetlitePlus"}
    }]
  }'

# 运行调度
curl -X POST http://localhost:5000/api/schedule \
  -H "Content-Type: application/json" \
  -d '{
    "time_window": 1000.0,
    "segment_mode": true,
    "launcher": "enhanced"
  }'
```

### 运行完整示例

```bash
# 启动API服务器
python interface/web_api.py

# 在另一个终端运行客户端示例
python interface/api_client_example.py
```

## 配置文件格式

### 完整配置示例

```json
{
  "scenario_name": "Camera Scenario",
  "description": "相机任务场景",
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
    {
      "task_id": "T1",
      "name": "AimetlitePlus",
      "priority": "HIGH",
      "runtime_type": "ACPU_RUNTIME",
      "segmentation_strategy": "FORCED_SEGMENTATION",
      "fps": 30.0,
      "latency": 33.3,
      "dependencies": [],
      "model": {
        "model_name": "AimetlitePlus"
      }
    },
    {
      "task_id": "T2",
      "name": "CustomModel",
      "priority": "NORMAL",
      "runtime_type": "KERNEL_RUNTIME",
      "segmentation_strategy": "NO_SEGMENTATION",
      "fps": 15.0,
      "latency": 66.7,
      "dependencies": ["T1"],
      "model": {
        "segments": [
          {
            "resource_type": "NPU",
            "duration_table": {
              "40": 5.0,
              "80": 3.0,
              "120": 2.5
            },
            "segment_id": "main",
            "power": 200.0,
            "ddr": 10.0
          }
        ],
        "cut_points": {
          "main": [
            {
              "op_id": "op1",
              "perf_lut": {
                "40": 2.5,
                "80": 1.5
              },
              "overhead_ms": 0.0
            }
          ]
        }
      }
    }
  ]
}
```

## 枚举值

### TaskPriority
- `HIGH` - 高优先级
- `NORMAL` - 普通优先级
- `LOW` - 低优先级

### RuntimeType
- `ACPU_RUNTIME` - ACPU运行时
- `KERNEL_RUNTIME` - 内核运行时

### SegmentationStrategy
- `NO_SEGMENTATION` - 不分段
- `ADAPTIVE_SEGMENTATION` - 自适应分段
- `FORCED_SEGMENTATION` - 强制分段

### ResourceType
- `NPU` - 神经处理单元
- `DSP` - 数字信号处理器

## 预定义模型

使用 `GET /api/models` 或 `JsonInterface.get_available_models()` 获取完整列表。

主要包括：
- 相机模型：AimetlitePlus, FaceEhnsLite, Vmask, FaceDet, etc.
- YOLO模型：yolov8n_big, yolov8n_small
- 其他模型：motr, parsing, reid, pose2d, etc.

## 错误处理

API返回格式：

```json
{
  "success": false,
  "error": "错误信息",
  "traceback": "详细错误堆栈（调试模式）"
}
```

## 性能指标

调度完成后返回的指标包括：

```json
{
  "fps_metrics": {
    "T1": {
      "required_fps": 30.0,
      "achieved_fps": 28.5,
      "achievement_rate": 0.95
    }
  },
  "latency_metrics": {
    "T1": {
      "required_latency": 33.3,
      "max_latency": 35.0,
      "avg_latency": 32.0,
      "p99_latency": 34.5,
      "met_requirement": true
    }
  },
  "utilization": {
    "NPU": 85.5,
    "DSP": 62.3
  },
  "task_stats": {
    "T1": {
      "total_instances": 30,
      "completed_instances": 30,
      "completion_rate": 1.0
    }
  }
}
```