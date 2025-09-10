#!/usr/bin/env python3
"""
Web API 层 - 基于Flask的RESTful API接口
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import traceback
from typing import Dict, Any, List
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNScheduler.interface.json_interface import JsonInterface
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.executor import ScheduleExecutor
from NNScheduler.core.evaluator import PerformanceEvaluator
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer

app = Flask(__name__)
CORS(app)  # 启用跨域访问

# 全局存储
class SchedulerState:
    """调度器状态管理"""
    def __init__(self):
        self.tasks = []
        self.resources = {}
        self.schedule_results = None
        self.tracer = None
        self.evaluator = None
        
scheduler_state = SchedulerState()


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "message": "AI Scheduler API is running"
    })


@app.route('/api/models', methods=['GET'])
def get_available_models():
    """获取所有可用的预定义模型"""
    try:
        models = JsonInterface.get_available_models()
        return jsonify({
            "success": True,
            "models": models
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/example', methods=['GET'])
def get_example_config():
    """获取示例配置"""
    try:
        example = JsonInterface.create_example_config()
        return jsonify({
            "success": True,
            "example": example
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/tasks', methods=['POST'])
def create_tasks():
    """创建任务
    
    请求体格式:
    {
        "tasks": [task_config1, task_config2, ...]
    }
    或
    {
        "scenario_name": "...",
        "description": "...",
        "tasks": [...]
    }
    """
    try:
        data = request.json
        
        # 解析任务配置
        if "scenario_name" in data:
            tasks = JsonInterface.parse_scenario_config(data)
        else:
            tasks = []
            for task_config in data.get("tasks", []):
                task = JsonInterface.parse_task_config(task_config)
                tasks.append(task)
        
        # 保存任务
        scheduler_state.tasks = tasks
        
        # 返回任务信息
        task_info = []
        for task in tasks:
            task_info.append({
                "task_id": task.task_id,
                "name": task.name,
                "priority": task.priority.name,
                "fps": task.fps_requirement,
                "latency": task.latency_requirement,
                "segments": len(task.segments)
            })
        
        return jsonify({
            "success": True,
            "message": f"Created {len(tasks)} tasks",
            "tasks": task_info
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400


@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取当前任务列表"""
    try:
        tasks_json = []
        for task in scheduler_state.tasks:
            tasks_json.append(JsonInterface.export_task_to_json(task))
        
        return jsonify({
            "success": True,
            "tasks": tasks_json
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/resources', methods=['POST'])
def set_resources():
    """设置资源配置
    
    请求体格式:
    {
        "resources": [
            {
                "resource_id": "NPU_0",
                "resource_type": "NPU",
                "bandwidth": 40.0
            },
            ...
        ]
    }
    """
    try:
        data = request.json
        resources = JsonInterface.parse_resource_config(data)
        scheduler_state.resources = resources
        
        return jsonify({
            "success": True,
            "message": f"Configured {len(resources)} resources",
            "resources": list(resources.keys())
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route('/api/resources', methods=['GET'])
def get_resources():
    """获取当前资源配置"""
    try:
        resources_list = []
        for resource_id, config in scheduler_state.resources.items():
            resources_list.append({
                "resource_id": resource_id,
                "resource_type": config["type"].name,
                "bandwidth": config["bandwidth"]
            })
        
        return jsonify({
            "success": True,
            "resources": resources_list
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/schedule', methods=['POST'])
def run_schedule():
    """运行调度
    
    请求体格式（可选）:
    {
        "time_window": 1000.0,  # 调度时间窗口（ms）
        "segment_mode": true,    # 是否使用段级调度
        "launcher": "enhanced"   # 使用的启动器类型：basic/enhanced
    }
    """
    try:
        data = request.json or {}
        time_window = data.get("time_window", 1000.0)
        segment_mode = data.get("segment_mode", True)
        launcher_type = data.get("launcher", "enhanced")
        
        # 检查是否有任务和资源
        if not scheduler_state.tasks:
            return jsonify({
                "success": False,
                "error": "No tasks configured. Please create tasks first."
            }), 400
        
        if not scheduler_state.resources:
            return jsonify({
                "success": False,
                "error": "No resources configured. Please set resources first."
            }), 400
        
        # 创建资源队列管理器
        queue_manager = ResourceQueueManager()
        for resource_id, config in scheduler_state.resources.items():
            queue_manager.add_resource(resource_id, config["type"], config["bandwidth"])
        
        # 创建跟踪器
        tracer = ScheduleTracer()
        
        # 选择启动器
        if launcher_type == "enhanced":
            launcher = EnhancedTaskLauncher(queue_manager, tracer, segment_mode=segment_mode)
        else:
            launcher = TaskLauncher(queue_manager, tracer, segment_mode=segment_mode)
        
        # 创建执行器
        executor = ScheduleExecutor(launcher, tracer)
        
        # 执行调度
        executor.execute(scheduler_state.tasks, time_window)
        
        # 保存结果
        scheduler_state.tracer = tracer
        scheduler_state.evaluator = PerformanceEvaluator(tracer)
        
        # 获取调度结果摘要
        results = scheduler_state.evaluator.evaluate_schedule(scheduler_state.tasks, time_window)
        
        return jsonify({
            "success": True,
            "message": "Schedule completed",
            "summary": {
                "total_tasks": len(scheduler_state.tasks),
                "time_window": time_window,
                "segment_mode": segment_mode,
                "launcher": launcher_type
            },
            "results": {
                "fps_metrics": results["fps_metrics"],
                "latency_metrics": results["latency_metrics"],
                "utilization": results["utilization"],
                "task_stats": results["task_stats"]
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/schedule/trace', methods=['GET'])
def get_schedule_trace():
    """获取调度跟踪数据（Chrome Trace格式）"""
    try:
        if not scheduler_state.tracer:
            return jsonify({
                "success": False,
                "error": "No schedule results available. Please run schedule first."
            }), 400
        
        trace_data = scheduler_state.tracer.export_chrome_trace()
        
        return jsonify({
            "success": True,
            "trace": trace_data
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/schedule/metrics', methods=['GET'])
def get_schedule_metrics():
    """获取调度性能指标"""
    try:
        if not scheduler_state.evaluator:
            return jsonify({
                "success": False,
                "error": "No schedule results available. Please run schedule first."
            }), 400
        
        results = scheduler_state.evaluator.evaluate_schedule(scheduler_state.tasks, 1000.0)
        
        return jsonify({
            "success": True,
            "metrics": results
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/schedule/visualization', methods=['POST'])
def generate_visualization():
    """生成调度可视化
    
    请求体格式:
    {
        "output_path": "schedule_result.png",
        "format": "png"  # 支持 png, html
    }
    """
    try:
        if not scheduler_state.tracer:
            return jsonify({
                "success": False,
                "error": "No schedule results available. Please run schedule first."
            }), 400
        
        data = request.json or {}
        output_path = data.get("output_path", "schedule_result.png")
        output_format = data.get("format", "png")
        
        visualizer = ScheduleVisualizer(scheduler_state.tracer)
        
        if output_format == "png":
            visualizer.plot_timeline(save_path=output_path)
        elif output_format == "html":
            # 可以扩展支持HTML格式的交互式可视化
            return jsonify({
                "success": False,
                "error": "HTML format not yet implemented"
            }), 501
        else:
            return jsonify({
                "success": False,
                "error": f"Unsupported format: {output_format}"
            }), 400
        
        return jsonify({
            "success": True,
            "message": f"Visualization saved to {output_path}",
            "path": output_path
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/config/load', methods=['POST'])
def load_config():
    """从文件或JSON加载完整配置
    
    请求体格式:
    {
        "filepath": "config.json"  # 文件路径
    }
    或直接传入配置JSON
    """
    try:
        data = request.json
        
        if "filepath" in data:
            # 从文件加载
            config = JsonInterface.load_from_file(data["filepath"])
        else:
            # 直接使用传入的配置
            config = data
        
        # 解析资源配置
        if "resources" in config:
            resources = JsonInterface.parse_resource_config(config)
            scheduler_state.resources = resources
        
        # 解析任务配置
        if "tasks" in config:
            tasks = JsonInterface.parse_scenario_config(config)
            scheduler_state.tasks = tasks
        
        return jsonify({
            "success": True,
            "message": "Configuration loaded successfully",
            "summary": {
                "resources": len(scheduler_state.resources),
                "tasks": len(scheduler_state.tasks)
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 400


@app.route('/api/config/save', methods=['POST'])
def save_config():
    """保存当前配置到文件
    
    请求体格式:
    {
        "filepath": "config.json",
        "scenario_name": "My Scenario",
        "description": "Description of the scenario"
    }
    """
    try:
        data = request.json
        filepath = data.get("filepath", "config.json")
        scenario_name = data.get("scenario_name", "")
        description = data.get("description", "")
        
        # 构建配置
        config = {
            "scenario_name": scenario_name,
            "description": description,
            "resources": [],
            "tasks": []
        }
        
        # 添加资源配置
        for resource_id, res_config in scheduler_state.resources.items():
            config["resources"].append({
                "resource_id": resource_id,
                "resource_type": res_config["type"].name,
                "bandwidth": res_config["bandwidth"]
            })
        
        # 添加任务配置
        for task in scheduler_state.tasks:
            config["tasks"].append(JsonInterface.export_task_to_json(task))
        
        # 保存到文件
        JsonInterface.save_to_file(config, filepath)
        
        return jsonify({
            "success": True,
            "message": f"Configuration saved to {filepath}",
            "path": filepath
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset_scheduler():
    """重置调度器状态"""
    global scheduler_state
    scheduler_state = SchedulerState()
    
    return jsonify({
        "success": True,
        "message": "Scheduler state reset successfully"
    })


if __name__ == '__main__':
    print("Starting AI Scheduler Web API...")
    print("API Documentation:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/models          - List available models")
    print("  GET  /api/example         - Get example configuration")
    print("  POST /api/tasks           - Create tasks")
    print("  GET  /api/tasks           - Get current tasks")
    print("  POST /api/resources       - Set resources")
    print("  GET  /api/resources       - Get current resources")
    print("  POST /api/schedule        - Run schedule")
    print("  GET  /api/schedule/trace  - Get schedule trace")
    print("  GET  /api/schedule/metrics - Get schedule metrics")
    print("  POST /api/schedule/visualization - Generate visualization")
    print("  POST /api/config/load     - Load configuration")
    print("  POST /api/config/save     - Save configuration")
    print("  POST /api/reset           - Reset scheduler")
    print()
    
    app.run(host='0.0.0.0', port=5000, debug=True)