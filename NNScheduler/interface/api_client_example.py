#!/usr/bin/env python3
"""
Web API 客户端示例
"""

import requests
import json
from typing import Dict, Any

class SchedulerAPIClient:
    """AI Scheduler API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url
    
    def check_health(self) -> Dict[str, Any]:
        """健康检查"""
        response = requests.get(f"{self.base_url}/api/health")
        return response.json()
    
    def get_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        response = requests.get(f"{self.base_url}/api/models")
        return response.json()
    
    def get_example(self) -> Dict[str, Any]:
        """获取示例配置"""
        response = requests.get(f"{self.base_url}/api/example")
        return response.json()
    
    def create_tasks(self, tasks_config: Dict[str, Any]) -> Dict[str, Any]:
        """创建任务"""
        response = requests.post(
            f"{self.base_url}/api/tasks",
            json=tasks_config
        )
        return response.json()
    
    def get_tasks(self) -> Dict[str, Any]:
        """获取任务列表"""
        response = requests.get(f"{self.base_url}/api/tasks")
        return response.json()
    
    def set_resources(self, resources_config: Dict[str, Any]) -> Dict[str, Any]:
        """设置资源"""
        response = requests.post(
            f"{self.base_url}/api/resources",
            json=resources_config
        )
        return response.json()
    
    def get_resources(self) -> Dict[str, Any]:
        """获取资源配置"""
        response = requests.get(f"{self.base_url}/api/resources")
        return response.json()
    
    def run_schedule(self, time_window: float = 1000.0, 
                    segment_mode: bool = True,
                    launcher: str = "enhanced") -> Dict[str, Any]:
        """运行调度"""
        response = requests.post(
            f"{self.base_url}/api/schedule",
            json={
                "time_window": time_window,
                "segment_mode": segment_mode,
                "launcher": launcher
            }
        )
        return response.json()
    
    def get_trace(self) -> Dict[str, Any]:
        """获取调度跟踪数据"""
        response = requests.get(f"{self.base_url}/api/schedule/trace")
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        response = requests.get(f"{self.base_url}/api/schedule/metrics")
        return response.json()
    
    def generate_visualization(self, output_path: str = "schedule.png") -> Dict[str, Any]:
        """生成可视化"""
        response = requests.post(
            f"{self.base_url}/api/schedule/visualization",
            json={
                "output_path": output_path,
                "format": "png"
            }
        )
        return response.json()
    
    def load_config(self, config_or_path: Any) -> Dict[str, Any]:
        """加载配置"""
        if isinstance(config_or_path, str):
            # 如果是文件路径
            data = {"filepath": config_or_path}
        else:
            # 如果是配置字典
            data = config_or_path
        
        response = requests.post(
            f"{self.base_url}/api/config/load",
            json=data
        )
        return response.json()
    
    def save_config(self, filepath: str, scenario_name: str = "", description: str = "") -> Dict[str, Any]:
        """保存配置"""
        response = requests.post(
            f"{self.base_url}/api/config/save",
            json={
                "filepath": filepath,
                "scenario_name": scenario_name,
                "description": description
            }
        )
        return response.json()
    
    def reset(self) -> Dict[str, Any]:
        """重置调度器"""
        response = requests.post(f"{self.base_url}/api/reset")
        return response.json()


def example_usage():
    """使用示例"""
    
    # 创建客户端
    client = SchedulerAPIClient()
    
    print("1. 健康检查")
    print(client.check_health())
    print()
    
    print("2. 获取可用模型")
    models = client.get_models()
    print(f"Available models: {models['models'][:5]}...")
    print()
    
    print("3. 获取示例配置")
    example = client.get_example()
    print(f"Example has {len(example['example']['tasks'])} tasks")
    print()
    
    print("4. 使用示例配置")
    # 加载示例配置
    config = example['example']
    result = client.load_config(config)
    print(f"Loaded: {result}")
    print()
    
    print("5. 运行调度")
    schedule_result = client.run_schedule(
        time_window=1000.0,
        segment_mode=True,
        launcher="enhanced"
    )
    if schedule_result['success']:
        print(f"Schedule completed: {schedule_result['summary']}")
        print(f"FPS metrics: {schedule_result['results']['fps_metrics']}")
    print()
    
    print("6. 获取性能指标")
    metrics = client.get_metrics()
    if metrics['success']:
        print(f"Utilization: NPU={metrics['metrics']['utilization']['NPU']:.1f}%, "
              f"DSP={metrics['metrics']['utilization']['DSP']:.1f}%")
    print()
    
    print("7. 生成可视化")
    viz_result = client.generate_visualization("api_test_schedule.png")
    print(f"Visualization: {viz_result}")
    print()
    
    print("8. 保存配置")
    save_result = client.save_config(
        "api_test_config.json",
        scenario_name="API Test Scenario",
        description="Created via API"
    )
    print(f"Save result: {save_result}")


def custom_scenario_example():
    """自定义场景示例"""
    
    client = SchedulerAPIClient()
    
    # 重置状态
    client.reset()
    
    # 1. 设置资源
    resources_config = {
        "resources": [
            {
                "resource_id": "NPU_0",
                "resource_type": "NPU",
                "bandwidth": 80.0  # 80GB/s
            },
            {
                "resource_id": "NPU_1",
                "resource_type": "NPU",
                "bandwidth": 80.0
            },
            {
                "resource_id": "DSP_0",
                "resource_type": "DSP",
                "bandwidth": 40.0
            }
        ]
    }
    print("Setting resources...")
    print(client.set_resources(resources_config))
    
    # 2. 创建任务
    tasks_config = {
        "scenario_name": "Custom Multi-NPU Scenario",
        "tasks": [
            {
                "task_id": "T1",
                "name": "Heavy_NPU_Task",
                "priority": "HIGH",
                "runtime_type": "ACPU_RUNTIME",
                "segmentation_strategy": "FORCED_SEGMENTATION",
                "fps": 30.0,
                "latency": 33.3,
                "model": {
                    "model_name": "yolov8n_big"  # 使用预定义的大模型
                }
            },
            {
                "task_id": "T2",
                "name": "Light_NPU_Task",
                "priority": "NORMAL",
                "runtime_type": "ACPU_RUNTIME",
                "segmentation_strategy": "NO_SEGMENTATION",
                "fps": 60.0,
                "latency": 16.7,
                "model": {
                    "model_name": "tk_template"  # 使用预定义的轻量模型
                }
            },
            {
                "task_id": "T3",
                "name": "Mixed_Task",
                "priority": "LOW",
                "runtime_type": "KERNEL_RUNTIME",
                "segmentation_strategy": "ADAPTIVE_SEGMENTATION",
                "fps": 15.0,
                "latency": 66.7,
                "dependencies": ["T1"],  # 依赖T1
                "model": {
                    "model_name": "motr"  # 使用混合NPU+DSP模型
                }
            }
        ]
    }
    print("\nCreating tasks...")
    print(client.create_tasks(tasks_config))
    
    # 3. 运行调度
    print("\nRunning schedule...")
    result = client.run_schedule(
        time_window=200.0,  # 200ms窗口
        segment_mode=True,
        launcher="enhanced"
    )
    
    if result['success']:
        print(f"\nSchedule Results:")
        print(f"  Total tasks: {result['summary']['total_tasks']}")
        print(f"  Time window: {result['summary']['time_window']}ms")
        print(f"  Segment mode: {result['summary']['segment_mode']}")
        
        print(f"\nFPS Achievement:")
        for task_id, metrics in result['results']['fps_metrics'].items():
            print(f"  {task_id}: {metrics['achieved_fps']:.1f}/{metrics['required_fps']:.1f} FPS "
                  f"({metrics['achievement_rate']:.1%})")
        
        print(f"\nResource Utilization:")
        for resource, util in result['results']['utilization'].items():
            print(f"  {resource}: {util:.1f}%")
    
    # 4. 生成可视化
    print("\nGenerating visualization...")
    viz_result = client.generate_visualization("custom_scenario.png")
    print(f"Visualization saved: {viz_result['path']}")
    
    # 5. 保存配置供后续使用
    print("\nSaving configuration...")
    save_result = client.save_config(
        "custom_scenario_config.json",
        scenario_name="Custom Multi-NPU Scenario",
        description="2xNPU + 1xDSP configuration with mixed tasks"
    )
    print(f"Configuration saved: {save_result['path']}")


if __name__ == "__main__":
    print("=" * 60)
    print("AI Scheduler API Client Examples")
    print("=" * 60)
    
    print("\n[Example 1] Basic Usage")
    print("-" * 40)
    example_usage()
    
    print("\n" * 2)
    print("[Example 2] Custom Scenario")
    print("-" * 40)
    custom_scenario_example()
    
    print("\n✅ Examples completed!")