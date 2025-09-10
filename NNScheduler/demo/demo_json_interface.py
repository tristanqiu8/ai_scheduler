#!/usr/bin/env python3
"""
JSON接口使用示例
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNScheduler.interface.json_interface import JsonInterface
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.executor import ScheduleExecutor
from NNScheduler.core.evaluator import PerformanceEvaluator
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
import json


def demo_json_interface():
    """演示JSON接口的使用"""
    
    print("=" * 80)
    print("JSON Interface Demo")
    print("=" * 80)
    
    # 1. 创建示例配置
    print("\n1. 创建示例配置")
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
            {
                "task_id": "T1",
                "name": "AimetlitePlus",
                "priority": "HIGH",
                "runtime_type": "ACPU_RUNTIME",
                "segmentation_strategy": "FORCED_SEGMENTATION",
                "fps": 30.0,
                "latency": 33.3,
                "model": {
                    "model_name": "AimetlitePlus"  # 使用预定义模型
                }
            },
            {
                "task_id": "T2",
                "name": "FaceEhnsLite",
                "priority": "NORMAL",
                "runtime_type": "ACPU_RUNTIME",
                "segmentation_strategy": "FORCED_SEGMENTATION",
                "fps": 30.0,
                "latency": 33.3,
                "dependencies": ["T1"],
                "model": {
                    "model_name": "FaceEhnsLite"
                }
            },
            {
                "task_id": "T3",
                "name": "CustomModel",
                "priority": "LOW",
                "runtime_type": "DSP_RUNTIME",
                "segmentation_strategy": "NO_SEGMENTATION",
                "fps": 15.0,
                "latency": 66.7,
                "model": {
                    "segments": [
                        {
                            "resource_type": "NPU",
                            "duration_table": {40: 3.0, 80: 2.0, 120: 1.5},
                            "segment_id": "npu_main",
                            "power": 150.0,
                            "ddr": 8.0
                        },
                        {
                            "resource_type": "DSP",
                            "duration_table": {40: 1.5, 80: 1.0, 120: 0.8},
                            "segment_id": "dsp_post",
                            "power": 0.0,
                            "ddr": 0.0
                        }
                    ]
                }
            }
        ]
    }
    
    print(f"配置包含 {len(config['tasks'])} 个任务，{len(config['resources'])} 个资源")
    
    # 2. 保存配置到文件
    print("\n2. 保存配置到文件")
    JsonInterface.save_to_file(config, "demo_config.json")
    print("配置已保存到 demo_config.json")
    
    # 3. 从文件加载配置
    print("\n3. 从文件加载配置")
    loaded_config = JsonInterface.load_from_file("demo_config.json")
    print(f"加载的场景: {loaded_config['scenario_name']}")
    
    # 4. 解析资源配置
    print("\n4. 解析资源配置")
    resources = JsonInterface.parse_resource_config(loaded_config)
    for resource_id, res_config in resources.items():
        print(f"  {resource_id}: {res_config['type'].name} @ {res_config['bandwidth']} GB/s")
    
    # 5. 解析任务配置
    print("\n5. 解析任务配置")
    tasks = JsonInterface.parse_scenario_config(loaded_config)
    for task in tasks:
        print(f"  {task.task_id} ({task.name}): "
              f"优先级={task.priority.name}, "
              f"FPS={task.fps_requirement}, "
              f"段数={len(task.segments)}")
    
    # 6. 运行调度
    print("\n6. 运行调度")
    
    # 创建资源队列管理器
    queue_manager = ResourceQueueManager()
    for resource_id, res_config in resources.items():
        queue_manager.add_resource(resource_id, res_config["type"], res_config["bandwidth"])
    
    # 创建跟踪器和启动器
    tracer = ScheduleTracer(queue_manager)
    launcher = EnhancedTaskLauncher(queue_manager, tracer)
    
    # 注册任务
    for task in tasks:
        launcher.register_task(task)
    
    # 创建发射计划
    time_window = 200.0
    launch_plan = launcher.create_launch_plan(time_window, strategy="eager")
    
    # 将任务列表转换为字典
    task_dict = {task.task_id: task for task in tasks}
    
    # 创建执行器并运行
    executor = ScheduleExecutor(queue_manager, tracer, task_dict)
    executor.execute_plan(launch_plan, time_window)
    
    print(f"调度完成，时间窗口: {time_window}ms")
    
    # 7. 评估结果
    print("\n7. 评估调度结果")
    evaluator = PerformanceEvaluator(tracer, task_dict, queue_manager)
    overall_metrics = evaluator.evaluate(time_window)
    
    print("\nFPS达成情况:")
    for task_id, task_metrics in evaluator.task_metrics.items():
        print(f"  {task_id}: {task_metrics.achieved_fps:.1f}/{task_metrics.fps_requirement:.1f} FPS "
              f"({task_metrics.fps_achievement_rate:.1%})")
    
    print("\n资源利用率:")
    for resource_id, resource_metrics in evaluator.resource_metrics.items():
        print(f"  {resource_id}: {resource_metrics.utilization_rate:.1f}%")
    
    # 8. 导出任务为JSON
    print("\n8. 导出任务为JSON")
    exported_scenario = JsonInterface.export_scenario_to_json(
        tasks,
        scenario_name="Exported Scenario",
        description="从运行结果导出的场景"
    )
    JsonInterface.save_to_file(exported_scenario, "exported_scenario.json")
    print("导出的场景已保存到 exported_scenario.json")
    
    # 9. 生成可视化
    print("\n9. 生成可视化")
    visualizer = ScheduleVisualizer(tracer)
    visualizer.plot_resource_timeline(filename="json_demo_schedule.png")
    print("可视化已保存到 json_demo_schedule.png")
    
    # 10. 导出Chrome Trace
    print("\n10. 导出Chrome Trace")
    visualizer.export_chrome_tracing("json_demo_trace.json")
    print("Chrome Trace已保存到 json_demo_trace.json")


def demo_available_models():
    """演示获取可用模型"""
    print("\n" + "=" * 80)
    print("Available Models")
    print("=" * 80)
    
    models = JsonInterface.get_available_models()
    print(f"\n共有 {len(models)} 个预定义模型:")
    
    # 分类显示
    camera_models = [m for m in models if any(x in m.lower() for x in ['ml10t', 'aimet', 'face', 'vmask', 'cam', 'pd', 'af', 'nn'])]
    yolo_models = [m for m in models if 'yolo' in m.lower()]
    other_models = [m for m in models if m not in camera_models and m not in yolo_models]
    
    if camera_models:
        print("\n相机相关模型:")
        for model in camera_models:
            print(f"  - {model}")
    
    if yolo_models:
        print("\nYOLO模型:")
        for model in yolo_models:
            print(f"  - {model}")
    
    if other_models:
        print("\n其他模型:")
        for model in other_models:
            print(f"  - {model}")


def demo_custom_model():
    """演示创建自定义模型"""
    print("\n" + "=" * 80)
    print("Custom Model Creation")
    print("=" * 80)
    
    # 创建一个复杂的自定义模型配置
    custom_model_config = {
        "segments": [
            {
                "resource_type": "NPU",
                "duration_table": {20: 5.0, 40: 3.0, 80: 2.0, 120: 1.8},
                "segment_id": "backbone",
                "power": 250.0,
                "ddr": 12.0
            },
            {
                "resource_type": "DSP",
                "duration_table": {20: 1.0, 40: 0.8, 80: 0.6, 120: 0.5},
                "segment_id": "preprocess",
                "power": 0.0,
                "ddr": 0.0
            },
            {
                "resource_type": "NPU",
                "duration_table": {20: 3.0, 40: 2.0, 80: 1.5, 120: 1.2},
                "segment_id": "head",
                "power": 180.0,
                "ddr": 8.0
            },
            {
                "resource_type": "DSP",
                "duration_table": {20: 0.5, 40: 0.4, 80: 0.3, 120: 0.25},
                "segment_id": "postprocess",
                "power": 0.0,
                "ddr": 0.0
            }
        ],
        "cut_points": {
            "backbone": [
                {
                    "op_id": "conv_block_1",
                    "perf_lut": {40: 1.0, 80: 0.7, 120: 0.6},
                    "overhead_ms": 0.0
                },
                {
                    "op_id": "conv_block_2",
                    "perf_lut": {40: 1.0, 80: 0.7, 120: 0.6},
                    "overhead_ms": 0.0
                }
            ],
            "head": [
                {
                    "op_id": "detection_head",
                    "perf_lut": {40: 1.5, 80: 1.0, 120: 0.8},
                    "overhead_ms": 0.0
                }
            ]
        }
    }
    
    print("\n创建的自定义模型:")
    print(f"  段数: {len(custom_model_config['segments'])}")
    print(f"  资源类型: NPU + DSP混合")
    print(f"  可切分段: {list(custom_model_config['cut_points'].keys())}")
    
    # 解析模型
    segments, cut_points = JsonInterface.parse_model_config(custom_model_config)
    
    print("\n解析后的段信息:")
    for seg in segments:
        print(f"  {seg.segment_id}: {seg.resource_type.name}, "
              f"功耗={seg.power}mW, DDR={seg.ddr}MB")
    
    if cut_points:
        print("\n切分点信息:")
        for segment_id, points in cut_points.items():
            print(f"  {segment_id}: {len(points)} 个切分点")
            for point in points:
                print(f"    - {point.op_id}")


if __name__ == "__main__":
    # 运行所有演示
    demo_json_interface()
    demo_available_models()
    demo_custom_model()
    
    print("\n" + "=" * 80)
    print("JSON Interface Demo Completed!")
    print("=" * 80)