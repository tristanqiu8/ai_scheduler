#!/usr/bin/env python3
"""
JSON 接口层 - 提供JSON格式的输入输出接口
"""

import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import asdict
from NNScheduler.core.models import ResourceSegment, CutPoint
from NNScheduler.core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from NNScheduler.core.artifacts import ensure_artifact_path
from NNScheduler.core.task import NNTask
from NNScheduler.scenario.model_repo import get_model, list_models


class JsonInterface:
    """JSON接口处理器"""
    
    @staticmethod
    def parse_model_config(config: Dict[str, Any]) -> Tuple[List[ResourceSegment], Optional[Dict[str, List[CutPoint]]]]:
        """
        解析模型配置JSON
        
        输入格式:
        {
            "model_name": "string",  # 使用预定义模型
            # 或者
            "segments": [
                {
                    "resource_type": "NPU/DSP",
                    "duration_table": {20: 1.5, 40: 1.0, ...},
                    "segment_id": "string",
                    "power": float,
                    "ddr": float
                }
            ],
            "cut_points": {  # 可选
                "segment_id": [
                    {
                        "op_id": "string",
                        "perf_lut": {40: 2.5, ...},
                        "overhead_ms": 0.0
                    }
                ]
            }
        }
        """
        # 如果指定了预定义模型
        if "model_name" in config:
            return get_model(config["model_name"])
        
        # 否则解析自定义模型
        segments = []
        for seg_config in config.get("segments", []):
            # 将duration_table的键转换为float类型
            duration_table = {
                float(k): v for k, v in seg_config["duration_table"].items()
            }
            
            segment = ResourceSegment(
                resource_type=ResourceType[seg_config["resource_type"]],
                duration_table=duration_table,
                start_time=seg_config.get("start_time", 0),
                segment_id=seg_config["segment_id"],
                power=seg_config.get("power", 0.0),
                ddr=seg_config.get("ddr", 0.0)
            )
            segments.append(segment)
        
        # 解析切分点（如果有）
        cut_points = None
        if "cut_points" in config:
            cut_points = {}
            for segment_id, points in config["cut_points"].items():
                cut_points[segment_id] = []
                for point_config in points:
                    # 将perf_lut的键转换为float类型
                    perf_lut = {
                        float(k): v for k, v in point_config["perf_lut"].items()
                    }
                    
                    cut_point = CutPoint(
                        op_id=point_config["op_id"],
                        perf_lut=perf_lut,
                        overhead_ms=point_config.get("overhead_ms", 0.0)
                    )
                    cut_points[segment_id].append(cut_point)
        
        return segments, cut_points
    
    @staticmethod
    def parse_task_config(config: Dict[str, Any]) -> NNTask:
        """
        解析任务配置JSON
        
        输入格式:
        {
            "task_id": "string",
            "name": "string",
            "priority": "HIGH/NORMAL/LOW",
            "runtime_type": "acpu_runtime/kernel_runtime",
            "segmentation_strategy": "NO_SEGMENTATION/ADAPTIVE_SEGMENTATION/FORCED_SEGMENTATION",
            "fps": float,
            "latency": float,
            "dependencies": ["task_id1", "task_id2"],  # 可选
            "model": {  # 模型配置，格式同parse_model_config
                ...
            }
        }
        """
        task = NNTask(
            task_id=config["task_id"],
            name=config["name"],
            priority=TaskPriority[config.get("priority", "NORMAL")],
            runtime_type=RuntimeType[config.get("runtime_type", "ACPU_RUNTIME")],
            segmentation_strategy=SegmentationStrategy[config.get("segmentation_strategy", "NO_SEGMENTATION")]
        )
        
        # 设置性能要求
        task.set_performance_requirements(
            fps=config.get("fps", 30.0),
            latency=config.get("latency", 1000.0/30.0)
        )
        
        # 添加依赖
        if "dependencies" in config:
            for dep in config["dependencies"]:
                task.add_dependency(dep)
        
        # 应用模型
        if "model" in config:
            model_result = JsonInterface.parse_model_config(config["model"])
            task.apply_model(model_result)
        
        return task
    
    @staticmethod
    def parse_scenario_config(config: Dict[str, Any]) -> List[NNTask]:
        """
        解析场景配置JSON（包含多个任务）
        
        输入格式:
        {
            "scenario_name": "string",
            "description": "string",
            "tasks": [
                {task_config_1},
                {task_config_2},
                ...
            ]
        }
        """
        tasks = []
        for task_config in config.get("tasks", []):
            task = JsonInterface.parse_task_config(task_config)
            tasks.append(task)
        return tasks
    
    @staticmethod
    def parse_resource_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析资源配置JSON
        
        输入格式:
        {
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
        """
        resources = {}
        for res_config in config.get("resources", []):
            resources[res_config["resource_id"]] = {
                "type": ResourceType[res_config["resource_type"]],
                "bandwidth": res_config["bandwidth"]
            }
        return resources
    
    @staticmethod
    def export_task_to_json(task: NNTask) -> Dict[str, Any]:
        """将任务导出为JSON格式"""
        result = {
            "task_id": task.task_id,
            "name": task.name,
            "priority": task.priority.name,
            "runtime_type": task.runtime_type.name,
            "segmentation_strategy": task.segmentation_strategy.name,
            "fps": task.fps_requirement,
            "latency": task.latency_requirement,
            "dependencies": list(task.dependencies)
        }
        
        # 导出段信息
        segments = []
        for seg in task.segments:
            segments.append({
                "resource_type": seg.resource_type.name,
                "duration_table": seg.duration_table,
                "segment_id": seg.segment_id,
                "power": seg.power,
                "ddr": seg.ddr
            })
        result["segments"] = segments
        
        # 导出切分点信息（如果有）
        if hasattr(task, 'cut_points') and task.cut_points:
            cut_points = {}
            for segment_id, points in task.cut_points.items():
                cut_points[segment_id] = []
                for point in points:
                    cut_points[segment_id].append({
                        "op_id": point.op_id,
                        "perf_lut": point.perf_lut,
                        "overhead_ms": point.overhead_ms
                    })
            result["cut_points"] = cut_points
        
        return result
    
    @staticmethod
    def export_scenario_to_json(tasks: List[NNTask], scenario_name: str = "", description: str = "") -> Dict[str, Any]:
        """将任务集导出为场景JSON"""
        return {
            "scenario_name": scenario_name,
            "description": description,
            "tasks": [JsonInterface.export_task_to_json(task) for task in tasks]
        }
    
    @staticmethod
    def load_from_file(filepath: str) -> Dict[str, Any]:
        """从文件加载JSON配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def save_to_file(data: Dict[str, Any], filepath: str):
        """保存JSON配置到文件"""
        output_path = ensure_artifact_path(filepath)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return output_path
    
    @staticmethod
    def get_available_models() -> List[str]:
        """获取所有可用的预定义模型"""
        return list_models()
    
    @staticmethod
    def create_example_config() -> Dict[str, Any]:
        """创建示例配置"""
        return {
            "scenario_name": "Camera Example",
            "description": "示例相机任务场景",
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
                        "model_name": "AimetlitePlus"
                    }
                },
                {
                    "task_id": "T2",
                    "name": "FaceDetection",
                    "priority": "NORMAL",
                    "runtime_type": "ACPU_RUNTIME",
                    "segmentation_strategy": "NO_SEGMENTATION",
                    "fps": 15.0,
                    "latency": 66.6,
                    "dependencies": ["T1"],
                    "model": {
                        "model_name": "FaceDet"
                    }
                },
                {
                    "task_id": "T3",
                    "name": "CustomModel",
                    "priority": "LOW",
                    "runtime_type": "DSP_RUNTIME",
                    "segmentation_strategy": "ADAPTIVE_SEGMENTATION",
                    "fps": 10.0,
                    "latency": 100.0,
                    "model": {
                        "segments": [
                            {
                                "resource_type": "NPU",
                                "duration_table": {40: 5.0, 80: 3.0, 120: 2.5},
                                "segment_id": "main",
                                "power": 200.0,
                                "ddr": 10.0
                            },
                            {
                                "resource_type": "DSP",
                                "duration_table": {40: 2.0, 80: 1.5, 120: 1.2},
                                "segment_id": "postprocess",
                                "power": 0.0,
                                "ddr": 0.0
                            }
                        ],
                        "cut_points": {
                            "main": [
                                {
                                    "op_id": "op1",
                                    "perf_lut": {40: 2.5, 80: 1.5, 120: 1.25},
                                    "overhead_ms": 0.0
                                }
                            ]
                        }
                    }
                }
            ]
        }
