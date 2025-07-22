#!/usr/bin/env python3
"""
神经网络模型库 - 预定义的模型segments和cut points
使用core.models中的数据结构
"""

from typing import List, Dict, Tuple
from core.models import ResourceSegment, CutPoint
from core.enums import ResourceType


def create_parsing_model() -> List[ResourceSegment]:
    """3A Parsing模型: NPU主处理 + DSP后处理"""
    segments = []
    
    # NPU主段
    seg1 = ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 1.63, 40: 1.156, 80: 0.93, 120: 0.90},
        start_time=0,
        segment_id="main"
    )
    segments.append(seg1)
    
    # DSP后处理段
    seg2 = ResourceSegment(
        resource_type=ResourceType.DSP,
        duration_table={20: 0.48, 40: 0.455, 80: 0.46, 120: 0.45},
        start_time=0,
        segment_id="postprocess"
    )
    segments.append(seg2)
    
    return segments


def create_reid_model() -> List[ResourceSegment]:
    """ReID模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 4.24, 40: 2.864, 80: 2.56, 120: 2.52},
        start_time=0,
        segment_id="main"
    )]


def create_motr_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """MOTR模型: 9段混合任务 (4 DSP + 5 NPU) with cut points"""
    segments = []
    
    # 定义所有段
    segment_configs = [
        (ResourceType.DSP, {20: 0.316, 40: 0.305, 120: 0.368}, "dsp_s0"),
        (ResourceType.NPU, {20: 0.430, 40: 0.303, 120: 0.326}, "npu_s1"),
        (ResourceType.NPU, {20: 12.868, 40: 7.506, 120: 4.312}, "npu_s2"),
        (ResourceType.DSP, {20: 1.734, 40: 1.226, 120: 0.994}, "dsp_s1"),
        (ResourceType.NPU, {20: 0.997, 40: 0.374, 120: 0.211}, "npu_s3"),
        (ResourceType.DSP, {20: 1.734, 40: 1.201, 120: 0.943}, "dsp_s2"),
        (ResourceType.NPU, {20: 0.602, 40: 0.373, 120: 0.209}, "npu_s4"),
        (ResourceType.DSP, {20: 1.690, 40: 1.208, 120: 0.975}, "dsp_s3"),
        (ResourceType.NPU, {20: 0.596, 40: 0.321, 120: 0.134}, "npu_s4"),
    ]
    
    for resource_type, duration_table, segment_id in segment_configs:
        seg = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id
        )
        segments.append(seg)
    
    # 定义切分点
    cut_points = {
        "npu_s2": [
            CutPoint(op_id="cut1", perf_lut={40: 2.5}, overhead_ms=0.0),
            CutPoint(op_id="cut2", perf_lut={40: 2.5}, overhead_ms=0.0),
        ]
    }
    
    return segments, cut_points


def create_qim_model() -> List[ResourceSegment]:
    """qim模型: NPU + DSP混合"""
    return [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={10: 1.339, 20: 0.758, 40: 0.474, 80: 0.32, 120: 0.292},
            start_time=0,
            segment_id="npu_sub"
        ),
        ResourceSegment(
            resource_type=ResourceType.DSP,
            duration_table={10: 1.238, 20: 1.122, 40: 1.04, 80: 1, 120: 1.014},
            start_time=0,
            segment_id="dsp_sub"
        ),
    ]


def create_pose2d_model() -> List[ResourceSegment]:
    """pose2d模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 4.324, 40: 3.096, 80: 2.28, 120: 2.04},
        start_time=0,
        segment_id="main"
    )]


def create_tk_template_model() -> List[ResourceSegment]:
    """tk_template模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 0.48, 40: 0.33, 80: 0.27, 120: 0.25},
        start_time=0,
        segment_id="main"
    )]


def create_tk_search_model() -> List[ResourceSegment]:
    """tk_search模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 1.16, 40: 0.72, 80: 0.54, 120: 0.50},
        start_time=0,
        segment_id="main"
    )]


def create_graymask_model() -> List[ResourceSegment]:
    """GrayMask模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 2.42, 40: 2.00, 80: 1.82, 120: 1.80},
        start_time=0,
        segment_id="main"
    )]


def create_yolov8n_big_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """YOLOv8n大模型: 可分段NPU with cut points"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={20: 20.28, 40: 12.31, 120: 7.50},
            start_time=0,
            segment_id="main"
        )
    ]
    
    cut_points = {
        "main": [
            CutPoint(op_id="op6", perf_lut={20: 4.699, 40: 2.737, 120: 1.482}, overhead_ms=0.0),
            CutPoint(op_id="op13", perf_lut={20: 4.699, 40: 2.737, 120: 1.482}, overhead_ms=0.0),
            CutPoint(op_id="op14", perf_lut={20: 4.698, 40: 2.736, 120: 1.483}, overhead_ms=0.0),
            CutPoint(op_id="op19", perf_lut={20: 4.699, 40: 2.737, 120: 1.482}, overhead_ms=0.0),
        ]
    }
    
    return segments, cut_points


def create_yolov8n_small_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """YOLOv8n小模型: 可分段NPU with cut points"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={20: 5.02, 40: 3.16, 120: 2.03},
            start_time=0,
            segment_id="main"
        )
    ]
    
    cut_points = {
        "main": [
            CutPoint(op_id="op5", perf_lut={20: 1.138, 40: 0.691, 120: 0.418}, overhead_ms=0.0),
            CutPoint(op_id="op15", perf_lut={20: 1.138, 40: 0.691, 120: 0.417}, overhead_ms=0.0),
            CutPoint(op_id="op19", perf_lut={20: 2.275, 40: 1.381, 120: 0.835}, overhead_ms=0.0),
        ]
    }
    
    return segments, cut_points


def create_stereo4x_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """Stereo4x模型: 8段混合任务 (3 DSP + 5 NPU) with cut points"""
    segments = []
    
    segment_configs = [
        (ResourceType.NPU, {20: 4.347, 40: 2.730, 80: 2.002, 120: 1.867}, "npu_s0"),
        (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s0"),
        (ResourceType.NPU, {20: 2.900, 40: 2.016, 80: 1.642, 120: 1.608}, "npu_s1"),
        (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s1"),
        (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s2"),
        (ResourceType.NPU, {20: 1.456, 40: 1.046, 80: 0.791, 120: 0.832}, "npu_s2"),
        (ResourceType.NPU, {20: 1.456, 40: 1.115, 80: 0.932, 120: 0.924}, "npu_s3"),
        (ResourceType.NPU, {20: 8.780, 40: 6.761, 80: 5.712, 120: 5.699}, "npu_s4"),
    ]
    
    for resource_type, duration_table, segment_id in segment_configs:
        seg = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id
        )
        segments.append(seg)
    
    cut_points = {
        "npu_s4": [
            CutPoint(op_id="cut1", perf_lut={40: 2.111}, overhead_ms=0.0),
            CutPoint(op_id="cut2", perf_lut={40: 2.222}, overhead_ms=0.0),
        ]
    }
    
    return segments, cut_points


def create_skywater_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """Skywater小模型: 可分段NPU+DSP with cut points"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={20: 2.31, 40: 1.49, 80: 1.14, 120: 1.02},
            start_time=0,
            segment_id="main"
        ),
        ResourceSegment(
            resource_type=ResourceType.DSP,
            duration_table={20: 1.23, 40: 0.71, 80: 0.45, 120: 0.41},
            start_time=0,
            segment_id="postprocess"
        ),
    ]
    
    cut_points = {
        "main": [
            CutPoint(op_id="op4", perf_lut={20: 0.924, 40: 0.596, 80: 0.456, 120: 0.408}, overhead_ms=0.0),
            CutPoint(op_id="op14", perf_lut={20: 0.277, 40: 0.179, 80: 0.137, 120: 0.122}, overhead_ms=0.0),
        ]
    }
    
    return segments, cut_points


def create_peak_detector_model() -> List[ResourceSegment]:
    """PeakDetector模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 1.51, 40: 0.97, 80: 0.70, 120: 0.62},
        start_time=0,
        segment_id="main"
    )]


def create_skywater_big_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """Skywater大模型: 可分段NPU+DSP with cut points"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={20: 4.19, 40: 2.49, 80: 1.70, 120: 1.67},
            start_time=0,
            segment_id="main"
        ),
        ResourceSegment(
            resource_type=ResourceType.DSP,
            duration_table={20: 1.52, 40: 0.90, 80: 0.58, 120: 0.58},
            start_time=0,
            segment_id="postprocess"
        ),
    ]
    
    cut_points = {
        "main": [
            CutPoint(op_id="op4", perf_lut={20: 1.676, 40: 0.996, 80: 0.680, 120: 0.668}, overhead_ms=0.0),
            CutPoint(op_id="op14", perf_lut={20: 0.503, 40: 0.299, 80: 0.204, 120: 0.200}, overhead_ms=0.0),
        ]
    }
    
    return segments, cut_points


def create_bonus_task_model() -> List[ResourceSegment]:
    """BonusTask模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={40: 7.5},
        start_time=0,
        segment_id="main"
    )]


# 模型注册表
MODEL_REGISTRY = {
    "parsing": create_parsing_model,
    "reid": create_reid_model,
    "motr": create_motr_model,
    "qim": create_qim_model,
    "pose2d": create_pose2d_model,
    "tk_template": create_tk_template_model,
    "tk_search": create_tk_search_model,
    "graymask": create_graymask_model,
    "yolov8n_big": create_yolov8n_big_model,
    "yolov8n_small": create_yolov8n_small_model,
    "stereo4x": create_stereo4x_model,
    "skywater": create_skywater_model,
    "peak_detector": create_peak_detector_model,
    "skywater_big": create_skywater_big_model,
    "bonus_task": create_bonus_task_model,
}


def get_model(model_name: str):
    """获取模型定义
    
    Args:
        model_name: 模型名称
        
    Returns:
        对于简单模型: List[ResourceSegment]
        对于带切分点的模型: Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    return MODEL_REGISTRY[model_name]()


def list_models() -> List[str]:
    """列出所有可用的模型"""
    return list(MODEL_REGISTRY.keys())