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
        segment_id="main",
        power=150.0,  # 150mW per frame
        ddr=8.0  # 8MB per frame
    )
    segments.append(seg1)
    
    # DSP后处理段
    seg2 = ResourceSegment(
        resource_type=ResourceType.DSP,
        duration_table={20: 0.48, 40: 0.455, 80: 0.46, 120: 0.45},
        start_time=0,
        segment_id="postprocess",
        power=0.0,  # DSP power set to 0
        ddr=0.0  # DSP DDR set to 0
    )
    segments.append(seg2)
    
    return segments


def create_reid_model() -> List[ResourceSegment]:
    """ReID模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 4.24, 40: 2.864, 80: 2.56, 120: 2.52},
        start_time=0,
        segment_id="main",
        power=250.0,  # 250mW per frame
        ddr=12.0  # 12MB per frame
    )]


def create_motr_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """MOTR模型: 9段混合任务 (4 DSP + 5 NPU) with cut points"""
    segments = []
    
    # 定义所有段，添加功耗和DDR带宽
    segment_configs = [
        (ResourceType.DSP, {20: 0.316, 40: 0.305, 120: 0.368}, "dsp_s0", 0.0, 0.0),  # power=0, ddr=0 for DSP
        (ResourceType.NPU, {20: 0.430, 40: 0.303, 120: 0.326}, "npu_s1", 60.0, 3.0),
        (ResourceType.NPU, {20: 12.868, 40: 7.506, 120: 4.312}, "npu_s2", 450.0, 25.0),
        (ResourceType.DSP, {20: 1.734, 40: 1.226, 120: 0.994}, "dsp_s1", 0.0, 0.0),
        (ResourceType.NPU, {20: 0.997, 40: 0.374, 120: 0.211}, "npu_s3", 80.0, 4.0),
        (ResourceType.DSP, {20: 1.734, 40: 1.201, 120: 0.943}, "dsp_s2", 0.0, 0.0),
        (ResourceType.NPU, {20: 0.602, 40: 0.373, 120: 0.209}, "npu_s4", 70.0, 3.5),
        (ResourceType.DSP, {20: 1.690, 40: 1.208, 120: 0.975}, "dsp_s3", 0.0, 0.0),
        (ResourceType.NPU, {20: 0.596, 40: 0.321, 120: 0.134}, "npu_s4", 65.0, 3.2),
    ]
    
    for resource_type, duration_table, segment_id, power, ddr in segment_configs:
        seg = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id,
            power=power,
            ddr=ddr
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
            segment_id="npu_sub",
            power=100.0,  # 100mW per frame
            ddr=6.0  # 6MB per frame
        ),
        ResourceSegment(
            resource_type=ResourceType.DSP,
            duration_table={10: 1.238, 20: 1.122, 40: 1.04, 80: 1, 120: 1.014},
            start_time=0,
            segment_id="dsp_sub",
            power=0.0,  # DSP power set to 0
            ddr=0.0  # DSP DDR set to 0
        ),
    ]


def create_pose2d_model() -> List[ResourceSegment]:
    """pose2d模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 4.324, 40: 3.096, 80: 2.28, 120: 2.04},
        start_time=0,
        segment_id="main",
        power=280.0,  # 280mW per frame
        ddr=14.0  # 14MB per frame
    )]


def create_tk_template_model() -> List[ResourceSegment]:
    """tk_template模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 0.48, 40: 0.33, 80: 0.27, 120: 0.25},
        start_time=0,
        segment_id="main",
        power=50.0,  # 50mW per frame
        ddr=2.5  # 2.5MB per frame
    )]


def create_tk_search_model() -> List[ResourceSegment]:
    """tk_search模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 1.16, 40: 0.72, 80: 0.54, 120: 0.50},
        start_time=0,
        segment_id="main",
        power=90.0,  # 90mW per frame
        ddr=5.0  # 5MB per frame
    )]


def create_graymask_model() -> List[ResourceSegment]:
    """GrayMask模型: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 2.42, 40: 2.00, 80: 1.82, 120: 1.80},
        start_time=0,
        segment_id="main",
        power=180.0,  # 180mW per frame
        ddr=10.0  # 10MB per frame
    )]


def create_yolov8n_big_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """YOLOv8n大模型: 可分段NPU with cut points"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={20: 20.28, 40: 12.31, 120: 7.50},
            start_time=0,
            segment_id="main",
            power=500.0,  # 500mW per frame
            ddr=30.0  # 30MB per frame
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
            segment_id="main",
            power=200.0,  # 200mW per frame
            ddr=15.0  # 15MB per frame
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
        (ResourceType.NPU, {20: 4.347, 40: 2.730, 80: 2.002, 120: 1.867}, "npu_s0", 220.0, 12.0),
        (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s0", 0.0, 0.0),
        (ResourceType.NPU, {20: 2.900, 40: 2.016, 80: 1.642, 120: 1.608}, "npu_s1", 180.0, 9.0),
        (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s1", 0.0, 0.0),
        (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s2", 0.0, 0.0),
        (ResourceType.NPU, {20: 1.456, 40: 1.046, 80: 0.791, 120: 0.832}, "npu_s2", 110.0, 6.0),
        (ResourceType.NPU, {20: 1.456, 40: 1.115, 80: 0.932, 120: 0.924}, "npu_s3", 115.0, 6.2),
        (ResourceType.NPU, {20: 8.780, 40: 6.761, 80: 5.712, 120: 5.699}, "npu_s4", 380.0, 20.0),
    ]
    
    for resource_type, duration_table, segment_id, power, ddr in segment_configs:
        seg = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id,
            power=power,
            ddr=ddr
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
            segment_id="main",
            power=160.0,  # 160mW per frame
            ddr=8.5  # 8.5MB per frame
        ),
        ResourceSegment(
            resource_type=ResourceType.DSP,
            duration_table={20: 1.23, 40: 0.71, 80: 0.45, 120: 0.41},
            start_time=0,
            segment_id="postprocess",
            power=0.0,  # DSP power set to 0
            ddr=0.0  # DSP DDR set to 0
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
        segment_id="main",
        power=120.0,  # 120mW per frame
        ddr=7.0  # 7MB per frame
    )]


def create_skywater_big_model() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """Skywater大模型: 可分段NPU+DSP with cut points"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={20: 4.19, 40: 2.49, 80: 1.70, 120: 1.67},
            start_time=0,
            segment_id="main",
            power=240.0,  # 240mW per frame
            ddr=13.0  # 13MB per frame
        ),
        ResourceSegment(
            resource_type=ResourceType.DSP,
            duration_table={20: 1.52, 40: 0.90, 80: 0.58, 120: 0.58},
            start_time=0,
            segment_id="postprocess",
            power=0.0,  # DSP power set to 0
            ddr=0.0  # DSP DDR set to 0
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
        segment_id="main",
        power=300.0,  # 300mW per frame
        ddr=15.0  # 15MB per frame
    )]

# Op 4k60 nets
def create_ml10t() -> List[ResourceSegment]:  # from vision
    """ML 10T Base Framne: 纯NPU (base 52.86ms)"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={40: 58.78, 160: 52.86}, # 40 multiple is from 1.112
        start_time=0,
        segment_id="main",
        power=800.0,  # 800mW per frame
        ddr=40.0  # 40MB per frame
    )]

def create_ml10t_075() -> List[ResourceSegment]:  # first 75% discount
    """ML 10T Big Framne: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={40: 44.085, 160: 39.645},
        start_time=0,
        segment_id="main",
        power=600.0,  # 600mW per frame
        ddr=30.0  # 30MB per frame
    )]
    
def create_ml10t_bigmid() -> List[ResourceSegment]:
    """ML 10T BigMid Framne: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={40: 33.06, 160: 29.73},
        start_time=0,
        segment_id="main",
        power=450.0,  # 450mW per frame
        ddr=25.0  # 25MB per frame
    )]
    
def create_ml10t_midsmall() -> List[ResourceSegment]:
    """ML 10T MidSmall Framne: 纯NPU"""
    return [ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={40: 16.535, 160: 14.87},
        start_time=0,
        segment_id="main",
        power=300.0,  # 300mW per frame
        ddr=18.0  # 18MB per frame
    )]
    
def create_aimetliteplus() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """aimetliteplus: 纯NPU + 可分段"""
    segments = [
        ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={40: 19.61, 160: 12.747},
        start_time=0,
        segment_id="main",
        power=392.01,
        ddr=19.792
    )]
    cut_points = {
        "main": [
            CutPoint(op_id="sub1", perf_lut={40: 1.346, 160: 1.013}, overhead_ms=0.0),
            CutPoint(op_id="sub2", perf_lut={40: 4.98, 160: 4.295}, overhead_ms=0.0),
            CutPoint(op_id="sub3", perf_lut={40: 3.803, 160: 3.191}, overhead_ms=0.0),
            CutPoint(op_id="sub4", perf_lut={40: 1.398, 160: 0.915}, overhead_ms=0.0),
            CutPoint(op_id="2023", perf_lut={40: 1.8, 160: 0.966}, overhead_ms=0.0),
            CutPoint(op_id="matmul", perf_lut={40: 2.804, 160: 1.402}, overhead_ms=0.0), # 40G is predicted
        ]
    }
    return segments, cut_points

def create_FaceEhnsLite() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """FaceEhnsLites: 纯NPU + 可分段"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={40: 19.62, 160: 8.19},
            start_time=0,
            segment_id="main",
            power=263.88,
            ddr=70.15
        )
    ]
    cut_points = {
        "main": [
            CutPoint(op_id="op4", perf_lut={40: 2.765, 160: 0.866}, overhead_ms=0.0),
            CutPoint(op_id="op33", perf_lut={40: 8.828, 160: 3.593}, overhead_ms=0.0),
            CutPoint(op_id="op42", perf_lut={40: 2.499, 160: 1.136}, overhead_ms=0.0),
            # CutPoint(op_id="op47", perf_lut={40: 4.094, 160: 2.593}, overhead_ms=0.0),
        ]
    }
    return segments, cut_points

def create_vmask() -> Tuple[List[ResourceSegment], Dict[str, List[CutPoint]]]:
    """VMASK: 纯NPU + 可分段"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={40: 11.0, 160: 3.746}, # 40Gbps perf is guessed
            start_time=0,
            segment_id="main",
            power=58.93,
            ddr=25.061
        )
    ]
    cut_points = {
        "main": [
            CutPoint(op_id="op104", perf_lut={40: 4.768, 160: 1.69}, overhead_ms=0.0),
        ]
    }
    return segments, cut_points

def create_FD() -> List[ResourceSegment]:
    """FD: 纯NPU + 可分段"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={40: 2.909, 160: 1.577},
            start_time=0,
            segment_id="main",
            power=22.4,
            ddr=11.473
        )
    ]
    return segments

def create_PD_depth() -> List[ResourceSegment]:
    """PD_depth: NPU + DSP复杂混合网络"""
    segments = []
    segment_configs = [
        (ResourceType.NPU, {40: 0.41, 160: 0.15}, "npu_s0", 59.89, 29.531),
        (ResourceType.DSP, {80: 0.46}, "dsp_s0", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.60, 160: 0.33}, "npu_s1", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s1", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.42, 160: 0.16}, "npu_s2", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s2", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.54, 160: 0.23}, "npu_s3", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s3", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.53, 160: 0.23}, "npu_s4", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s4", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.71, 160: 0.43}, "npu_s5", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s5", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.57, 160: 0.38}, "npu_s6", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s6", 0.0, 0.0),
        (ResourceType.NPU, {40: 2.42, 160: 1.52}, "npu_s7", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s7", 0.0, 0.0),
        (ResourceType.NPU, {40: 1.92, 160: 1.23}, "npu_s8", 0.0, 0.0),
        (ResourceType.DSP, {80: 0.46}, "dsp_s8", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.58, 160: 0.31}, "npu_s9", 0.0, 0.0),
    ]
    
    for resource_type, duration_table, segment_id, power, ddr in segment_configs:
        seg = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id,
            power=power,
            ddr=ddr
        )
        segments.append(seg)
    return segments

def create_cam_parsing() -> List[ResourceSegment]:
    """Parsing: 纯NPU"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={40: 0.595, 160: 0.421},
            start_time=0,
            segment_id="main",
            power=4.0,  # 60mW per frame
            ddr=2.961  # 3.5MB per frame
        )
    ]
    return segments

def create_AFTK() -> List[ResourceSegment]:
    """AF TK: NPU + DSP复杂混合"""
    segments = []
    
    # 定义所有段
    # DSP all 6.802, averaged as for now
    segment_configs = [
        (ResourceType.DSP, {80: 1.700}, "dsp_s0", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.232, 160: 0.140}, "npu_s0", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.392, 160: 0.203}, "npu_s1", 0.0, 0.0),
        (ResourceType.DSP, {80: 1.700}, "dsp_s1", 0.0, 0.0),
        (ResourceType.NPU, {40:6.438, 160: 2.988}, "npu_s2", 180.0, 10.0),
        (ResourceType.DSP, {80: 1.700}, "dsp_s2", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.392, 160: 0.203}, "npu_s3", 53.42, 29.352),
        (ResourceType.DSP, {80: 1.700}, "dsp_s3", 0.0, 0.0),
        (ResourceType.NPU, {40: 0.176, 160: 0.068}, "npu_s4", 0.0, 0.0),
    ]
    
    for resource_type, duration_table, segment_id, power, ddr in segment_configs:
        seg = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id,
            power=power,
            ddr=ddr
        )
        segments.append(seg)
    
    return segments

def create_PD_dns() -> List[ResourceSegment]:
    """PD DNS: 纯NPU"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={160: 0.593}, # no 40Gbps perf
            start_time=0,
            segment_id="main",
            power=8.21,
            ddr=0.772
        )
    ]
    return segments

def create_NN_tone() -> List[ResourceSegment]:
    """NN Tone: 纯NPU"""
    segments = [
        ResourceSegment(
            resource_type=ResourceType.NPU,
            duration_table={40: 3.496, 160: 2.15}, # no 40Gbps perf
            start_time=0,
            segment_id="main",
            power=41.46,
            ddr=9.546
        )
    ]
    return segments

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
    # Camera 9+2 nets
    "ML10T_bigmid": create_ml10t_bigmid,
    "ML10T_midsmall": create_ml10t_midsmall,
    "AimetlitePlus": create_aimetliteplus,
    "FaceEhnsLite": create_FaceEhnsLite,
    "Vmask": create_vmask,
    "FaceDet": create_FD,
    "Cam_Parsing": create_cam_parsing,
    "NNTone": create_NN_tone,
    "PD_DNS": create_PD_dns,
    "PD_Depth": create_PD_depth,
    "AF_TK": create_AFTK,
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