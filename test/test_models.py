#!/usr/bin/env python3
"""
测试 models.py 中的数据结构
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.models import (
    CutPoint, SubSegment, ResourceSegment, ResourceUnit, 
    TaskScheduleInfo, SegmentationDecision
)
from core.enums import ResourceType, RuntimeType, CutPointStatus


def test_resource_segment_basic():
    """测试基本的ResourceSegment功能"""
    print("=== 测试 ResourceSegment 基本功能 ===")
    
    # 创建一个NPU段
    segment = ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 10.0, 40: 6.0, 120: 3.0},
        start_time=0,
        segment_id="npu_main"
    )
    
    print(f"创建的段: {segment.segment_id}")
    print(f"资源类型: {segment.resource_type.value}")
    print(f"耗时表: {segment.duration_table}")
    
    # 测试不同带宽下的耗时
    test_bandwidths = [20, 40, 80, 120]
    print("\n不同带宽下的执行时间:")
    for bw in test_bandwidths:
        duration = segment.get_duration(bw)
        print(f"  BW={bw}: {duration:.1f}ms")
    
    print()


def test_segmentation_with_cuts():
    """测试分段功能"""
    print("=== 测试分段功能 ===")
    
    # 创建一个较大的NPU段
    segment = ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 20.0, 40: 12.0, 120: 6.0},
        start_time=0,
        segment_id="npu_yolo"
    )
    
    print(f"原始段总耗时: {segment.duration_table}")
    
    # 添加两个断点，将任务分成3段
    # 第一段：前30%的工作量
    segment.add_cut_point(
        "op5",
        perf_lut={20: 6.0, 40: 3.6, 120: 1.8},
        overhead_ms=0.2
    )
    
    # 第二段：中间40%的工作量
    segment.add_cut_point(
        "op10", 
        perf_lut={20: 8.0, 40: 4.8, 120: 2.4},
        overhead_ms=0.2
    )
    
    print(f"\n添加了 {len(segment.cut_points)} 个断点")
    
    # 测试不同的分段策略
    test_cases = [
        ([], "不分段"),
        (["op5"], "只在op5处分段"),
        (["op10"], "只在op10处分段"),
        (["op5", "op10"], "在op5和op10处都分段")
    ]
    
    for cuts, description in test_cases:
        print(f"\n{description}:")
        sub_segments = segment.apply_segmentation(cuts)
        
        print(f"  生成了 {len(sub_segments)} 个子段")
        
        for i, sub_seg in enumerate(sub_segments):
            print(f"  子段 {sub_seg.sub_id}:")
            print(f"    耗时表: {sub_seg.duration_table}")
            print(f"    切割开销: {sub_seg.cut_overhead}ms")
            
            # 验证在40带宽下的执行时间
            duration_40 = sub_seg.get_duration(40)
            print(f"    在BW=40时执行时间: {duration_40:.1f}ms")
        
        # 计算总开销
        total_overhead = segment.segmentation_overhead
        print(f"  总切割开销: {total_overhead:.1f}ms")
        
        # 验证总时间
        total_time_40 = sum(sub.get_duration(40) for sub in sub_segments)
        original_time_40 = segment.get_duration(40)
        print(f"  验证(BW=40): 原始时间={original_time_40}ms, "
              f"分段后总时间={total_time_40:.1f}ms "
              f"(差异={total_time_40-original_time_40:.1f}ms)")


def test_complex_task_scenario():
    """测试复杂任务场景"""
    print("\n=== 测试复杂任务场景 ===")
    
    # 创建一个DSP+NPU混合任务
    segments = []
    
    # DSP预处理段
    dsp_seg = ResourceSegment(
        resource_type=ResourceType.DSP,
        duration_table={40: 2.0, 120: 1.5},
        start_time=0,
        segment_id="dsp_preprocess"
    )
    segments.append(dsp_seg)
    
    # NPU主处理段（可分段）
    npu_seg = ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={20: 15.0, 40: 9.0, 120: 4.5},
        start_time=2.0,  # DSP段之后
        segment_id="npu_main"
    )
    
    # 为NPU段添加3个断点，可以分成最多4段
    npu_seg.add_cut_point(
        "conv1_out",
        perf_lut={20: 3.0, 40: 1.8, 120: 0.9},
        overhead_ms=0.15
    )
    
    npu_seg.add_cut_point(
        "conv2_out",
        perf_lut={20: 4.0, 40: 2.4, 120: 1.2},
        overhead_ms=0.15
    )
    
    npu_seg.add_cut_point(
        "conv3_out",
        perf_lut={20: 5.0, 40: 3.0, 120: 1.5},
        overhead_ms=0.15
    )
    
    segments.append(npu_seg)
    
    # DSP后处理段
    dsp_post_seg = ResourceSegment(
        resource_type=ResourceType.DSP,
        duration_table={40: 1.0, 120: 0.8},
        start_time=11.0,  # NPU段之后（假设BW=40）
        segment_id="dsp_postprocess"
    )
    segments.append(dsp_post_seg)
    
    print("创建的混合任务包含3个段:")
    for seg in segments:
        print(f"  - {seg.segment_id}: {seg.resource_type.value}, "
              f"开始时间={seg.start_time}ms")
    
    # 测试NPU段的分段
    print("\n对NPU段应用分段 (使用所有断点):")
    npu_sub_segments = npu_seg.apply_segmentation(["conv1_out", "conv2_out", "conv3_out"])
    
    print(f"NPU段被分成了 {len(npu_sub_segments)} 个子段:")
    for sub in npu_sub_segments:
        print(f"  - {sub.sub_id}: BW=40时耗时 {sub.get_duration(40):.1f}ms")


def test_schedule_info():
    """测试调度信息结构"""
    print("\n=== 测试 TaskScheduleInfo ===")
    
    # 创建一个调度信息
    schedule = TaskScheduleInfo(
        task_id="T1",
        start_time=10.0,
        end_time=25.0,
        assigned_resources={
            ResourceType.NPU: "NPU_0",
            ResourceType.DSP: "DSP_0"
        },
        actual_latency=15.0,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        used_cuts={"npu_main": ["op5", "op10"]},
        segmentation_overhead=0.4,
        sub_segment_schedule=[
            ("npu_main_0", 10.0, 13.0),
            ("npu_main_1", 13.2, 17.5),
            ("npu_main_2", 17.7, 20.0),
            ("dsp_post_0", 20.0, 21.0)
        ]
    )
    
    print(f"任务 {schedule.task_id} 调度信息:")
    print(f"  执行时间: {schedule.start_time:.1f}ms - {schedule.end_time:.1f}ms")
    print(f"  实际延迟: {schedule.actual_latency:.1f}ms")
    print(f"  运行时类型: {schedule.runtime_type.value}")
    print(f"  分配的资源: {schedule.assigned_resources}")
    print(f"  使用的断点: {schedule.used_cuts}")
    print(f"  分段开销: {schedule.segmentation_overhead:.1f}ms")
    print(f"  子段调度:")
    for sub_id, start, end in schedule.sub_segment_schedule:
        print(f"    - {sub_id}: {start:.1f}ms - {end:.1f}ms")


def test_resource_unit():
    """测试资源单元"""
    print("\n=== 测试 ResourceUnit ===")
    
    # 创建NPU和DSP资源
    npu0 = ResourceUnit("NPU_0", ResourceType.NPU, 120.0)
    npu1 = ResourceUnit("NPU_1", ResourceType.NPU, 40.0)
    dsp0 = ResourceUnit("DSP_0", ResourceType.DSP, 40.0)
    
    resources = [npu0, npu1, dsp0]
    
    print("创建的资源:")
    for res in resources:
        print(f"  - {res.unit_id}: 类型={res.resource_type.value}, "
              f"带宽={res.bandwidth}")
    
    # 测试哈希功能
    print(f"\n测试哈希: hash(npu0)={hash(npu0)}, hash(npu1)={hash(npu1)}")
    
    # 在集合中使用
    resource_set = set(resources)
    print(f"资源集合大小: {len(resource_set)}")


def main():
    """运行所有测试"""
    print("开始测试 models.py\n")
    
    test_resource_segment_basic()
    print("\n" + "="*50 + "\n")
    
    test_segmentation_with_cuts()
    print("\n" + "="*50 + "\n")
    
    test_complex_task_scenario()
    print("\n" + "="*50 + "\n")
    
    test_schedule_info()
    print("\n" + "="*50 + "\n")
    
    test_resource_unit()
    
    print("\n所有测试完成！")


if __name__ == "__main__":
    main()
