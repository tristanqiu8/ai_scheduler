#!/usr/bin/env python3
"""
测试精简后的 NNTask 类
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.task import NNTask, create_npu_task, create_dsp_task, create_mixed_task
from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy


def test_basic_task_creation():
    """测试基本任务创建"""
    print("=== 测试基本任务创建 ===\n")
    
    # 1. 创建简单NPU任务
    task1 = NNTask("T1", "SimpleNPU")
    task1.set_npu_only({40: 5.0, 120: 2.5})
    
    print(f"任务1: {task1}")
    print(f"  段数: {len(task1.segments)}")
    print(f"  资源需求: {task1.get_resource_requirements()}")
    print(f"  40带宽下执行时间: {task1.estimate_duration({ResourceType.NPU: 40})}ms")
    print(f"  120带宽下执行时间: {task1.estimate_duration({ResourceType.NPU: 120})}ms")
    
    # 2. 创建DSP任务
    task2 = create_dsp_task("T2", "SimpleDSP", {40: 3.0, 120: 2.0})
    print(f"\n任务2: {task2}")
    print(f"  使用DSP: {task2.uses_dsp}")
    print(f"  使用NPU: {task2.uses_npu}")
    
    # 3. 设置性能需求
    task1.set_performance_requirements(fps=30, latency=50)
    print(f"\n任务1性能需求:")
    print(f"  FPS: {task1.fps_requirement}")
    print(f"  延迟: {task1.latency_requirement}ms")
    print(f"  最小间隔: {task1.min_interval_ms:.1f}ms")


def test_mixed_resource_task():
    """测试混合资源任务"""
    print("\n=== 测试混合资源任务 ===\n")
    
    # 创建DSP+NPU混合任务
    segments = [
        (ResourceType.NPU, {40: 0.410, 120: 0.249}, "npu_s0"),
        (ResourceType.DSP, {40: 1.2}, "dsp_s0"),
        (ResourceType.NPU, {40: 0.626, 120: 0.379}, "npu_s1"),
        (ResourceType.DSP, {40: 2.2}, "dsp_s1"),
    ]
    
    task = create_mixed_task(
        "T3", "MOTR",
        segments,
        priority=TaskPriority.CRITICAL,
        runtime_type=RuntimeType.ACPU_RUNTIME
    )
    
    print(f"混合任务: {task}")
    print(f"  段数: {len(task.segments)}")
    print(f"  是否混合资源: {task.is_mixed_resource}")
    print(f"  资源需求: {task.get_resource_requirements()}")
    
    print("\n各段详情:")
    for i, seg in enumerate(task.segments):
        print(f"  段{i} ({seg.segment_id}): {seg.resource_type.value}")
        print(f"    时长表: {seg.duration_table}")
    
    # 估算执行时间
    bandwidth_map = {ResourceType.NPU: 40, ResourceType.DSP: 40}
    total_duration = task.estimate_duration(bandwidth_map)
    print(f"\n在40带宽下的总执行时间: {total_duration:.3f}ms")


def test_task_segmentation():
    """测试任务分段功能"""
    print("\n=== 测试任务分段功能 ===\n")
    
    # 创建可分段的NPU任务
    task = NNTask(
        "T4", "YoloV8n",
        priority=TaskPriority.NORMAL,
        segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION
    )
    
    # 添加主段
    segment = task.add_segment(
        ResourceType.NPU,
        {40: 12.71, 120: 6.35},
        "npu_main"
    )
    
    # 添加切分点
    print("添加4个切分点...")
    task.add_cut_points_to_segment("npu_main", [
        ("op6", {40: 2.54, 120: 1.27}, 0.1),   # 20%处切分
        ("op13", {40: 5.08, 120: 2.54}, 0.1),  # 40%处切分
        ("op14", {40: 7.63, 120: 3.81}, 0.1),  # 60%处切分
        ("op19", {40: 10.17, 120: 5.08}, 0.1), # 80%处切分
    ])
    
    # 设置预定义的切分配置
    task.set_preset_cut_configurations("npu_main", [
        [],                                  # 配置0: 不切分
        ["op6", "op19"],                    # 配置1: 切成3段
        ["op6", "op13", "op19"],           # 配置2: 切成4段
        ["op6", "op13", "op14", "op19"],   # 配置3: 切成5段
    ])
    
    # 测试不同的切分配置
    print("\n测试不同切分配置:")
    for config_idx in range(4):
        task.select_cut_configuration("npu_main", config_idx)
        sub_segments = task.apply_segmentation()
        
        print(f"\n配置{config_idx}: 生成{len(sub_segments)}个子段")
        total_duration_40 = 0
        total_overhead = 0
        
        for sub_seg in sub_segments:
            duration = sub_seg.get_duration(40)
            print(f"  {sub_seg.sub_id}: {duration:.2f}ms (含开销{sub_seg.cut_overhead}ms)")
            total_duration_40 += duration
            total_overhead += sub_seg.cut_overhead
        
        print(f"  总时间: {total_duration_40:.2f}ms (原始: 12.71ms)")
        print(f"  总开销: {total_overhead:.2f}ms")


def test_task_dependencies():
    """测试任务依赖"""
    print("\n=== 测试任务依赖 ===\n")
    
    # 创建任务链
    task1 = create_npu_task("T1", "Detector", {40: 10.0})
    task2 = create_npu_task("T2", "Tracker", {40: 5.0})
    task3 = create_dsp_task("T3", "PostProcess", {40: 2.0})
    
    # 设置依赖关系
    task2.add_dependency("T1")  # T2依赖T1
    task3.add_dependencies(["T1", "T2"])  # T3依赖T1和T2
    
    tasks = [task1, task2, task3]
    
    print("任务依赖关系:")
    for task in tasks:
        if task.dependencies:
            print(f"  {task.task_id} 依赖: {task.dependencies}")
        else:
            print(f"  {task.task_id} 无依赖")


def test_real_world_scenario():
    """测试真实场景的任务定义"""
    print("\n=== 真实场景任务定义 ===\n")
    
    tasks = []
    
    # 1. MOTR - 关键的多目标跟踪任务
    motr = create_mixed_task(
        "T1", "MOTR",
        segments=[
            (ResourceType.NPU, {40: 0.410, 120: 0.249}, "npu_s0"),
            (ResourceType.DSP, {40: 1.2}, "dsp_s0"),
            (ResourceType.NPU, {40: 0.626, 120: 0.379}, "npu_s1"),
            (ResourceType.NPU, {40: 9.333, 120: 5.147}, "npu_s2"),
            (ResourceType.DSP, {40: 2.2}, "dsp_s1"),
            (ResourceType.NPU, {40: 0.626, 120: 0.379}, "npu_s3"),
            (ResourceType.DSP, {40: 1.5}, "dsp_s2"),
            (ResourceType.NPU, {40: 0.285, 120: 0.153}, "npu_s4"),
            (ResourceType.DSP, {40: 2.0}, "dsp_s3"),
        ],
        priority=TaskPriority.CRITICAL,
        runtime_type=RuntimeType.ACPU_RUNTIME
    )
    motr.set_performance_requirements(fps=25, latency=40)
    tasks.append(motr)
    
    # 2. YOLO检测器
    yolo = create_npu_task(
        "T2", "YoloV8n",
        {40: 12.71, 120: 6.35},
        priority=TaskPriority.NORMAL,
        segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION
    )
    yolo.set_performance_requirements(fps=10, latency=100)
    tasks.append(yolo)
    
    # 3. 姿态估计（依赖MOTR）
    pose = create_npu_task(
        "T3", "Pose2D",
        {40: 3.096, 120: 2.232},
        priority=TaskPriority.NORMAL
    )
    pose.set_performance_requirements(fps=25, latency=40)
    pose.add_dependency("T1")  # 依赖MOTR
    tasks.append(pose)
    
    # 4. 重识别（高频任务）
    reid = create_npu_task(
        "T4", "ReID",
        {40: 0.778, 120: 0.631},
        priority=TaskPriority.HIGH
    )
    reid.set_performance_requirements(fps=100, latency=10)
    tasks.append(reid)
    
    # 打印任务摘要
    print("创建的任务:")
    print(f"{'ID':<4} {'名称':<10} {'优先级':<10} {'FPS':<6} {'资源':<15} {'段数':<6}")
    print("-" * 60)
    
    for task in tasks:
        resources = ", ".join(r.value for r in task.get_resource_requirements())
        print(f"{task.task_id:<4} {task.name:<10} {task.priority.name:<10} "
              f"{task.fps_requirement:<6.0f} {resources:<15} {len(task.segments):<6}")
    
    # 估算在不同带宽下的执行时间
    print("\n不同带宽下的执行时间估算:")
    test_bandwidths = [40, 80, 120]
    
    for bw in test_bandwidths:
        print(f"\n带宽 = {bw}:")
        bandwidth_map = {ResourceType.NPU: bw, ResourceType.DSP: bw}
        
        for task in tasks:
            duration = task.estimate_duration(bandwidth_map)
            print(f"  {task.name}: {duration:.2f}ms")


def main():
    """运行所有测试"""
    print("开始测试精简后的 NNTask 类\n")
    
    test_basic_task_creation()
    print("\n" + "="*60)
    
    test_mixed_resource_task()
    print("\n" + "="*60)
    
    test_task_segmentation()
    print("\n" + "="*60)
    
    test_task_dependencies()
    print("\n" + "="*60)
    
    test_real_world_scenario()
    
    print("\n\n所有测试完成！")


if __name__ == "__main__":
    main()
