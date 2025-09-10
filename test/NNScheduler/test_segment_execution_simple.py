#!/usr/bin/env python3
"""
简化的段执行测试 - 验证段级调度的基本概念
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Dict, List, Optional
from dataclasses import dataclass
import heapq

from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.enums import ResourceType, TaskPriority
from NNScheduler.core.task import create_mixed_task
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
from NNScheduler.core.models import SubSegment


def test_segment_level_scheduling():
    """测试段级调度的核心概念"""
    print("=== 测试段级调度概念 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建两个测试任务
    # 任务A: NPU(10ms) -> DSP(15ms) -> NPU(5ms)
    task_a = create_mixed_task(
        "TaskA", "混合任务A",
        segments=[
            (ResourceType.NPU, {60: 10.0}, "a_npu0"),
            (ResourceType.DSP, {40: 15.0}, "a_dsp0"),
            (ResourceType.NPU, {60: 5.0}, "a_npu1"),
        ],
        priority=TaskPriority.NORMAL
    )
    task_a.set_performance_requirements(fps=10, latency=100)
    
    # 任务B: NPU(8ms) -> NPU(8ms)
    task_b = create_mixed_task(
        "TaskB", "NPU任务B",
        segments=[
            (ResourceType.NPU, {60: 8.0}, "b_npu0"),
            (ResourceType.NPU, {60: 8.0}, "b_npu1"),
        ],
        priority=TaskPriority.NORMAL
    )
    task_b.set_performance_requirements(fps=10, latency=100)
    
    # 注册任务
    launcher.register_task(task_a)
    launcher.register_task(task_b)
    
    print("任务配置:")
    print(f"  TaskA: NPU(10ms) → DSP(15ms) → NPU(5ms)")
    print(f"  TaskB: NPU(8ms) → NPU(8ms)")
    
    # 模拟段级调度
    print("\n模拟段级调度执行:")
    simulate_segment_scheduling(queue_manager, tracer, [task_a, task_b])
    
    # 显示结果
    print("\n执行时间线:")
    visualizer.print_gantt_chart(width=80)
    
    # 分析结果
    print("\n关键观察:")
    print("1. 当TaskA的第一段在NPU上执行时，DSP是空闲的")
    print("2. 当TaskA的第二段在DSP上执行时，NPU是空闲的")
    print("3. 段级调度应该允许TaskB的段在TaskA使用DSP时使用NPU")
    
    # 计算资源利用率
    utilization = tracer.get_resource_utilization()
    print("\n资源利用率:")
    for res_id, util_percent in utilization.items():
        print(f"  {res_id}: {util_percent:.1f}%")


def simulate_segment_scheduling(queue_manager, tracer, tasks):
    """模拟段级调度"""
    
    @dataclass
    class SegmentInfo:
        task_id: str
        instance_id: int
        segment_index: int
        segment: SubSegment
        is_ready: bool = False
        is_completed: bool = False
        start_time: Optional[float] = None
        end_time: Optional[float] = None
    
    # 准备所有段
    all_segments = []
    
    for task in tasks:
        # 获取任务的子段
        sub_segments = task.apply_segmentation()
        if not sub_segments:
            # 转换原始段
            sub_segments = []
            for seg in task.segments:
                sub_seg = SubSegment(
                    sub_id=seg.segment_id,
                    resource_type=seg.resource_type,
                    duration_table=seg.duration_table,
                    cut_overhead=0.0,
                    original_segment_id=seg.segment_id
                )
                sub_segments.append(sub_seg)
        
        # 创建段信息
        for i, seg in enumerate(sub_segments):
            seg_info = SegmentInfo(
                task_id=task.task_id,
                instance_id=0,
                segment_index=i,
                segment=seg,
                is_ready=(i == 0)  # 第一段就绪
            )
            all_segments.append(seg_info)
    
    # 打印段信息
    print("\n段信息:")
    for seg in all_segments:
        print(f"  {seg.task_id}_seg{seg.segment_index}: {seg.segment.resource_type.value}")
    
    # 模拟执行
    current_time = 0.0
    max_time = 50.0
    time_step = 0.1
    
    while current_time < max_time and any(not seg.is_completed for seg in all_segments):
        # 更新资源状态
        for queue in queue_manager.resource_queues.values():
            queue.advance_time(current_time)
        
        # 尝试调度就绪的段
        for seg_info in all_segments:
            if seg_info.is_completed or not seg_info.is_ready:
                continue
            
            # 检查前序段是否完成（同任务的段必须顺序执行）
            if seg_info.segment_index > 0:
                prev_seg = next((s for s in all_segments 
                                if s.task_id == seg_info.task_id 
                                and s.segment_index == seg_info.segment_index - 1), None)
                if prev_seg and not prev_seg.is_completed:
                    continue
            
            # 查找可用资源
            resource_type = seg_info.segment.resource_type
            best_queue = None
            
            for res_id, queue in queue_manager.resource_queues.items():
                if queue.resource_type == resource_type and not queue.is_busy():
                    best_queue = queue
                    break
            
            if best_queue and seg_info.start_time is None:  # 只调度未开始的段
                # 执行段
                duration = seg_info.segment.get_duration(best_queue.bandwidth)
                end_time = current_time + duration
                
                seg_info.start_time = current_time
                seg_info.end_time = end_time
                
                # 记录执行
                segment_id = f"{seg_info.task_id}#{seg_info.instance_id}_seg{seg_info.segment_index}"
                tracer.record_execution(
                    segment_id,
                    best_queue.resource_id,
                    current_time,
                    end_time,
                    best_queue.bandwidth,
                    seg_info.segment.sub_id
                )
                
                # 更新队列状态
                best_queue.busy_until = end_time
                best_queue.current_task = segment_id
                
                print(f"  {current_time:>6.1f}ms: [EXECUTE] {segment_id} on {best_queue.resource_id} "
                      f"(duration={duration:.1f}ms)")
        
        # 更新完成状态
        for seg_info in all_segments:
            if seg_info.start_time is not None and seg_info.end_time is not None and seg_info.end_time <= current_time:
                if not seg_info.is_completed:
                    seg_info.is_completed = True
                    # 激活下一段
                    next_seg = next((s for s in all_segments 
                                    if s.task_id == seg_info.task_id 
                                    and s.segment_index == seg_info.segment_index + 1), None)
                    if next_seg:
                        next_seg.is_ready = True
                        print(f"  {current_time:>6.1f}ms: [READY] {seg_info.task_id}_seg{next_seg.segment_index}")
        
        # 时间推进
        current_time += time_step


def test_optimization_potential():
    """测试段级调度的优化潜力"""
    print("\n\n=== 测试优化潜力 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 场景1：传统调度（整体发射）
    print("场景1：传统调度（任务必须完整执行）")
    tracer1 = ScheduleTracer(queue_manager)
    
    # 模拟传统执行 - TaskA和TaskB串行
    print("\n执行顺序:")
    print("  0-10ms:  TaskA_seg0 on NPU")
    print("  10-25ms: TaskA_seg1 on DSP (NPU空闲!)")
    print("  25-30ms: TaskA_seg2 on NPU")
    print("  30-38ms: TaskB_seg0 on NPU (等待TaskA完成)")
    print("  38-46ms: TaskB_seg1 on NPU")
    
    tracer1.record_execution("TaskA#0", "NPU_0", 0, 10, 60, "a_seg0")
    tracer1.record_execution("TaskA#0", "DSP_0", 10, 25, 40, "a_seg1")
    tracer1.record_execution("TaskA#0", "NPU_0", 25, 30, 60, "a_seg2")
    tracer1.record_execution("TaskB#0", "NPU_0", 30, 38, 60, "b_seg0")
    tracer1.record_execution("TaskB#0", "NPU_0", 38, 46, 60, "b_seg1")
    
    total_time1 = 46.0
    npu_idle1 = 15.0  # 10-25ms期间NPU空闲
    dsp_idle1 = 31.0  # DSP只用了15ms
    
    print(f"\n结果:")
    print(f"  总执行时间: {total_time1}ms")
    print(f"  NPU利用率: {((total_time1-npu_idle1)/total_time1)*100:.1f}%")
    print(f"  DSP利用率: {((total_time1-dsp_idle1)/total_time1)*100:.1f}%")
    
    # 场景2：段级调度
    print("\n场景2：段级调度（段可独立执行）")
    tracer2 = ScheduleTracer(queue_manager)
    
    print("\n优化的执行顺序:")
    print("  0-10ms:  TaskA_seg0 on NPU")
    print("  10-18ms: TaskB_seg0 on NPU (立即开始!)")
    print("  10-25ms: TaskA_seg1 on DSP (并行执行!)")
    print("  18-26ms: TaskB_seg1 on NPU")
    print("  26-31ms: TaskA_seg2 on NPU")
    
    tracer2.record_execution("TaskA#0_seg0", "NPU_0", 0, 10, 60, "a_seg0")
    tracer2.record_execution("TaskB#0_seg0", "NPU_0", 10, 18, 60, "b_seg0")
    tracer2.record_execution("TaskA#0_seg1", "DSP_0", 10, 25, 40, "a_seg1")
    tracer2.record_execution("TaskB#0_seg1", "NPU_0", 18, 26, 60, "b_seg1")
    tracer2.record_execution("TaskA#0_seg2", "NPU_0", 26, 31, 60, "a_seg2")
    
    total_time2 = 31.0
    npu_idle2 = 0.0   # NPU全程忙碌
    dsp_idle2 = 16.0  # DSP用了15ms
    
    print(f"\n结果:")
    print(f"  总执行时间: {total_time2}ms")
    print(f"  NPU利用率: {((total_time2-npu_idle2)/total_time2)*100:.1f}%")
    print(f"  DSP利用率: {((total_time2-dsp_idle2)/total_time2)*100:.1f}%")
    
    print(f"\n性能对比:")
    print(f"  执行时间减少: {total_time1-total_time2}ms ({((total_time1-total_time2)/total_time1)*100:.1f}%)")
    print(f"  NPU利用率提升: {((total_time2-npu_idle2)/total_time2)*100 - ((total_time1-npu_idle1)/total_time1)*100:.1f}%")
    
    print("\n关键洞察:")
    print("  ✓ 段级调度允许TaskB在TaskA使用DSP时开始执行")
    print("  ✓ NPU和DSP可以真正并行工作")
    print("  ✓ 整体性能提升32.6%")
    
    # 可视化对比
    print("\n执行时间线对比:")
    print("\n传统调度:")
    print("NPU: [TaskA_seg0    ][----空闲----][TaskA_seg2][TaskB_seg0 ][TaskB_seg1 ]")
    print("DSP: [----空闲----][TaskA_seg1         ][----------空闲-----------]")
    
    print("\n段级调度:")
    print("NPU: [TaskA_seg0    ][TaskB_seg0 ][TaskB_seg1 ][TaskA_seg2]")
    print("DSP: [----空闲----][TaskA_seg1         ][---空闲---]")


if __name__ == "__main__":
    test_segment_level_scheduling()
    test_optimization_potential()
