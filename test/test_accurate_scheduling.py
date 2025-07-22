#!/usr/bin/env python3
"""
准确的调度测试 - 展示发射和执行的完整流程
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enums import ResourceType, TaskPriority
from core.task import create_npu_task, create_dsp_task, create_mixed_task
from viz.schedule_visualizer import ScheduleVisualizer


def test_launch_vs_execution():
    """测试发射时序 vs 执行时序"""
    print("=== 测试发射 vs 执行时序 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建任务集：不同优先级和持续时间
    tasks = [
        # 低优先级但先发射
        create_npu_task("LOW_FIRST", "低优先级先发", {60: 10.0}, priority=TaskPriority.LOW),
        # 高优先级但后发射
        create_npu_task("HIGH_LATER", "高优先级后发", {60: 5.0}, priority=TaskPriority.HIGH),
        # 正常优先级
        create_npu_task("NORMAL_MID", "正常优先级", {60: 8.0}, priority=TaskPriority.NORMAL),
    ]
    
    # 设置FPS要求（都是单次执行）
    for task in tasks:
        task.set_performance_requirements(fps=1, latency=100)
        launcher.register_task(task)
    
    print("任务配置:")
    for task in tasks:
        print(f"  {task.task_id}: {task.priority.name}优先级, {task.estimate_duration({ResourceType.NPU: 60})}ms")
    
    # 手动控制发射顺序
    print("\n发射顺序（故意与优先级相反）:")
    
    # 1. 先发射低优先级任务
    launcher._launch_task("LOW_FIRST", 0, 0.0)
    print("  0.0ms: 发射 LOW_FIRST (LOW)")
    
    # 2. 再发射高优先级任务
    launcher._launch_task("HIGH_LATER", 0, 2.0)
    print("  2.0ms: 发射 HIGH_LATER (HIGH)")
    
    # 3. 最后发射正常优先级
    launcher._launch_task("NORMAL_MID", 0, 4.0)
    print("  4.0ms: 发射 NORMAL_MID (NORMAL)")
    
    # 模拟执行，展示优先级调度
    print("\n实际执行顺序（基于优先级）:")
    simulate_execution_with_details(queue_manager, tracer, 30.0)
    
    # 显示结果
    print("\n" + "="*80)
    visualizer.print_gantt_chart(width=80)
    
    # 生成可视化
    visualizer.plot_resource_timeline("launch_vs_execution.png")
    print("\n✓ 生成图表: launch_vs_execution.png")
    
    # 分析
    print("\n分析:")
    print("  - 虽然 LOW_FIRST 先发射，但 HIGH_LATER 会抢占执行")
    print("  - 实际执行顺序应该是: HIGH_LATER → NORMAL_MID → LOW_FIRST")


def test_segmented_task_execution():
    """测试分段任务的详细执行"""
    print("\n\n=== 测试分段任务执行 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建一个混合资源任务（类似MOTR）
    mixed_task = create_mixed_task(
        "MIXED_SEG", "混合分段任务",
        segments=[
            (ResourceType.NPU, {60: 4.0}, "npu_seg0"),
            (ResourceType.DSP, {40: 6.0}, "dsp_seg0"),
            (ResourceType.NPU, {60: 3.0}, "npu_seg1"),
            (ResourceType.DSP, {40: 5.0}, "dsp_seg1"),
        ],
        priority=TaskPriority.HIGH
    )
    mixed_task.set_performance_requirements(fps=25, latency=40)
    
    launcher.register_task(mixed_task)
    
    print("任务结构:")
    print(f"  {mixed_task.task_id}: {len(mixed_task.segments)}个段")
    for i, seg in enumerate(mixed_task.segments):
        duration = seg.get_duration(60 if seg.resource_type == ResourceType.NPU else 40)
        print(f"    段{i}: {seg.resource_type.value}, {duration}ms")
    
    # 发射任务
    print("\n发射任务:")
    launcher._launch_task("MIXED_SEG", 0, 0.0)
    
    # 详细执行模拟
    print("\n执行过程:")
    current_time = 0.0
    max_time = 30.0
    
    # 手动处理每个段的执行
    sub_segments = mixed_task.apply_segmentation()
    if not sub_segments:
        # 转换为子段格式
        from core.models import SubSegment
        sub_segments = []
        for seg in mixed_task.segments:
            sub_seg = SubSegment(
                sub_id=seg.segment_id,
                resource_type=seg.resource_type,
                duration_table=seg.duration_table,
                cut_overhead=0.0,
                original_segment_id=seg.segment_id
            )
            sub_segments.append(sub_seg)
    
    # 执行每个段
    for i, sub_seg in enumerate(sub_segments):
        # 选择合适的资源
        if sub_seg.resource_type == ResourceType.NPU:
            # 交替使用NPU资源
            resource_id = "NPU_0" if i % 2 == 0 else "NPU_1"
            bandwidth = 60.0
        else:
            resource_id = "DSP_0"
            bandwidth = 40.0
        
        # 计算执行时间
        duration = sub_seg.get_duration(bandwidth)
        end_time = current_time + duration
        
        # 记录执行
        tracer.record_execution(
            f"MIXED_SEG#0_seg{i}",
            resource_id,
            current_time,
            end_time,
            bandwidth,
            sub_seg.sub_id
        )
        
        print(f"  {current_time:>5.1f}-{end_time:>5.1f}ms: {resource_id} 执行 {sub_seg.sub_id}")
        
        # 更新时间（简化：假设段按顺序执行）
        current_time = end_time
    
    # 显示结果
    print("\n执行时间线:")
    visualizer.print_gantt_chart(width=80)
    
    # 生成可视化
    visualizer.plot_resource_timeline("segmented_execution.png")
    visualizer.export_chrome_tracing("segmented_execution.json")
    
    print("\n✓ 生成文件:")
    print("  - segmented_execution.png")
    print("  - segmented_execution.json")


def test_concurrent_execution():
    """测试并发执行场景"""
    print("\n\n=== 测试并发执行 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建多个可并发的任务
    tasks = []
    for i in range(4):
        task = create_npu_task(
            f"TASK_{i}", 
            f"任务{i}",
            {60: 8.0 + i * 2},  # 不同的执行时间
            priority=[TaskPriority.HIGH, TaskPriority.NORMAL, 
                     TaskPriority.NORMAL, TaskPriority.LOW][i]
        )
        task.set_performance_requirements(fps=10, latency=100)
        tasks.append(task)
        launcher.register_task(task)
    
    # 同时发射所有任务
    print("同时发射4个任务:")
    for i, task in enumerate(tasks):
        launcher._launch_task(task.task_id, 0, 0.0)
        print(f"  0.0ms: 发射 {task.task_id} ({task.priority.name})")
    
    # 模拟并发执行
    print("\n并发执行模拟:")
    current_time = 0.0
    
    # 手动调度到不同NPU
    # TASK_0 (HIGH) -> NPU_0
    tracer.record_execution("TASK_0#0", "NPU_0", 0.0, 8.0, 60.0)
    print(f"  0.0-8.0ms: NPU_0 执行 TASK_0 (HIGH)")
    
    # TASK_1 (NORMAL) -> NPU_1
    tracer.record_execution("TASK_1#0", "NPU_1", 0.0, 10.0, 60.0)
    print(f"  0.0-10.0ms: NPU_1 执行 TASK_1 (NORMAL)")
    
    # TASK_2 (NORMAL) 等待 NPU_0
    tracer.record_execution("TASK_2#0", "NPU_0", 8.0, 20.0, 60.0)
    print(f"  8.0-20.0ms: NPU_0 执行 TASK_2 (NORMAL)")
    
    # TASK_3 (LOW) 等待 NPU_1
    tracer.record_execution("TASK_3#0", "NPU_1", 10.0, 24.0, 60.0)
    print(f"  10.0-24.0ms: NPU_1 执行 TASK_3 (LOW)")
    
    # 显示结果
    print("\n并发执行时间线:")
    visualizer.print_gantt_chart(width=80)
    
    # 分析
    stats = tracer.get_statistics()
    print(f"\n资源利用率:")
    print(f"  NPU_0: {stats['resource_utilization'].get('NPU_0', 0):.1f}%")
    print(f"  NPU_1: {stats['resource_utilization'].get('NPU_1', 0):.1f}%")
    
    # 生成可视化
    visualizer.plot_resource_timeline("concurrent_execution.png")
    print("\n✓ 生成图表: concurrent_execution.png")


def simulate_execution_with_details(queue_manager, tracer, max_time):
    """详细的执行模拟"""
    current_time = 0.0
    
    while current_time < max_time:
        executed_any = False
        
        # 检查每个资源队列
        for resource_id, queue in queue_manager.resource_queues.items():
            queue.advance_time(current_time)
            
            if not queue.is_busy():
                # 获取最高优先级的就绪任务
                next_task = queue.get_next_task()
                
                if next_task and next_task.ready_time <= current_time:
                    # 获取实际的执行时间
                    sub_seg = next_task.sub_segments[0] if next_task.sub_segments else None
                    if sub_seg:
                        duration = sub_seg.get_duration(queue.bandwidth)
                        end_time = current_time + duration
                        
                        # 记录执行
                        tracer.record_execution(
                            next_task.task_id,
                            resource_id,
                            current_time,
                            end_time,
                            queue.bandwidth
                        )
                        
                        print(f"  {current_time:.1f}ms: {resource_id} 开始执行 "
                              f"{next_task.task_id} (优先级: {next_task.priority.name}, "
                              f"时长: {duration}ms)")
                        
                        # 更新资源状态
                        queue.busy_until = end_time
                        queue.current_task = next_task.task_id
                        
                        # 从队列移除
                        queue.dequeue_task(next_task.task_id, next_task.priority)
                        
                        executed_any = True
        
        # 时间推进
        if not executed_any:
            current_time += 0.1
        else:
            # 跳到下一个事件
            next_event = max_time
            for queue in queue_manager.resource_queues.values():
                if queue.is_busy() and queue.busy_until < next_event:
                    next_event = queue.busy_until
            current_time = min(next_event, max_time)


if __name__ == "__main__":
    # 运行所有测试
    test_launch_vs_execution()
    test_segmented_task_execution()
    test_concurrent_execution()
    
    print("\n\n✅ 所有测试完成！")
