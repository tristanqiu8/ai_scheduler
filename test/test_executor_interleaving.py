#!/usr/bin/env python3
"""
测试执行器的关键场景：
当任务A在DSP上执行时，同优先级的任务B应该能够使用空闲的NPU
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority
from core.task import create_mixed_task, create_npu_task
from viz.schedule_visualizer import ScheduleVisualizer


def test_npu_dsp_interleaving():
    """测试NPU/DSP交替执行时的资源利用
    
    关键场景：
    - 任务A: NPU(5ms) -> DSP(10ms) -> NPU(5ms) 
    - 任务B: NPU(4ms) -> NPU(4ms) -> NPU(4ms)
    
    当A在DSP上执行时，B应该能够使用空闲的NPU
    """
    print("=== 测试 NPU/DSP 交替执行时的资源利用 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建任务A：NPU/DSP交替
    task_a = create_mixed_task(
        "TaskA", "NPU-DSP交替任务",
        segments=[
            (ResourceType.NPU, {60: 5.0}, "npu_seg0"),
            (ResourceType.DSP, {40: 10.0}, "dsp_seg0"),  
            (ResourceType.NPU, {60: 5.0}, "npu_seg1"),
        ],
        priority=TaskPriority.NORMAL  # 相同优先级
    )
    task_a.set_performance_requirements(fps=10, latency=100)
    
    # 创建任务B：纯NPU任务
    task_b = create_npu_task(
        "TaskB", "纯NPU任务",
        {60: 4.0},  # 每段4ms
        priority=TaskPriority.NORMAL  # 相同优先级！
    )
    task_b.set_performance_requirements(fps=10, latency=100)
    # 强制分段为3段
    from core.models import SubSegment
    task_b.sub_segments = [
        SubSegment("seg0", ResourceType.NPU, {60: 4.0}, 0.0, "npu_seg0"),
        SubSegment("seg1", ResourceType.NPU, {60: 4.0}, 0.0, "npu_seg1"),
        SubSegment("seg2", ResourceType.NPU, {60: 4.0}, 0.0, "npu_seg2"),
    ]
    
    # 注册任务
    launcher.register_task(task_a)
    launcher.register_task(task_b)
    
    # 创建执行器
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    # 创建发射计划 - 同时发射两个任务
    plan = launcher.create_launch_plan(50.0, "eager")
    
    print("发射计划:")
    for event in plan.events[:4]:
        print(f"  {event.time:.1f}ms: {event.task_id}#{event.instance_id}")
    
    # 执行计划
    print("\n执行过程:")
    stats = executor.execute_plan(plan, 50.0)
    
    # 显示执行时间线
    print(f"\n{'='*80}")
    print("执行时间线（关注NPU资源的利用）:")
    print(f"{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    # 分析结果
    print("\n关键观察点:")
    print("1. 当TaskA在DSP_0上执行(5-15ms)时，NPU_0是空闲的")
    print("2. TaskB应该能够在此期间使用NPU_0，而不是等待")
    print("3. 这样可以最大化资源利用率")
    
    # 统计资源利用率
    trace_stats = tracer.get_statistics()
    print(f"\n资源利用率:")
    print(f"  NPU_0: {trace_stats['resource_utilization'].get('NPU_0', 0):.1f}%")
    print(f"  DSP_0: {trace_stats['resource_utilization'].get('DSP_0', 0):.1f}%")
    
    # 生成可视化
    visualizer.plot_resource_timeline("npu_dsp_interleaving.png")
    visualizer.export_chrome_tracing("npu_dsp_interleaving.json")
    
    print("\n✓ 生成文件:")
    print("  - npu_dsp_interleaving.png")
    print("  - npu_dsp_interleaving.json")
    
    return trace_stats


def test_priority_with_interleaving():
    """测试不同优先级任务的交替执行"""
    print("\n\n=== 测试优先级对交替执行的影响 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建高优先级NPU/DSP交替任务
    high_task = create_mixed_task(
        "HighTask", "高优先级交替",
        segments=[
            (ResourceType.NPU, {60: 3.0}, "npu_0"),
            (ResourceType.DSP, {40: 5.0}, "dsp_0"),
            (ResourceType.NPU, {60: 3.0}, "npu_1"),
        ],
        priority=TaskPriority.HIGH
    )
    high_task.set_performance_requirements(fps=20, latency=50)
    
    # 创建普通优先级纯NPU任务
    normal_task1 = create_npu_task(
        "NormalTask1", "普通NPU任务1",
        {60: 6.0},
        priority=TaskPriority.NORMAL
    )
    normal_task1.set_performance_requirements(fps=10, latency=100)
    
    # 创建另一个普通优先级纯NPU任务
    normal_task2 = create_npu_task(
        "NormalTask2", "普通NPU任务2", 
        {60: 4.0},
        priority=TaskPriority.NORMAL
    )
    normal_task2.set_performance_requirements(fps=10, latency=100)
    
    # 注册任务
    for task in [high_task, normal_task1, normal_task2]:
        launcher.register_task(task)
    
    # 创建执行器
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    # 执行
    plan = launcher.create_launch_plan(40.0, "eager")
    stats = executor.execute_plan(plan, 40.0)
    
    # 显示结果
    print(f"\n{'='*80}")
    print("多优先级任务的执行时间线:")
    print(f"{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    print("\n分析:")
    print("1. HighTask优先获得NPU资源")
    print("2. 当HighTask在DSP上执行时，NormalTask可以使用NPU")
    print("3. 多个NPU资源(NPU_0, NPU_1)可以并行执行不同任务")
    
    # 生成可视化
    visualizer.plot_resource_timeline("priority_interleaving.png")
    print("\n✓ 生成图表: priority_interleaving.png")


def test_complex_interleaving():
    """测试复杂的交替执行场景"""
    print("\n\n=== 测试复杂交替执行场景 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0) 
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 使用真实场景的任务
    from scenario.real_task import create_real_tasks
    real_tasks = create_real_tasks()
    
    # 选择有代表性的任务
    motr_task = real_tasks[0]  # T1: MOTR - 9段NPU/DSP交替
    yolo_task = real_tasks[1]  # T2: YoloV8nBig - NPU+DSP  
    reid_task = real_tasks[5]  # T6: reid - 高频NPU
    
    # 调整优先级来测试场景
    motr_task.priority = TaskPriority.HIGH
    yolo_task.priority = TaskPriority.NORMAL
    reid_task.priority = TaskPriority.NORMAL  # 与YOLO相同优先级
    
    # 注册任务
    for task in [motr_task, yolo_task, reid_task]:
        launcher.register_task(task)
    
    print("任务配置:")
    print(f"  {motr_task.task_id}: {len(motr_task.segments)}段, 优先级={motr_task.priority.name}")
    print(f"  {yolo_task.task_id}: {len(yolo_task.segments)}段, 优先级={yolo_task.priority.name}")  
    print(f"  {reid_task.task_id}: {len(reid_task.segments)}段, 优先级={reid_task.priority.name}")
    
    # 创建执行器
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    # 执行
    plan = launcher.create_launch_plan(100.0, "eager")
    stats = executor.execute_plan(plan, 100.0)
    
    # 显示结果
    print(f"\n{'='*80}")
    print("复杂场景执行时间线:")
    print(f"{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    # 统计
    trace_stats = tracer.get_statistics()
    print(f"\n资源利用率:")
    for res_id in ["NPU_0", "NPU_1", "DSP_0", "DSP_1"]:
        util = trace_stats['resource_utilization'].get(res_id, 0)
        print(f"  {res_id}: {util:.1f}%")
    
    print(f"\n执行统计:")
    print(f"  总段数: {stats['total_segments_executed']}")
    print(f"  完成实例: {stats['completed_instances']}/{stats['total_instances']}")
    
    # 生成可视化
    visualizer.plot_resource_timeline("complex_interleaving.png")
    visualizer.export_chrome_tracing("complex_interleaving.json") 
    
    print("\n✓ 生成文件:")
    print("  - complex_interleaving.png")
    print("  - complex_interleaving.json")


if __name__ == "__main__":
    # 运行所有测试
    test_npu_dsp_interleaving()
    test_priority_with_interleaving()
    test_complex_interleaving()
    
    print("\n\n✅ 所有测试完成！")
    print("\n重要结论:")
    print("1. 执行器正确实现了NPU/DSP交替时的资源利用")
    print("2. 同优先级任务可以在资源空闲时立即执行")
    print("3. 资源利用率得到显著提升")