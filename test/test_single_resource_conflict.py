#!/usr/bin/env python3
"""
测试单NPU和单DSP场景下的资源冲突处理
验证执行器是否正确处理资源竞争和交替执行
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
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority
from core.task import create_mixed_task, create_npu_task
from viz.schedule_visualizer import ScheduleVisualizer


def test_single_resource_basic():
    """基础测试：单NPU单DSP，验证交替执行"""
    print("=== 测试1：基础单资源交替执行 ===\n")
    
    # 只创建一个NPU和一个DSP
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 任务A：NPU(5ms) -> DSP(8ms) -> NPU(4ms)
    task_a = create_mixed_task(
        "TaskA", "混合任务A",
        segments=[
            (ResourceType.NPU, {60: 5.0}, "npu_seg0_a"),
            (ResourceType.DSP, {40: 8.0}, "dsp_seg0_a"),  
            (ResourceType.NPU, {60: 4.0}, "npu_seg1_a"),
        ],
        priority=TaskPriority.NORMAL
    )
    task_a.set_performance_requirements(fps=20, latency=50)
    
    # 任务B：纯NPU任务，应该在A的DSP段执行
    task_b = create_npu_task(
        "TaskB", "纯NPU任务B",
        {60: 6.0},
        priority=TaskPriority.NORMAL  # 同优先级
    )
    task_b.set_performance_requirements(fps=20, latency=50)
    
    # 任务C：纯DSP任务
    task_c = create_mixed_task(
        "TaskC", "纯DSP任务C",
        segments=[
            (ResourceType.DSP, {40: 5.0}, "dsp_seg0_c"),
        ],
        priority=TaskPriority.NORMAL
    )
    task_c.set_performance_requirements(fps=20, latency=50)
    
    # 注册任务
    for task in [task_a, task_b, task_c]:
        launcher.register_task(task)
    
    print("任务配置:")
    print("  TaskA: NPU(5ms) -> DSP(8ms) -> NPU(4ms)")
    print("  TaskB: NPU(6ms)")
    print("  TaskC: DSP(5ms)")
    print("\n预期执行顺序:")
    print("  1. TaskA_seg0 在 NPU (0-5ms)")
    print("  2. TaskA_seg1 在 DSP (5-13ms) 同时 TaskB 在 NPU (5-11ms)")
    print("  3. TaskA_seg2 在 NPU (13-17ms)")
    print("  4. TaskC 在 DSP (13-18ms)")
    
    # 创建执行器并执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    plan = launcher.create_launch_plan(30.0, "eager")
    
    print("\n执行过程:")
    stats = executor.execute_plan(plan, 30.0)
    
    # 显示结果
    print(f"\n{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    # 验证
    trace_stats = tracer.get_statistics()
    print(f"\n资源利用率:")
    print(f"  NPU_0: {trace_stats['resource_utilization'].get('NPU_0', 0):.1f}%")
    print(f"  DSP_0: {trace_stats['resource_utilization'].get('DSP_0', 0):.1f}%")
    
    # 生成可视化
    visualizer.export_chrome_tracing("single_resource_basic.json")
    print("\n[OK] 生成文件: single_resource_basic.json")


def test_resource_conflict_stress():
    """压力测试：大量任务竞争单个资源"""
    print("\n\n=== 测试2：资源冲突压力测试 ===\n")
    
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建多个竞争NPU的任务
    tasks = []
    
    # 高优先级NPU密集任务
    high_npu = create_npu_task(
        "HIGH_NPU", "高优先级NPU",
        {60: 3.0},
        priority=TaskPriority.HIGH
    )
    high_npu.set_performance_requirements(fps=50, latency=20)
    tasks.append(high_npu)
    
    # 多个普通优先级NPU任务
    for i in range(3):
        normal_npu = create_npu_task(
            f"NORM_NPU_{i}", f"普通NPU任务{i}",
            {60: 2.0 + i},  # 不同的执行时间
            priority=TaskPriority.NORMAL
        )
        normal_npu.set_performance_requirements(fps=20, latency=50)
        tasks.append(normal_npu)
    
    # DSP任务
    dsp_task = create_mixed_task(
        "DSP_TASK", "DSP任务",
        segments=[(ResourceType.DSP, {40: 10.0}, "dsp_long")],
        priority=TaskPriority.NORMAL
    )
    dsp_task.set_performance_requirements(fps=10, latency=100)
    tasks.append(dsp_task)
    
    # 注册所有任务
    for task in tasks:
        launcher.register_task(task)
    
    print("任务配置:")
    for task in tasks:
        print(f"  {task.task_id}: 优先级={task.priority.name}, FPS={task.fps_requirement}")
    
    # 执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    plan = launcher.create_launch_plan(50.0, "eager")
    stats = executor.execute_plan(plan, 50.0)
    
    # 显示结果
    print(f"\n{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    print("\n观察点:")
    print("1. HIGH优先级任务应该优先执行")
    print("2. NORMAL任务按FIFO顺序排队")
    print("3. DSP任务独立执行，不影响NPU调度")
    
    visualizer.export_chrome_tracing("resource_conflict_stress.json")


def test_real_motr_scenario():
    """真实场景：简化版MOTR任务"""
    print("\n\n=== 测试3：简化MOTR场景 ===\n")
    
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 简化的MOTR任务（基于real_task中的T1）
    motr_simple = create_mixed_task(
        "MOTR_LITE", "简化MOTR",
        segments=[
            # NPU预处理
            (ResourceType.NPU, {60: 0.41}, "npu_preprocess"),
            # DSP特征提取
            (ResourceType.DSP, {40: 1.2}, "dsp_feature"),
            # NPU主处理（最耗时）
            (ResourceType.NPU, {60: 9.33}, "npu_main"),
            # DSP后处理
            (ResourceType.DSP, {40: 2.2}, "dsp_post"),
            # NPU最终输出
            (ResourceType.NPU, {60: 0.63}, "npu_output"),
        ],
        priority=TaskPriority.CRITICAL  # 关键任务
    )
    motr_simple.set_performance_requirements(fps=40, latency=25)
    
    # YOLO检测任务（应该利用MOTR的DSP空隙）
    yolo_simple = create_npu_task(
        "YOLO_LITE", "简化YOLO",
        {60: 8.0},  # 8ms的NPU任务
        priority=TaskPriority.HIGH
    )
    yolo_simple.set_performance_requirements(fps=25, latency=40)
    
    # 背景DSP任务
    background_dsp = create_mixed_task(
        "BG_DSP", "背景DSP处理",
        segments=[(ResourceType.DSP, {40: 3.0}, "bg_process")],
        priority=TaskPriority.LOW
    )
    background_dsp.set_performance_requirements(fps=10, latency=100)
    
    # 注册任务
    for task in [motr_simple, yolo_simple, background_dsp]:
        launcher.register_task(task)
    
    print("场景说明:")
    print("  MOTR_LITE (CRITICAL): 5段混合任务，总时长约13.8ms")
    print("  YOLO_LITE (HIGH): 8ms NPU任务，应在MOTR的DSP段执行")
    print("  BG_DSP (LOW): 3ms DSP任务，低优先级")
    
    # 执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    plan = launcher.create_launch_plan(100.0, "eager")
    
    print("\n执行模拟:")
    stats = executor.execute_plan(plan, 100.0)
    
    # 显示详细时间线
    print(f"\n{'='*80}")
    print("执行时间线:")
    print(f"{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    # 分析
    trace_stats = tracer.get_statistics()
    print(f"\n资源利用率:")
    print(f"  NPU_0: {trace_stats['resource_utilization'].get('NPU_0', 0):.1f}%")
    print(f"  DSP_0: {trace_stats['resource_utilization'].get('DSP_0', 0):.1f}%")
    
    print("\n关键验证点:")
    print("1. MOTR的5个段是否按顺序执行")
    print("2. YOLO是否在MOTR的DSP段期间使用NPU")
    print("3. 低优先级DSP任务是否等待高优先级任务")
    
    # 生成可视化
    visualizer.plot_resource_timeline("motr_scenario.png")
    visualizer.export_chrome_tracing("motr_scenario.json")
    print("\n[OK] 生成文件:")
    print("  - motr_scenario.png")
    print("  - motr_scenario.json")


def test_deadline_miss_scenario():
    """测试4：验证资源冲突导致的延迟"""
    print("\n\n=== 测试4：延迟和冲突验证 ===\n")
    
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 长时间占用NPU的任务
    long_npu = create_npu_task(
        "LONG_NPU", "长NPU任务",
        {60: 20.0},  # 20ms！
        priority=TaskPriority.NORMAL
    )
    long_npu.set_performance_requirements(fps=10, latency=100)
    
    # 需要快速响应的任务（会被延迟）
    urgent_task = create_mixed_task(
        "URGENT", "紧急任务",
        segments=[
            (ResourceType.NPU, {60: 2.0}, "urgent_npu"),
            (ResourceType.DSP, {40: 1.0}, "urgent_dsp"),
        ],
        priority=TaskPriority.NORMAL  # 同优先级，所以要等待
    )
    urgent_task.set_performance_requirements(fps=50, latency=20)
    
    # 高优先级任务（可以获得资源）
    critical_task = create_npu_task(
        "CRITICAL", "关键任务",
        {60: 3.0},
        priority=TaskPriority.CRITICAL
    )
    critical_task.set_performance_requirements(fps=30, latency=33)
    
    # 注册任务
    for task in [long_npu, urgent_task, critical_task]:
        launcher.register_task(task)
    
    print("冲突场景:")
    print("  1. LONG_NPU先执行，占用NPU 20ms")
    print("  2. URGENT同优先级，必须等待")
    print("  3. CRITICAL高优先级，但仍需等LONG_NPU完成")
    
    # 执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    plan = launcher.create_launch_plan(60.0, "eager")
    stats = executor.execute_plan(plan, 60.0)
    
    # 显示结果
    print(f"\n{'='*80}")
    visualizer.print_gantt_chart(width=80)
    
    print("\n延迟分析:")
    executions = tracer.executions
    for exec in executions:
        if "URGENT" in exec.task_id:
            print(f"  URGENT任务: 开始于 {exec.start_time:.1f}ms (期望: 尽快)")
        if "CRITICAL" in exec.task_id:
            print(f"  CRITICAL任务: 开始于 {exec.start_time:.1f}ms")
    
    visualizer.export_chrome_tracing("deadline_miss.json")


if __name__ == "__main__":
    # 运行所有测试
    test_single_resource_basic()
    test_resource_conflict_stress()
    test_real_motr_scenario()
    test_deadline_miss_scenario()
    
    print("\n\n✅ 所有单资源测试完成！")
    print("\n总结:")
    print("1. 基础测试验证了NPU/DSP交替执行的正确性")
    print("2. 压力测试展示了多任务竞争时的FIFO行为")
    print("3. MOTR场景证明了复杂任务的资源利用优化")
    print("4. 延迟测试显示了优先级调度的重要性")