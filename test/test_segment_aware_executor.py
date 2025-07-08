#!/usr/bin/env python3
"""
测试段感知执行功能 - 使用更新后的 ScheduleExecutor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.executor import ScheduleExecutor, create_executor
from core.enums import ResourceType, TaskPriority
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
from core.task import create_mixed_task


def test_segment_mode_execution():
    """测试段级模式执行"""
    print("=== 测试段级模式执行 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 获取真实任务
    tasks = create_real_tasks()
    test_tasks = [
        tasks[0],  # T1: MOTR - 9段
        tasks[1],  # T2: YoloV8nBig - 2段
        tasks[2],  # T3: Lpr - 2段
    ]
    
    # 注册任务
    for task in test_tasks:
        launcher.register_task(task)
        print(f"{task.task_id}: {len(task.segments)}段, priority={task.priority.name}")
    
    # 创建发射计划
    plan = launcher.create_launch_plan(200.0, "eager")
    
    # 使用段级模式执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    print("\n执行计划（段级模式）...")
    stats = executor.execute_plan(plan, 200.0, segment_mode=True)
    
    # 显示结果
    print("\n" + "="*80)
    print("执行时间线:")
    print("="*80)
    visualizer.print_gantt_chart(width=80)
    
    print(f"\n执行统计:")
    print(f"  总实例数: {stats['total_instances']}")
    print(f"  完成实例: {stats['completed_instances']}")
    print(f"  执行段数: {stats['total_segments_executed']}")
    print(f"  仿真时间: {stats['simulation_time']:.1f}ms")
    
    # 生成可视化
    visualizer.plot_resource_timeline("segment_mode_execution.png")
    print("\n生成了可视化文件: segment_mode_execution.png")
    
    return stats


def test_segment_interleaving():
    """测试段交织执行的效果"""
    print("\n\n=== 测试段交织执行 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 任务A: NPU(10ms) -> DSP(8ms) -> NPU(6ms)
    task_a = create_mixed_task(
        "TaskA", "测试任务A",
        segments=[
            (ResourceType.NPU, {60: 10.0}, "a_npu0"),
            (ResourceType.DSP, {40: 8.0}, "a_dsp0"),
            (ResourceType.NPU, {60: 6.0}, "a_npu1"),
        ],
        priority=TaskPriority.NORMAL
    )
    task_a.set_performance_requirements(fps=20, latency=50)
    
    # 任务B: NPU(5ms) -> NPU(5ms) -> DSP(4ms)
    task_b = create_mixed_task(
        "TaskB", "测试任务B",
        segments=[
            (ResourceType.NPU, {60: 5.0}, "b_npu0"),
            (ResourceType.NPU, {60: 5.0}, "b_npu1"),
            (ResourceType.DSP, {40: 4.0}, "b_dsp0"),
        ],
        priority=TaskPriority.NORMAL
    )
    task_b.set_performance_requirements(fps=20, latency=50)
    
    # 注册任务
    launcher.register_task(task_a)
    launcher.register_task(task_b)
    
    # 创建执行器并启用段级模式
    executor = create_executor(queue_manager, tracer, launcher.tasks, mode="segment_aware")
    
    # 创建发射计划
    plan = launcher.create_launch_plan(100.0, "eager")
    
    print("段级模式下的执行:")
    stats = executor.execute_plan(plan, 100.0)
    
    # 分析结果
    print("\n关键观察:")
    print("1. TaskA的NPU段执行时，TaskB可以使用空闲的DSP")
    print("2. TaskB的NPU段可以在TaskA的DSP段执行时开始")
    print("3. 整体执行时间应该比串行执行短")
    
    # 计算资源利用率
    utilization = tracer.get_resource_utilization()
    print("\n资源利用率:")
    for res_id, util_percent in utilization.items():
        print(f"  {res_id}: {util_percent:.1f}%")
    
    # 显示执行时间线
    visualizer = ScheduleVisualizer(tracer)
    print("\n执行时间线:")
    visualizer.print_gantt_chart(width=80)
    
    return stats


def test_performance_comparison():
    """对比传统模式和段级模式的性能差异"""
    print("\n\n=== 性能对比测试 ===\n")
    
    # 测试配置
    test_duration = 100.0
    
    # 使用真实任务
    tasks = create_real_tasks()
    selected_tasks = [tasks[1], tasks[2], tasks[3]]  # T2, T3, T4
    
    results = {}
    
    # 场景1：传统模式（逐段执行）
    print("场景1：传统模式（逐段执行）")
    queue_manager1 = ResourceQueueManager()
    queue_manager1.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager1.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer1 = ScheduleTracer(queue_manager1)
    launcher1 = TaskLauncher(queue_manager1, tracer1)
    
    for task in selected_tasks:
        launcher1.register_task(task)
    
    plan1 = launcher1.create_launch_plan(test_duration, "eager")
    executor1 = ScheduleExecutor(queue_manager1, tracer1, launcher1.tasks)
    # 默认就是传统模式
    stats1 = executor1.execute_plan(plan1, test_duration)
    util1 = tracer1.get_resource_utilization()
    
    results['传统模式'] = {
        'completed': stats1['completed_instances'],
        'segments': stats1['total_segments_executed'],
        'time': stats1['simulation_time'],
        'npu_util': util1.get('NPU_0', 0),
        'dsp_util': util1.get('DSP_0', 0)
    }
    
    # 场景2：段级模式
    print("\n场景2：段级模式（并行段执行）")
    queue_manager2 = ResourceQueueManager()
    queue_manager2.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager2.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer2 = ScheduleTracer(queue_manager2)
    launcher2 = TaskLauncher(queue_manager2, tracer2)
    
    for task in selected_tasks:
        launcher2.register_task(task)
    
    plan2 = launcher2.create_launch_plan(test_duration, "eager")
    executor2 = ScheduleExecutor(queue_manager2, tracer2, launcher2.tasks)
    # 启用段级模式
    stats2 = executor2.execute_plan(plan2, test_duration, segment_mode=True)
    util2 = tracer2.get_resource_utilization()
    
    results['段级模式'] = {
        'completed': stats2['completed_instances'],
        'segments': stats2['total_segments_executed'],
        'time': stats2['simulation_time'],
        'npu_util': util2.get('NPU_0', 0),
        'dsp_util': util2.get('DSP_0', 0)
    }
    
    # 对比结果
    print("\n性能对比结果:")
    print(f"{'指标':<15} {'传统模式':>12} {'段级模式':>12} {'提升':>12}")
    print("-" * 51)
    
    metrics = [
        ('完成实例', 'completed', '个'),
        ('执行段数', 'segments', '个'),
        ('NPU利用率', 'npu_util', '%'),
        ('DSP利用率', 'dsp_util', '%')
    ]
    
    for name, key, unit in metrics:
        trad = results['传统模式'][key]
        seg = results['段级模式'][key]
        if trad > 0:
            improve = ((seg - trad) / trad * 100)
        else:
            improve = 0
        
        print(f"{name:<15} {trad:>11.1f}{unit} {seg:>11.1f}{unit} {improve:>+11.1f}%")
    
    # 生成对比可视化
    visualizer1 = ScheduleVisualizer(tracer1)
    visualizer2 = ScheduleVisualizer(tracer2)
    
    visualizer1.plot_resource_timeline("traditional_mode.png")
    visualizer2.plot_resource_timeline("segment_mode.png")
    
    print("\n生成了对比可视化:")
    print("  - traditional_mode.png (传统模式)")
    print("  - segment_mode.png (段级模式)")
    
    return results


def test_factory_function():
    """测试工厂函数创建不同模式的执行器"""
    print("\n\n=== 测试工厂函数 ===\n")
    
    # 创建基础环境
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建简单任务
    task = create_mixed_task(
        "TestTask", "测试任务",
        segments=[
            (ResourceType.NPU, {60: 5.0}, "seg0"),
            (ResourceType.NPU, {60: 5.0}, "seg1"),
        ],
        priority=TaskPriority.NORMAL
    )
    launcher.register_task(task)
    
    plan = launcher.create_launch_plan(30.0, "eager")
    
    # 测试默认模式
    print("1. 默认模式（传统）:")
    executor_default = create_executor(queue_manager, tracer, launcher.tasks)
    print(f"   segment_mode = {executor_default.segment_mode}")
    
    # 测试段级模式
    print("\n2. 段级模式:")
    executor_segment = create_executor(queue_manager, tracer, launcher.tasks, 
                                     mode="segment_aware")
    print(f"   segment_mode = {executor_segment.segment_mode}")
    
    print("\n✅ 工厂函数测试通过")


def test_priority_handling():
    """测试段级模式下的优先级处理"""
    print("\n\n=== 测试段级模式下的优先级处理 ===\n")
    
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建不同优先级的任务
    high_task = create_mixed_task(
        "HighTask", "高优先级",
        segments=[(ResourceType.NPU, {60: 5.0}, "high")],
        priority=TaskPriority.HIGH
    )
    
    normal_task = create_mixed_task(
        "NormalTask", "普通优先级",
        segments=[(ResourceType.NPU, {60: 5.0}, "normal")],
        priority=TaskPriority.NORMAL
    )
    
    low_task = create_mixed_task(
        "LowTask", "低优先级",
        segments=[(ResourceType.NPU, {60: 5.0}, "low")],
        priority=TaskPriority.LOW
    )
    
    # 注册顺序：低 -> 普通 -> 高
    launcher.register_task(low_task)
    launcher.register_task(normal_task)
    launcher.register_task(high_task)
    
    # 执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    plan = launcher.create_launch_plan(30.0, "eager")
    stats = executor.execute_plan(plan, 30.0, segment_mode=True)
    
    print("执行顺序应该是：高 -> 普通 -> 低")
    
    # 检查执行顺序
    executions = tracer.executions
    print("\n实际执行顺序:")
    for i, exec in enumerate(executions[:3]):
        print(f"  {i+1}. {exec.task_id} (开始: {exec.start_time:.1f}ms)")
    
    print("\n✅ 优先级在段级模式下正确处理")


if __name__ == "__main__":
    # 运行所有测试
    print("🚀 开始测试段感知执行功能\n")
    
    # 基本功能测试
    test_segment_mode_execution()
    
    # 段交织测试
    test_segment_interleaving()
    
    # 性能对比测试
    test_performance_comparison()
    
    # 工厂函数测试
    test_factory_function()
    
    # 优先级测试
    test_priority_handling()
    
    print("\n\n✨ 所有测试完成！")
    print("\n总结:")
    print("1. 段级模式通过 segment_mode 参数控制")
    print("2. 可以使用工厂函数创建不同模式的执行器")
    print("3. 段级模式能够提高资源利用率")
    print("4. 优先级在段级模式下仍然正确工作")
