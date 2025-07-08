#!/usr/bin/env python3
"""
测试段感知执行器的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enums import ResourceType, TaskPriority
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer

# 确保 SegmentAwareExecutor 可以被导入
try:
    # 如果已经复制到 core 目录
    from core.segment_aware_executor import SegmentAwareExecutor
except ImportError:
    try:
        # 尝试从当前目录导入
        from segment_aware_executor import SegmentAwareExecutor
    except ImportError:
        print("错误：无法导入 SegmentAwareExecutor")
        print("请确保 segment_aware_executor.py 文件存在于:")
        print("  1. core/ 目录下，或")
        print("  2. 当前测试目录下")
        sys.exit(1)


def test_mixed_mode_execution():
    """测试混合模式执行：T1整体发射，T2/T3段级发射"""
    print("=== 测试混合模式执行 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 获取真实任务
    tasks = create_real_tasks()
    test_tasks = {
        "T1": tasks[0],  # MOTR - 9段，整体发射
        "T2": tasks[1],  # YoloV8nBig - 2段，段级发射
        "T3": tasks[2],  # Lpr - 2段，段级发射
    }
    
    # 注册任务
    for task_id, task in test_tasks.items():
        launcher.register_task(task)
        print(f"{task_id}: {len(task.segments)}段, priority={task.priority.name}")
    
    # 创建发射计划
    plan = launcher.create_launch_plan(200.0, "eager")
    
    # 使用新的执行器
    executor = SegmentAwareExecutor(queue_manager, tracer, launcher.tasks)
    
    print("\n执行计划...")
    stats = executor.execute_plan(plan, 200.0)
    
    # 显示结果
    print("\n" + "="*80)
    print("执行时间线:")
    print("="*80)
    visualizer.print_gantt_chart(width=80)
    
    print(f"\n执行统计:")
    print(f"  总实例数: {stats['total_instances']}")
    print(f"  完成实例: {stats['completed_instances']}")
    print(f"  总段数: {stats['total_segments']}")
    print(f"  完成段数: {stats['completed_segments']}")
    
    # 生成可视化
    visualizer.plot_resource_timeline("mixed_mode_execution.png")
    print("\n生成了可视化文件: mixed_mode_execution.png")


def test_segment_interleaving():
    """测试段交织执行的效果"""
    print("\n\n=== 测试段交织执行 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建测试任务
    from core.task import create_mixed_task
    
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
    
    # 创建执行器，设置A和B都使用段级发射
    executor = SegmentAwareExecutor(queue_manager, tracer, launcher.tasks)
    executor.segment_launch_tasks = {"TaskA", "TaskB"}
    
    # 创建发射计划
    plan = launcher.create_launch_plan(100.0, "eager")
    
    print("段级发射模式下的执行:")
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


def test_performance_improvement():
    """对比整体发射和段级发射的性能差异"""
    print("\n\n=== 性能对比测试 ===\n")
    
    # 测试配置
    test_duration = 100.0
    
    # 场景1：所有任务整体发射
    print("场景1：所有任务整体发射")
    queue_manager1 = ResourceQueueManager()
    queue_manager1.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager1.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer1 = ScheduleTracer(queue_manager1)
    launcher1 = TaskLauncher(queue_manager1, tracer1)
    
    # 使用真实任务T2和T3
    tasks = create_real_tasks()
    launcher1.register_task(tasks[1])  # T2
    launcher1.register_task(tasks[2])  # T3
    
    plan1 = launcher1.create_launch_plan(test_duration, "eager")
    executor1 = SegmentAwareExecutor(queue_manager1, tracer1, launcher1.tasks)
    executor1.segment_launch_tasks = set()  # 空集合，所有任务整体发射
    
    stats1 = executor1.execute_plan(plan1, test_duration)
    
    # 场景2：T2/T3段级发射
    print("\n场景2：T2/T3段级发射")
    queue_manager2 = ResourceQueueManager()
    queue_manager2.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager2.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer2 = ScheduleTracer(queue_manager2)
    launcher2 = TaskLauncher(queue_manager2, tracer2)
    
    launcher2.register_task(tasks[1])  # T2
    launcher2.register_task(tasks[2])  # T3
    
    plan2 = launcher2.create_launch_plan(test_duration, "eager")
    executor2 = SegmentAwareExecutor(queue_manager2, tracer2, launcher2.tasks)
    executor2.segment_launch_tasks = {"T2", "T3"}  # 段级发射
    
    stats2 = executor2.execute_plan(plan2, test_duration)
    
    # 对比结果
    print("\n性能对比:")
    print(f"  整体发射完成实例: {stats1['completed_instances']}")
    print(f"  段级发射完成实例: {stats2['completed_instances']}")
    
    # 资源利用率对比
    util1 = tracer1.get_resource_utilization()
    util2 = tracer2.get_resource_utilization()
    
    print("\n资源利用率对比:")
    for res_id in ["NPU_0", "DSP_0"]:
        u1 = util1.get(res_id, 0.0)
        u2 = util2.get(res_id, 0.0)
        print(f"  {res_id}:")
        print(f"    整体发射: {u1:.1f}%")
        print(f"    段级发射: {u2:.1f}%")
        if u1 > 0:
            print(f"    提升: {u2-u1:.1f}% (相对提升 {((u2-u1)/u1)*100:.1f}%)")
    
    # 生成对比可视化
    from viz.schedule_visualizer import ScheduleVisualizer
    visualizer1 = ScheduleVisualizer(tracer1)
    visualizer2 = ScheduleVisualizer(tracer2)
    
    visualizer1.plot_resource_timeline("whole_launch_mode.png")
    visualizer2.plot_resource_timeline("segment_launch_mode.png")
    
    print("\n生成了对比可视化:")
    print("  - whole_launch_mode.png")
    print("  - segment_launch_mode.png")


def test_with_ga_optimizer():
    """测试新执行器与GA优化器的集成"""
    print("\n\n=== 测试与GA优化器集成 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 获取真实任务
    tasks = create_real_tasks()
    
    # 选择一些任务进行优化
    opt_tasks = []
    for i in [1, 2, 3, 5, 6]:  # T2, T3, T4, T6, T7
        task = tasks[i]
        launcher.register_task(task)
        opt_tasks.append(task)
    
    print("待优化任务:")
    for task in opt_tasks:
        print(f"  {task.task_id}: {len(task.segments)}段, priority={task.priority.name}")
    
    # 创建段感知执行器
    executor = SegmentAwareExecutor(queue_manager, tracer, launcher.tasks)
    
    # 运行基准测试
    print("\n运行基准测试...")
    plan = launcher.create_launch_plan(200.0, "eager")
    baseline_stats = executor.execute_plan(plan, 200.0)
    
    print(f"\n基准结果:")
    print(f"  完成实例: {baseline_stats['completed_instances']}")
    print(f"  总执行时间: {baseline_stats['current_time']:.1f}ms")
    
    # 这里可以集成GA优化器
    print("\n注：此处可集成GA优化器来优化:")
    print("  - 任务优先级")
    print("  - 分段策略")
    print("  - 资源分配")
    print("\n段级发射模式为GA优化提供了更大的优化空间！")


if __name__ == "__main__":
    # 运行所有测试
    test_mixed_mode_execution()
    test_segment_interleaving()
    test_performance_improvement()
    test_with_ga_optimizer()
