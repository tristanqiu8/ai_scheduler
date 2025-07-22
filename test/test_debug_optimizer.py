#!/usr/bin/env python3
"""
调试优化器执行问题
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core import (
    ResourceType, TaskPriority,
    ResourceQueueManager, ScheduleTracer,
    TaskLauncher, ScheduleExecutor,
    LaunchOptimizer, OptimizationConfig
)
from scenario.real_task import create_real_tasks


def debug_optimizer_execution():
    """调试优化器执行问题"""
    print("="*80)
    print("调试优化器执行问题")
    print("="*80)
    
    # 1. 创建基础系统
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 2. 使用一个简单的任务集
    tasks = create_real_tasks()
    selected_tasks = [tasks[0], tasks[5]]  # T1 (MOTR) 和 T6 (reid)
    
    for task in selected_tasks:
        launcher.register_task(task)
    
    print("\n任务:")
    for task in selected_tasks:
        print(f"  {task.task_id}: {task.name} (FPS={task.fps_requirement})")
    
    # 3. 创建基线发射计划
    time_window = 50.0
    base_plan = launcher.create_launch_plan(time_window, "eager")
    
    print(f"\n基线发射计划 ({len(base_plan.events)} 个事件):")
    for i, event in enumerate(base_plan.events[:5]):
        print(f"  {event.time:>6.1f}ms: {event.task_id}#{event.instance_id}")
    if len(base_plan.events) > 5:
        print(f"  ... 还有 {len(base_plan.events)-5} 个事件")
    
    # 4. 执行基线计划
    print("\n执行基线计划...")
    executor_base = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    stats_base = executor_base.execute_plan(base_plan, time_window)
    
    print(f"\n基线执行结果:")
    print(f"  执行段数: {stats_base['total_segments_executed']}")
    print(f"  完成实例: {stats_base['completed_instances']}/{stats_base['total_instances']}")
    
    # 5. 测试优化器
    print("\n\n" + "="*80)
    print("测试优化器")
    print("="*80)
    
    # 创建新的系统用于优化
    queue_manager_opt = ResourceQueueManager()
    queue_manager_opt.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager_opt.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager_opt.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager_opt.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer_opt = ScheduleTracer(queue_manager_opt)
    launcher_opt = TaskLauncher(queue_manager_opt, tracer_opt)
    
    for task in selected_tasks:
        launcher_opt.register_task(task)
    
    # 创建简单的优化配置
    config = OptimizationConfig(
        max_iterations=1,  # 只运行一次迭代
        population_size=1  # 单个体
    )
    
    optimizer = LaunchOptimizer(launcher_opt, queue_manager_opt, config)
    
    # 测试策略评估
    print("\n测试策略评估...")
    from core.launch_optimizer import LaunchStrategy
    test_strategy = LaunchStrategy(strategy_type="eager")
    
    print("  评估基础策略...")
    metrics = optimizer._evaluate_strategy(test_strategy, time_window)
    
    print(f"\n  评估结果:")
    print(f"    空闲时间: {metrics.idle_time:.1f}ms ({metrics.idle_time_ratio:.1f}%)")
    print(f"    FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
    print(f"    执行段数: {metrics.total_segments}")
    
    # 检查执行器状态
    print("\n  检查资源队列状态:")
    for res_id, queue in queue_manager_opt.resource_queues.items():
        print(f"    {res_id}: 总执行数={queue.total_tasks_executed}")
    
    # 6. 测试修改后的计划
    print("\n\n测试修改后的发射计划...")
    
    # 创建一个有延迟的策略
    delayed_strategy = LaunchStrategy(strategy_type="custom")
    delayed_strategy.delay_factors["T1"] = 0.0  # T1不延迟
    delayed_strategy.delay_factors["T6"] = 5.0  # T6延迟5ms
    
    # 应用策略到计划
    base_plan_opt = launcher_opt.create_launch_plan(time_window, "eager")
    modified_plan = delayed_strategy.apply_to_plan(base_plan_opt)
    
    print(f"\n修改后的发射计划 ({len(modified_plan.events)} 个事件):")
    for i, event in enumerate(modified_plan.events[:5]):
        print(f"  {event.time:>6.1f}ms: {event.task_id}#{event.instance_id}")
    
    # 执行修改后的计划
    print("\n执行修改后的计划...")
    
    # 重新创建执行环境
    queue_manager_test = ResourceQueueManager()
    queue_manager_test.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager_test.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager_test.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager_test.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer_test = ScheduleTracer(queue_manager_test)
    executor_test = ScheduleExecutor(queue_manager_test, tracer_test, launcher_opt.tasks)
    
    stats_test = executor_test.execute_plan(modified_plan, time_window)
    
    print(f"\n修改后执行结果:")
    print(f"  执行段数: {stats_test['total_segments_executed']}")
    print(f"  完成实例: {stats_test['completed_instances']}/{stats_test['total_instances']}")
    
    # 显示执行时间线
    from viz.schedule_visualizer import ScheduleVisualizer
    visualizer = ScheduleVisualizer(tracer_test)
    print("\n执行时间线:")
    visualizer.print_gantt_chart(width=60)


if __name__ == "__main__":
    debug_optimizer_execution()
