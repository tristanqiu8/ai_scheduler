#!/usr/bin/env python3
"""
测试有优化潜力的场景
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ResourceType, TaskPriority, NNTask,
    ResourceQueueManager, ScheduleTracer,
    TaskLauncher, ScheduleExecutor,
    LaunchOptimizer, OptimizationConfig,
    PerformanceEvaluator
)


def create_optimization_scenario():
    """创建一个有明显优化空间的场景"""
    tasks = []
    
    # 1. 高优先级关键任务 - 必须尽快执行
    critical_task = NNTask("CRITICAL_1", "关键检测", priority=TaskPriority.CRITICAL)
    critical_task.add_segment(ResourceType.NPU, {60: 5.0}, "detect")
    critical_task.add_segment(ResourceType.DSP, {40: 3.0}, "process")
    critical_task.set_performance_requirements(fps=20, latency=50)
    tasks.append(critical_task)
    
    # 2. 中优先级定期任务 - 可以适度延迟
    regular_task1 = NNTask("REGULAR_1", "常规分析1", priority=TaskPriority.NORMAL)
    regular_task1.add_segment(ResourceType.NPU, {60: 8.0}, "analyze")
    regular_task1.set_performance_requirements(fps=10, latency=100)
    tasks.append(regular_task1)
    
    regular_task2 = NNTask("REGULAR_2", "常规分析2", priority=TaskPriority.NORMAL)
    regular_task2.add_segment(ResourceType.NPU, {60: 6.0}, "analyze")
    regular_task2.set_performance_requirements(fps=10, latency=100)
    tasks.append(regular_task2)
    
    # 3. 低优先级批处理任务 - 可以大幅延迟
    batch_task1 = NNTask("BATCH_1", "批处理1", priority=TaskPriority.LOW)
    batch_task1.add_segment(ResourceType.DSP, {40: 10.0}, "batch_process")
    batch_task1.set_performance_requirements(fps=2, latency=500)
    tasks.append(batch_task1)
    
    batch_task2 = NNTask("BATCH_2", "批处理2", priority=TaskPriority.LOW)
    batch_task2.add_segment(ResourceType.DSP, {40: 8.0}, "batch_process")
    batch_task2.set_performance_requirements(fps=2, latency=500)
    tasks.append(batch_task2)
    
    # 4. 混合资源任务 - 会在NPU和DSP间切换
    mixed_task = NNTask("MIXED_1", "混合处理", priority=TaskPriority.HIGH)
    mixed_task.add_segment(ResourceType.NPU, {60: 3.0}, "preprocess")
    mixed_task.add_segment(ResourceType.DSP, {40: 5.0}, "compute")
    mixed_task.add_segment(ResourceType.NPU, {60: 2.0}, "postprocess")
    mixed_task.set_performance_requirements(fps=15, latency=70)
    tasks.append(mixed_task)
    
    return tasks


def test_optimization_potential():
    """测试有优化潜力的场景"""
    print("="*80)
    print("测试优化潜力场景")
    print("="*80)
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 注册任务
    tasks = create_optimization_scenario()
    for task in tasks:
        launcher.register_task(task)
    
    print("\n任务配置:")
    print(f"{'任务ID':<12} {'名称':<12} {'优先级':<10} {'FPS要求':<8} {'资源需求'}")
    print("-" * 70)
    for task in tasks:
        segments_info = []
        for seg in task.segments:
            duration = list(seg.duration_table.values())[0]
            segments_info.append(f"{seg.resource_type.value}({duration:.1f}ms)")
        print(f"{task.task_id:<12} {task.name:<12} {task.priority.name:<10} "
              f"{task.fps_requirement:<8} {' -> '.join(segments_info)}")
    
    # 时间窗口设置为200ms
    time_window = 200.0
    
    # 1. 执行基线（激进策略）
    print("\n" + "="*80)
    print("1. 基线执行（激进策略 - 所有任务立即发射）")
    print("="*80)
    
    plan_eager = launcher.create_launch_plan(time_window, "eager")
    print(f"\n发射计划: {len(plan_eager.events)} 个事件")
    
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    stats_eager = executor.execute_plan(plan_eager, time_window)
    
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    metrics_eager = evaluator.evaluate(time_window, plan_eager.events)
    
    print(f"\n基线性能:")
    print(f"  完成实例: {stats_eager['completed_instances']}/{stats_eager['total_instances']}")
    print(f"  空闲时间: {metrics_eager.idle_time:.1f}ms ({metrics_eager.idle_time_ratio:.1f}%)")
    print(f"  FPS满足率: {metrics_eager.fps_satisfaction_rate:.1f}%")
    print(f"  资源利用率:")
    print(f"    NPU平均: {metrics_eager.avg_npu_utilization:.1f}%")
    print(f"    DSP平均: {metrics_eager.avg_dsp_utilization:.1f}%")
    
    # 显示任务性能
    print(f"\n任务性能详情:")
    for task_id, task_metrics in evaluator.task_metrics.items():
        # 计算FPS满足率（百分比）
        fps_rate = 100.0 if task_metrics.fps_satisfaction else 0.0
        print(f"  {task_id}: FPS满足={fps_rate:.1f}%, "
              f"平均延迟={task_metrics.avg_latency:.1f}ms")
    
    # 2. 优化发射策略
    print("\n" + "="*80)
    print("2. 优化发射策略")
    print("="*80)
    
    # 为优化创建新的环境
    queue_manager_opt = ResourceQueueManager()
    queue_manager_opt.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager_opt.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager_opt.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager_opt.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer_opt = ScheduleTracer(queue_manager_opt)
    launcher_opt = TaskLauncher(queue_manager_opt, tracer_opt)
    
    for task in tasks:
        launcher_opt.register_task(task)
    
    # 优化配置
    config = OptimizationConfig(
        max_iterations=20,
        population_size=30,
        idle_time_weight=0.6,
        fps_satisfaction_weight=0.3,
        resource_balance_weight=0.1,
        fps_tolerance=0.95  # 允许5%的FPS降低
    )
    
    # 使用静默版本的优化器（如果存在）
    try:
        from quiet_launch_optimizer import QuietLaunchOptimizer
        optimizer = QuietLaunchOptimizer(launcher_opt, queue_manager_opt, config, verbose=False)
    except ImportError:
        # 使用原始优化器
        optimizer = LaunchOptimizer(launcher_opt, queue_manager_opt, config)
    
    print("\n开始优化...")
    print(f"  目标: 最大化空闲时间，同时保持95%以上的FPS满足率")
    print(f"  策略: 延迟低优先级任务，批量处理相似任务")
    
    best_strategy = optimizer.optimize(time_window, "eager")
    
    if optimizer.best_metrics:
        print(f"\n✨ 优化结果:")
        print(f"  最佳策略: {best_strategy.strategy_type}")
        print(f"  延迟因子: {best_strategy.delay_factors}")
        print(f"  空闲时间: {optimizer.best_metrics.idle_time:.1f}ms "
              f"({optimizer.best_metrics.idle_time_ratio:.1f}%)")
        print(f"  FPS满足率: {optimizer.best_metrics.fps_satisfaction_rate:.1f}%")
        
        # 计算改进
        idle_improvement = optimizer.best_metrics.idle_time - metrics_eager.idle_time
        print(f"\n改进:")
        print(f"  空闲时间增加: +{idle_improvement:.1f}ms")
        print(f"  空闲时间比例: {metrics_eager.idle_time_ratio:.1f}% -> "
              f"{optimizer.best_metrics.idle_time_ratio:.1f}%")
    
    # 3. 可视化对比
    print("\n" + "="*80)
    print("3. 生成可视化对比")
    print("="*80)
    
    from viz.schedule_visualizer import ScheduleVisualizer
    
    # 基线可视化
    visualizer_eager = ScheduleVisualizer(tracer)
    print("\n基线调度（激进策略）:")
    visualizer_eager.print_gantt_chart(width=80)
    
    # 优化后可视化（需要实际执行优化后的计划）
    if optimizer.best_strategy:
        print("\n执行优化后的计划...")
        optimized_plan = best_strategy.apply_to_plan(
            launcher_opt.create_launch_plan(time_window, "eager")
        )
        
        # 创建新环境执行
        queue_manager_vis = ResourceQueueManager()
        queue_manager_vis.add_resource("NPU_0", ResourceType.NPU, 60.0)
        queue_manager_vis.add_resource("NPU_1", ResourceType.NPU, 60.0)
        queue_manager_vis.add_resource("DSP_0", ResourceType.DSP, 40.0)
        queue_manager_vis.add_resource("DSP_1", ResourceType.DSP, 40.0)
        
        tracer_vis = ScheduleTracer(queue_manager_vis)
        executor_vis = ScheduleExecutor(queue_manager_vis, tracer_vis, launcher_opt.tasks)
        executor_vis.execute_plan(optimized_plan, time_window)
        
        visualizer_opt = ScheduleVisualizer(tracer_vis)
        print("\n优化后调度:")
        visualizer_opt.print_gantt_chart(width=80)


if __name__ == "__main__":
    test_optimization_potential()
