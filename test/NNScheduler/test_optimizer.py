#!/usr/bin/env python3
"""
修改LaunchOptimizer的_evaluate_strategy方法，使其在评估时不输出详细日志
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import Optional
from NNScheduler.core import (
    ResourceType, 
    ResourceQueueManager, ScheduleTracer,
    TaskLauncher, ScheduleExecutor,
    LaunchOptimizer, OptimizationConfig,
    PerformanceEvaluator
)
from NNScheduler.core.launch_optimizer import LaunchStrategy, OverallPerformanceMetrics
from NNScheduler.scenario.real_task import create_real_tasks


class QuietScheduleExecutor(ScheduleExecutor):
    """静默版本的执行器，不输出日志"""
    
    def __init__(self, *args, verbose: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
    
    def _log(self, time: float, message: str):
        """只在verbose模式下输出日志"""
        if self.verbose:
            super()._log(time, message)


class QuietLaunchOptimizer(LaunchOptimizer):
    """静默版本的优化器"""
    
    def __init__(self, *args, verbose: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self.evaluation_count = 0
    
    def _evaluate_strategy(self, strategy: LaunchStrategy, time_window: float) -> OverallPerformanceMetrics:
        """评估发射策略的性能（静默版本）"""
        self.evaluation_count += 1
        
        # 为每次评估创建独立的资源管理器
        eval_queue_manager = ResourceQueueManager()
        
        # 复制原始资源配置
        for res_id, queue in self.queue_manager.resource_queues.items():
            eval_queue_manager.add_resource(res_id, queue.resource_type, queue.bandwidth)
        
        # 创建新的tracer
        eval_tracer = ScheduleTracer(eval_queue_manager)
        
        # 生成基础发射计划
        base_plan = self.launcher.create_launch_plan(time_window, strategy.strategy_type)
        
        # 应用策略修改
        modified_plan = strategy.apply_to_plan(base_plan)
        
        # 使用静默执行器
        executor = QuietScheduleExecutor(
            eval_queue_manager, 
            eval_tracer, 
            self.launcher.tasks,
            verbose=self.verbose  # 只在verbose模式下输出
        )
        
        # 如果是第一次评估或verbose模式，显示简要信息
        if self.evaluation_count == 1 or self.verbose:
            print(f"\n评估策略 #{self.evaluation_count}: {strategy.strategy_type}")
        
        executor.execute_plan(modified_plan, time_window)
        
        # 评估性能
        evaluator = PerformanceEvaluator(eval_tracer, self.launcher.tasks, eval_queue_manager)
        metrics = evaluator.evaluate(time_window, modified_plan.events)
        
        # 显示简要结果
        if self.evaluation_count == 1 or self.verbose:
            print(f"  - 空闲时间: {metrics.idle_time:.1f}ms ({metrics.idle_time_ratio:.1f}%)")
            print(f"  - FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
        
        return metrics


def test_quiet_optimizer():
    """测试静默版本的优化器"""
    print("="*80)
    print("测试静默版优化器")
    print("="*80)
    
    # 创建资源和任务
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 使用测试任务
    tasks = create_real_tasks()
    selected_tasks = [tasks[0], tasks[5]]  # T1 和 T6
    
    for task in selected_tasks:
        launcher.register_task(task)
    
    print("\n任务配置:")
    for task in selected_tasks:
        print(f"  {task.task_id}: {task.name} (FPS={task.fps_requirement})")
    
    # 1. 首先执行基线以显示详细日志
    print("\n" + "="*80)
    print("1. 执行基线（显示详细日志）")
    print("="*80)
    
    plan = launcher.create_launch_plan(50.0, "eager")
    executor = QuietScheduleExecutor(queue_manager, tracer, launcher.tasks, verbose=True)
    stats = executor.execute_plan(plan, 50.0)
    
    print(f"\n基线结果:")
    print(f"  完成实例: {stats['completed_instances']}/{stats['total_instances']}")
    print(f"  执行段数: {stats['total_segments_executed']}")
    
    # 2. 使用静默优化器
    print("\n" + "="*80)
    print("2. 运行优化器（静默模式）")
    print("="*80)
    
    # 创建优化配置
    config = OptimizationConfig(
        max_iterations=10,
        population_size=20,
        idle_time_weight=0.7,
        fps_satisfaction_weight=0.3
    )
    
    # 使用静默优化器
    optimizer = QuietLaunchOptimizer(launcher, queue_manager, config, verbose=False)
    
    print("\n开始优化（静默模式，只显示摘要）...")
    best_strategy = optimizer.optimize(50.0, "eager")
    
    print(f"\n优化完成! 共评估了 {optimizer.evaluation_count} 个策略")
    
    if optimizer.best_metrics:
        print(f"\n最终结果:")
        print(f"  最佳策略: {best_strategy.strategy_type}")
        print(f"  空闲时间: {optimizer.best_metrics.idle_time:.1f}ms")
        print(f"  FPS满足率: {optimizer.best_metrics.fps_satisfaction_rate:.1f}%")
    
    # 3. 演示verbose模式
    print("\n" + "="*80)
    print("3. 运行优化器（verbose模式）")
    print("="*80)
    
    optimizer_verbose = QuietLaunchOptimizer(launcher, queue_manager, config, verbose=True)
    print("\n开始优化（verbose模式，显示所有细节）...")
    # 只运行几次迭代作为演示
    config_short = OptimizationConfig(max_iterations=2, population_size=3)
    optimizer_verbose.config = config_short
    optimizer_verbose.optimize(50.0, "eager")


if __name__ == "__main__":
    test_quiet_optimizer()
