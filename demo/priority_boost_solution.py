#!/usr/bin/env python3
"""
优先级提升解决方案 - 直接调整延迟紧张任务的优先级
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, List, Tuple
import copy
from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority
from core.evaluator import PerformanceEvaluator
from scenario.hybrid_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
import numpy as np


def analyze_task_performance(tasks: List, time_window: float = 1000.0) -> Dict[str, Dict]:
    """分析任务性能，找出延迟紧张的任务"""
    print("分析任务性能...")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 注册任务
    for task in tasks:
        launcher.register_task(task)
    
    # 执行基准测试
    plan = launcher.create_launch_plan(time_window, "eager")
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    executor.segment_mode = True
    executor.execute_plan(plan, time_window)
    
    # 评估性能
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    evaluator.evaluate(time_window, plan.events)
    
    # 分析结果
    results = {}
    for task_id, metrics in evaluator.task_metrics.items():
        task = launcher.tasks[task_id]
        results[task_id] = {
            'priority': task.priority,
            'fps_requirement': task.fps_requirement,
            'latency_requirement': task.latency_requirement,
            'achieved_fps': metrics.achieved_fps,
            'avg_latency': metrics.avg_latency,
            'max_latency': metrics.max_latency,
            'satisfaction_rate': metrics.latency_satisfaction_rate,
            'violations': len([l for l in metrics.latencies if l > task.latency_requirement]),
            'total_instances': len(metrics.latencies),
            'margin': task.latency_requirement - metrics.max_latency,  # 负数表示超标
            'needs_boost': metrics.latency_satisfaction_rate < 0.95  # 需要提升优先级
        }
    
    return results


def optimize_task_priorities(tasks: List, performance_data: Dict[str, Dict]) -> List:
    """基于性能数据优化任务优先级"""
    print("\n优化任务优先级...")
    
    # 复制任务列表
    optimized_tasks = copy.deepcopy(tasks)
    
    # 找出需要提升优先级的任务
    boost_candidates = []
    for task_id, data in performance_data.items():
        if data['needs_boost'] and data['priority'] != TaskPriority.CRITICAL:
            boost_candidates.append((task_id, data['margin'], data['satisfaction_rate']))
    
    # 按延迟余量排序（负数越小越紧急）
    boost_candidates.sort(key=lambda x: (x[1], x[2]))
    
    # 提升优先级
    priority_changes = []
    for task_id, margin, satisfaction in boost_candidates:
        for task in optimized_tasks:
            if task.task_id == task_id:
                old_priority = task.priority
                
                # 根据紧急程度决定提升幅度
                if margin < -10:  # 严重超标
                    if task.priority == TaskPriority.LOW:
                        task.priority = TaskPriority.HIGH
                    elif task.priority == TaskPriority.NORMAL:
                        task.priority = TaskPriority.CRITICAL
                    elif task.priority == TaskPriority.HIGH:
                        task.priority = TaskPriority.CRITICAL
                else:  # 轻微超标
                    if task.priority == TaskPriority.LOW:
                        task.priority = TaskPriority.NORMAL
                    elif task.priority == TaskPriority.NORMAL:
                        task.priority = TaskPriority.HIGH
                
                if task.priority != old_priority:
                    priority_changes.append({
                        'task_id': task_id,
                        'old': old_priority.name,
                        'new': task.priority.name,
                        'margin': margin,
                        'satisfaction': satisfaction * 100
                    })
                break
    
    # 打印优先级变化
    if priority_changes:
        print("\n优先级调整:")
        print("-" * 80)
        print(f"{'任务ID':<10} {'原优先级':<12} {'新优先级':<12} {'延迟余量':<12} {'满足率':<10}")
        print("-" * 80)
        for change in priority_changes:
            print(f"{change['task_id']:<10} {change['old']:<12} {change['new']:<12} "
                  f"{change['margin']:<12.1f} {change['satisfaction']:<10.1f}%")
    else:
        print("  无需调整优先级")
    
    return optimized_tasks


def evaluate_optimized_solution(optimized_tasks: List, time_window: float = 1000.0):
    """评估优化后的方案"""
    print("\n评估优化方案...")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 注册优化后的任务
    for task in optimized_tasks:
        launcher.register_task(task)
    
    # 执行调度
    plan = launcher.create_launch_plan(time_window, "eager")
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    executor.segment_mode = True
    
    stats = executor.execute_plan(plan, time_window)
    
    # 评估性能
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    overall_metrics = evaluator.evaluate(time_window, plan.events)
    
    return evaluator, tracer


def main():
    """主函数"""
    print("=" * 100)
    print("优先级提升解决方案")
    print("=" * 100)
    
    # 1. 创建任务
    tasks = create_real_tasks()
    
    # 2. 分析基准性能
    print("\n步骤1: 分析基准性能")
    print("-" * 100)
    baseline_performance = analyze_task_performance(tasks)
    
    # 打印问题任务
    print("\n问题任务识别:")
    problem_tasks = [(tid, data) for tid, data in baseline_performance.items() if data['needs_boost']]
    if problem_tasks:
        for task_id, data in problem_tasks:
            print(f"  {task_id}: 满足率={data['satisfaction_rate']*100:.1f}%, "
                  f"延迟余量={data['margin']:.1f}ms, 当前优先级={data['priority'].name}")
    else:
        print("  无问题任务")
    
    # 3. 优化优先级
    print("\n步骤2: 优化任务优先级")
    print("-" * 100)
    optimized_tasks = optimize_task_priorities(tasks, baseline_performance)
    
    # 4. 评估优化效果
    print("\n步骤3: 评估优化效果")
    print("-" * 100)
    evaluator, tracer = evaluate_optimized_solution(optimized_tasks)
    
    # 5. 对比结果
    print("\n" + "=" * 100)
    print("优化效果对比:")
    print("=" * 100)
    print(f"{'任务ID':<10} {'任务名':<15} {'原优先级':<12} {'新优先级':<12} "
          f"{'基准满足率':<15} {'优化满足率':<15} {'改进':<10}")
    print("-" * 100)
    
    improved_count = 0
    for task_id, baseline_data in baseline_performance.items():
        if task_id in evaluator.task_metrics:
            metrics = evaluator.task_metrics[task_id]
            task = evaluator.tasks[task_id]
            
            baseline_rate = baseline_data['satisfaction_rate'] * 100
            optimized_rate = metrics.latency_satisfaction_rate * 100
            improvement = optimized_rate - baseline_rate
            
            # 找到原始优先级
            original_priority = baseline_data['priority']
            current_priority = task.priority
            
            if improvement > 0:
                improved_count += 1
                status = "✓"
            else:
                status = ""
            
            print(f"{task_id:<10} {task.name:<15} {original_priority.name:<12} "
                  f"{current_priority.name:<12} {baseline_rate:<15.1f}% "
                  f"{optimized_rate:<15.1f}% {improvement:+10.1f}% {status}")
    
    print(f"\n总结: {improved_count} 个任务得到改进")
    
    # 6. 特别关注T7
    if 'T7' in evaluator.task_metrics:
        print("\n特别关注 T7 (tk_search):")
        print(f"  基准满足率: {baseline_performance['T7']['satisfaction_rate']*100:.1f}%")
        print(f"  优化满足率: {evaluator.task_metrics['T7'].latency_satisfaction_rate*100:.1f}%")
        print(f"  改进: {(evaluator.task_metrics['T7'].latency_satisfaction_rate - baseline_performance['T7']['satisfaction_rate'])*100:+.1f}%")
    
    # 7. 可视化
    visualizer = ScheduleVisualizer(tracer)
    print("\n生成可视化...")
    
    png_filename = "priority_optimized_scheduling.png"
    visualizer.plot_resource_timeline(png_filename)
    print(f"✓ 生成甘特图: {png_filename}")
    
    print("\n" + "=" * 100)
    print("💡 结论:")
    print("=" * 100)
    print("1. 通过分析基准性能，识别延迟紧张的任务")
    print("2. 根据延迟余量和满足率，智能调整任务优先级")
    print("3. 优先级提升能有效改善边界任务的延迟满足率")
    print("4. 这是一种简单有效的静态优化方法")


if __name__ == "__main__":
    main()
