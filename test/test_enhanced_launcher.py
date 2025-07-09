#!/usr/bin/env python3
"""
测试增强发射器的效果，特别是对依赖任务的处理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enhanced_launcher import EnhancedTaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority
from core.evaluator import PerformanceEvaluator
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer


def compare_launchers():
    """对比原始发射器和增强发射器"""
    print("=" * 80)
    print("对比原始发射器 vs 增强发射器")
    print("=" * 80)
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 加载任务
    tasks = create_real_tasks()
    
    # 测试两种发射器
    results = {}
    
    for launcher_type in ["原始", "增强"]:
        print(f"\n{'='*40}")
        print(f"测试{launcher_type}发射器")
        print(f"{'='*40}")
        
        tracer = ScheduleTracer(queue_manager)
        
        if launcher_type == "原始":
            launcher = TaskLauncher(queue_manager, tracer)
        else:
            launcher = EnhancedTaskLauncher(queue_manager, tracer)
        
        # 注册所有任务
        for task in tasks:
            launcher.register_task(task)
        
        # 创建发射计划
        duration = 200.0
        plan = launcher.create_launch_plan(duration, "eager")
        
        # 分析发射计划
        print(f"\n发射计划分析:")
        task_launches = {}
        for event in plan.events:
            if event.task_id not in task_launches:
                task_launches[event.task_id] = []
            task_launches[event.task_id].append(event.time)
        
        # 特别关注T7和T8
        for task_id in ["T1", "T7", "T8"]:
            if task_id in task_launches:
                launches = task_launches[task_id]
                task = next(t for t in tasks if t.task_id == task_id)
                expected = int(duration / (1000.0 / task.fps_requirement))
                
                print(f"\n{task_id} ({task.name}):")
                print(f"  依赖: {task.dependencies}")
                print(f"  期望实例数: {expected}")
                print(f"  实际发射数: {len(launches)}")
                print(f"  发射时间: {[f'{t:.1f}ms' for t in launches[:5]]}")
                if len(launches) > 5:
                    print(f"  ... 还有{len(launches)-5}个发射")
            else:
                print(f"\n{task_id}: 未发射！")
        
        # 执行计划
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=True)
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(duration, plan.events)
        
        results[launcher_type] = {
            'plan': plan,
            'stats': stats,
            'metrics': metrics,
            'evaluator': evaluator
        }
        
        print(f"\n执行结果:")
        print(f"  完成实例: {stats['completed_instances']}")
        print(f"  FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
        
        # 检查T7和T8的执行情况
        for task_id in ["T7", "T8"]:
            if task_id in evaluator.task_metrics:
                tm = evaluator.task_metrics[task_id]
                print(f"\n  {task_id}:")
                print(f"    完成实例: {tm.instance_count}")
                print(f"    达成FPS: {tm.achieved_fps:.1f}")
                print(f"    满足率: {(tm.achieved_fps/25.0)*100:.1f}%")
    
    # 对比结果
    print(f"\n{'='*80}")
    print("对比总结")
    print(f"{'='*80}")
    
    orig = results["原始"]
    enh = results["增强"]
    
    print(f"\nFPS满足率提升: {enh['metrics'].fps_satisfaction_rate - orig['metrics'].fps_satisfaction_rate:.1f}%")
    print(f"完成实例增加: {enh['stats']['completed_instances'] - orig['stats']['completed_instances']}")
    
    # 特别检查T7和T8
    print(f"\n依赖任务改进:")
    for task_id in ["T7", "T8"]:
        orig_tm = orig['evaluator'].task_metrics.get(task_id)
        enh_tm = enh['evaluator'].task_metrics.get(task_id)
        
        if orig_tm and enh_tm:
            print(f"  {task_id}: {orig_tm.instance_count} → {enh_tm.instance_count} 实例 "
                  f"(+{enh_tm.instance_count - orig_tm.instance_count})")


def visualize_dependency_handling():
    """可视化依赖任务的处理"""
    print("\n\n" + "="*80)
    print("可视化依赖任务处理")
    print("="*80)
    
    # 创建简单场景：只有T1、T7、T8
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tasks = create_real_tasks()
    selected_tasks = [
        tasks[0],  # T1 (被依赖)
        tasks[6],  # T7 (依赖T1)
        tasks[7],  # T8 (依赖T1)
    ]
    
    # 使用增强发射器
    tracer = ScheduleTracer(queue_manager)
    launcher = EnhancedTaskLauncher(queue_manager, tracer)
    
    for task in selected_tasks:
        launcher.register_task(task)
    
    # 创建发射计划
    plan = launcher.create_launch_plan(100.0, "eager")
    
    print("\n发射计划时间线:")
    print("时间(ms)  任务#实例")
    print("-" * 30)
    
    for event in sorted(plan.events, key=lambda e: e.time):
        print(f"{event.time:>8.1f}  {event.task_id}#{event.instance_id}")
    
    # 执行并可视化
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    stats = executor.execute_plan(plan, 100.0, segment_mode=True)
    
    # 生成甘特图
    visualizer = ScheduleVisualizer(tracer)
    print("\n执行时间线:")
    visualizer.print_gantt_chart(width=80)
    
    # 保存可视化
    visualizer.plot_resource_timeline("dependency_handling.png", figsize=(12, 4))
    print("\n时间线图已保存到: dependency_handling.png")


if __name__ == "__main__":
    # 1. 对比两种发射器
    compare_launchers()
    
    # 2. 可视化依赖处理
    visualize_dependency_handling()
