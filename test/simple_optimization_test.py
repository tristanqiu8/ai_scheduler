#!/usr/bin/env python3
"""
简化的优化测试 - 验证基本功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ResourceType, TaskPriority, NNTask,
    ResourceQueueManager, ScheduleTracer,
    TaskLauncher, ScheduleExecutor,
    PerformanceEvaluator
)


def test_simple_optimization():
    """测试简单的优化场景"""
    print("="*80)
    print("简化优化测试")
    print("="*80)
    
    # 1. 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 2. 创建简单任务
    # 高优先级任务
    task1 = NNTask("HIGH_1", "高优先级", priority=TaskPriority.HIGH)
    task1.add_segment(ResourceType.NPU, {60: 10.0}, "compute")
    task1.set_performance_requirements(fps=10, latency=100)
    
    # 低优先级任务
    task2 = NNTask("LOW_1", "低优先级", priority=TaskPriority.LOW)
    task2.add_segment(ResourceType.DSP, {40: 15.0}, "process")
    task2.set_performance_requirements(fps=5, latency=200)
    
    launcher.register_task(task1)
    launcher.register_task(task2)
    
    print("\n任务配置:")
    for task in [task1, task2]:
        print(f"  {task.task_id}: {task.name} ({task.priority.name})")
    
    # 3. 执行基线
    time_window = 100.0
    plan = launcher.create_launch_plan(time_window, "eager")
    
    print(f"\n发射计划: {len(plan.events)} 个事件")
    for event in plan.events[:5]:
        print(f"  {event.time:>6.1f}ms: {event.task_id}#{event.instance_id}")
    
    # 4. 执行
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    stats = executor.execute_plan(plan, time_window)
    
    print(f"\n执行结果:")
    print(f"  完成实例: {stats['completed_instances']}/{stats['total_instances']}")
    print(f"  执行段数: {stats['total_segments_executed']}")
    
    # 5. 评估性能
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    metrics = evaluator.evaluate(time_window, plan.events)
    
    print(f"\n性能指标:")
    print(f"  空闲时间: {metrics.idle_time:.1f}ms ({metrics.idle_time_ratio:.1f}%)")
    print(f"  FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
    
    # 6. 详细检查资源利用率
    print(f"\n资源利用率详情:")
    
    # 手动计算资源利用率
    resource_busy_time = {res_id: 0.0 for res_id in queue_manager.resource_queues.keys()}
    
    for execution in tracer.executions:
        if execution.resource_id in resource_busy_time:
            duration = execution.end_time - execution.start_time
            resource_busy_time[execution.resource_id] += duration
    
    for res_id, busy_time in resource_busy_time.items():
        utilization = (busy_time / time_window) * 100.0
        res_type = queue_manager.resource_queues[res_id].resource_type
        print(f"  {res_id} ({res_type.value}): {utilization:.1f}% (忙碌 {busy_time:.1f}ms)")
    
    # 7. 检查评估器的资源指标
    print(f"\n评估器资源指标:")
    if hasattr(evaluator, 'resource_metrics'):
        for res_id, res_metrics in evaluator.resource_metrics.items():
            print(f"  {res_id}: 利用率={res_metrics.utilization_rate:.1f}%")
    else:
        print("  (resource_metrics 不存在)")
    
    # 8. 显示任务指标
    print(f"\n任务性能:")
    for task_id, task_metrics in evaluator.task_metrics.items():
        print(f"  {task_id}:")
        print(f"    - 实例数: {task_metrics.instance_count}")
        print(f"    - 实际FPS: {task_metrics.achieved_fps:.1f}")
        print(f"    - 平均延迟: {task_metrics.avg_latency:.1f}ms")
    
    # 9. 验证空闲时间计算
    if tracer.executions:
        last_execution_time = max(exec.end_time for exec in tracer.executions)
        calculated_idle = time_window - last_execution_time
        print(f"\n空闲时间验证:")
        print(f"  最后执行时间: {last_execution_time:.1f}ms")
        print(f"  计算的空闲时间: {calculated_idle:.1f}ms")
        print(f"  报告的空闲时间: {metrics.idle_time:.1f}ms")


if __name__ == "__main__":
    test_simple_optimization()
