#!/usr/bin/env python3
"""
最终测试相机任务性能 - 验证修复效果
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enhanced_launcher import EnhancedTaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority
from core.evaluator import PerformanceEvaluator
from scenario.camera_task import create_real_tasks
import io
import contextlib


def final_camera_test():
    """最终的相机任务性能测试"""
    
    print("=" * 100)
    print("Final Camera Task Performance Test (After frame-rate aware dependency fix)")
    print("=" * 100)
    
    # 创建任务
    tasks = create_real_tasks()
    
    print(f"\n[INFO] Task Summary:")
    total_fps = sum(task.fps_requirement for task in tasks)
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Total FPS requirement: {total_fps}")
    
    # 测试不同的优先级配置
    configs = {
        "Original": lambda t: t.priority,  # 保持原优先级
        "Smart": get_smart_priority  # 智能优先级
    }
    
    results = {}
    
    for config_name, priority_func in configs.items():
        print(f"\n[TEST] {config_name} Configuration:")
        print("-" * 80)
        
        # 应用优先级配置
        test_tasks = [create_task_copy(task) for task in tasks]
        for task in test_tasks:
            task.priority = priority_func(task)
        
        # 显示优先级配置
        print("Priority assignment:")
        for task in sorted(test_tasks, key=lambda x: x.task_id):
            deps = f" (deps: {','.join(task.dependencies)})" if task.dependencies else ""
            print(f"  {task.task_id}: {task.priority.name}{deps}")
        
        # 执行测试
        result = run_performance_test(test_tasks, config_name)
        results[config_name] = result
    
    # 对比结果
    print(f"\n\n[COMPARISON] Performance Comparison:")
    print("=" * 100)
    
    header = f"{'Metric':<25} {'Original':<15} {'Smart':<15} {'Improvement':<15}"
    print(header)
    print("-" * 100)
    
    metrics = [
        ("FPS Satisfaction", "fps_rate", "%"),
        ("Tasks Meeting FPS", "fps_count", ""),
        ("Latency Satisfaction", "latency_rate", "%"),
        ("NPU Utilization", "npu_util", "%"),
        ("DSP Utilization", "dsp_util", "%")
    ]
    
    for metric_name, key, unit in metrics:
        orig_val = results["Original"][key]
        smart_val = results["Smart"][key]
        
        if unit == "%":
            orig_str = f"{orig_val:.1%}"
            smart_str = f"{smart_val:.1%}"
            if orig_val > 0:
                improvement = f"{(smart_val - orig_val) / orig_val * 100:+.1f}%"
            else:
                improvement = "N/A"
        else:
            orig_str = f"{orig_val}"
            smart_str = f"{smart_val}"
            improvement = f"{smart_val - orig_val:+}"
        
        print(f"{metric_name:<25} {orig_str:<15} {smart_str:<15} {improvement:<15}")
    
    # 特别关注T2的表现
    print(f"\n[T2 FOCUS] T2 (FaceEhnsLite) Performance:")
    print("-" * 60)
    
    for config_name, result in results.items():
        t2_perf = result.get('t2_performance', {})
        if t2_perf:
            print(f"{config_name}:")
            print(f"  FPS: {t2_perf['fps']:.1f} / 32 ({t2_perf['fps']/32:.1%})")
            print(f"  Instances: {t2_perf['instances']}")
            print(f"  Latency: {t2_perf['latency']:.1f}ms")
    
    # 最终结论
    smart_fps_rate = results["Smart"]["fps_rate"]
    print(f"\n[CONCLUSION]")
    print("=" * 100)
    
    if smart_fps_rate >= 0.99:  # 99%以上
        print(f"SUCCESS: All tasks meeting FPS requirements! ({smart_fps_rate:.1%})")
        print("The frame-rate aware dependency fix successfully resolved T2's performance issue.")
    elif smart_fps_rate >= 0.90:  # 90%以上
        print(f"GOOD: Most tasks meeting FPS requirements ({smart_fps_rate:.1%})")
        print("Significant improvement achieved.")
    else:
        print(f"NEEDS WORK: Only {smart_fps_rate:.1%} of tasks meeting FPS requirements")
        print("Further optimization may be needed.")
    
    return results


def get_smart_priority(task):
    """智能优先级配置"""
    # 分析被依赖情况
    all_tasks = create_real_tasks()
    dependency_count = {}
    for t in all_tasks:
        dependency_count[t.task_id] = 0
    
    for t in all_tasks:
        for dep in t.dependencies:
            if dep in dependency_count:
                dependency_count[dep] += 1
    
    # 优先级规则
    if dependency_count[task.task_id] > 0:  # 被依赖的任务
        return TaskPriority.HIGH
    elif task.fps_requirement > 16:  # 高帧率任务
        return TaskPriority.NORMAL
    else:  # 低帧率任务
        return TaskPriority.LOW


def create_task_copy(task):
    """创建任务副本"""
    from copy import deepcopy
    return deepcopy(task)


def run_performance_test(tasks, config_name):
    """运行性能测试"""
    
    # 创建调度环境
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 120.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 120.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = EnhancedTaskLauncher(queue_manager, tracer)
    
    # 注册任务
    for task in tasks:
        launcher.register_task(task)
    
    # 执行调度（静默）
    time_window = 125.0
    plan = launcher.create_launch_plan(time_window, "balanced")
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        stats = executor.execute_plan(plan, time_window, segment_mode=True)
    
    # 评估性能
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    metrics = evaluator.evaluate(time_window, plan.events)
    
    # 统计满足率
    fps_satisfied = 0
    latency_satisfied = 0
    t2_performance = {}
    
    for task_id, task_metrics in evaluator.task_metrics.items():
        if task_metrics.fps_satisfaction:
            fps_satisfied += 1
        if task_metrics.latency_satisfaction_rate > 0.9:
            latency_satisfied += 1
        
        # 记录T2的特殊表现
        if task_id == "T2":
            t2_performance = {
                'fps': task_metrics.achieved_fps,
                'instances': task_metrics.instance_count,
                'latency': task_metrics.avg_latency
            }
    
    num_tasks = len(evaluator.task_metrics)
    fps_rate = fps_satisfied / num_tasks if num_tasks > 0 else 0
    latency_rate = latency_satisfied / num_tasks if num_tasks > 0 else 0
    
    print(f"  Results: FPS {fps_satisfied}/{num_tasks} ({fps_rate:.1%}), "
          f"Latency {latency_satisfied}/{num_tasks} ({latency_rate:.1%})")
    print(f"  Resource: NPU {metrics.avg_npu_utilization:.1f}%, "
          f"DSP {metrics.avg_dsp_utilization:.1f}%")
    
    return {
        'fps_rate': fps_rate,
        'latency_rate': latency_rate,
        'fps_count': fps_satisfied,
        'latency_count': latency_satisfied,
        'npu_util': metrics.avg_npu_utilization,
        'dsp_util': metrics.avg_dsp_utilization,
        't2_performance': t2_performance
    }


if __name__ == "__main__":
    results = final_camera_test()