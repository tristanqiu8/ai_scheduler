#!/usr/bin/env python3
"""
测试真实任务在段级模式下的优化效果
使用 FORCED_SEGMENTATION 策略强制T2和T3分段
确保所有任务都能达到FPS要求
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enhanced_launcher import EnhancedTaskLauncher  # 使用增强版本
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority, SegmentationStrategy
from core.evaluator import PerformanceEvaluator
from core.models import SubSegment
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
import copy
import numpy as np


def calculate_system_utilization(tracer, window_size):
    """计算系统利用率（至少有一个硬件单元忙碌的时间比例）"""
    busy_intervals = []
    
    # 收集所有执行时间段
    for exec in tracer.executions:
        if exec.start_time is not None and exec.end_time is not None:
            busy_intervals.append((exec.start_time, exec.end_time))
    
    if not busy_intervals:
        return 0.0
    
    # 合并重叠的时间段
    busy_intervals.sort()
    merged_intervals = []
    
    for start, end in busy_intervals:
        if merged_intervals and start <= merged_intervals[-1][1]:
            # 重叠，扩展最后一个区间
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
        else:
            # 不重叠，添加新区间
            merged_intervals.append((start, end))
    
    # 计算总忙碌时间
    total_busy_time = sum(end - start for start, end in merged_intervals)
    
    return (total_busy_time / window_size) * 100.0


def prepare_tasks_with_segmentation():
    """准备任务并设置T2和T3为强制分段"""
    tasks = create_real_tasks()
    
    # T2 (YoloV8nBig) - 设置为强制分段
    t2 = tasks[1]
    t2.segmentation_strategy = SegmentationStrategy.FORCED_SEGMENTATION
    
    # T3 (Lpr) - 设置为强制分段  
    t3 = tasks[2]
    t3.segmentation_strategy = SegmentationStrategy.FORCED_SEGMENTATION
    
    return tasks


def analyze_segmented_tasks():
    """分析分段后的任务特征"""
    print("=== 分段策略分析 ===\n")
    
    tasks = prepare_tasks_with_segmentation()
    
    print(f"{'任务ID':<10} {'任务名称':<20} {'分段策略':<25} {'原段数':>8} {'子段数':>8} {'FPS要求':>8}")
    print("-" * 85)
    
    for task in tasks[:8]:  # 显示所有8个任务
        sub_segments = task.apply_segmentation()
        seg_count = len(sub_segments) if sub_segments else len(task.segments)
        
        print(f"{task.task_id:<10} {task.name:<20} {task.segmentation_strategy.value:<25} "
              f"{len(task.segments):>8} {seg_count:>8} {task.fps_requirement:>8}")
    
    print("\n关键变化：")
    print("- T2 (YoloV8nBig): 使用 FORCED_SEGMENTATION，NPU段被切分")
    print("- T3 (Lpr): 使用 FORCED_SEGMENTATION，NPU段被切分")
    print("- 其他任务保持 NO_SEGMENTATION 策略")


def test_single_npu_dsp_baseline():
    """测试单NPU+单DSP的基准性能 - 确保所有任务都执行"""
    print("\n\n=== 基准测试：单NPU + 单DSP (所有任务) ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备分段后的任务
    tasks = prepare_tasks_with_segmentation()
    
    # 打印所有任务信息
    print("注册的任务:")
    for i, task in enumerate(tasks):
        print(f"  {i}. {task.task_id} ({task.name}): FPS={task.fps_requirement}, "
              f"Priority={task.priority.name}, Segments={len(task.segments)}")
    
    results = {}
    tracers = {}
    
    # 测试两种模式
    for mode_name, segment_mode in [("传统模式", False), ("段级模式", True)]:
        print(f"\n{mode_name}:")
        
        tracer = ScheduleTracer(queue_manager)
        launcher = EnhancedTaskLauncher(queue_manager, tracer)
        
        # 注册所有任务，确保每个任务都能执行
        for task in tasks:
            launcher.register_task(task)
        
        # 执行
        duration = 200.0
        plan = launcher.create_launch_plan(duration, "eager")
        
        # 验证发射计划包含所有任务
        launched_tasks = set()
        for event in plan.events:
            # event.instance_id 是整数，event.task_id 是任务ID
            launched_tasks.add(event.task_id)
        
        print(f"  发射的任务: {sorted(launched_tasks)}")
        if len(launched_tasks) < len(tasks):
            print(f"  ⚠️ 警告: 只发射了{len(launched_tasks)}/{len(tasks)}个任务")
        
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=segment_mode)
        
        # 分析执行时间线
        trace_stats = tracer.get_statistics()
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(duration, plan.events)
        
        # 计算系统利用率
        system_util = calculate_system_utilization(tracer, duration)
        
        results[mode_name] = {
            'stats': stats,
            'metrics': metrics,
            'utilization': tracer.get_resource_utilization(),
            'system_utilization': system_util,
            'trace_stats': trace_stats,
            'evaluator': evaluator
        }
        tracers[mode_name] = tracer
        
        print(f"  完成实例: {stats['completed_instances']}")
        print(f"  执行段数: {stats['total_segments_executed']}")
        print(f"  NPU利用率: {results[mode_name]['utilization'].get('NPU_0', 0):.1f}%")
        print(f"  DSP利用率: {results[mode_name]['utilization'].get('DSP_0', 0):.1f}%")
        print(f"  System利用率: {system_util:.1f}%")
        print(f"  平均等待时间: {trace_stats['average_wait_time']:.2f}ms")
        print(f"  FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
        
        # 添加各任务的FPS信息
        print("\n  各任务FPS达成情况:")
        for task_id, task_metrics in evaluator.task_metrics.items():
            task = launcher.tasks.get(task_id)
            if task:
                achieved_fps = task_metrics.achieved_fps
                required_fps = task.fps_requirement
                satisfaction = (achieved_fps / required_fps * 100) if required_fps > 0 else 0
                completed = task_metrics.instance_count
                expected = int(duration / (1000.0 / required_fps))
                
                print(f"    {task_id} ({task.name}): "
                      f"要求={required_fps} FPS, "
                      f"达成={achieved_fps:.1f} FPS, "
                      f"满足率={satisfaction:.1f}%, "
                      f"完成={completed}/{expected}实例")
    
    # 计算提升
    print("\n性能提升分析:")
    trad = results['传统模式']
    seg = results['段级模式']
    
    improvements = {
        'NPU利用率': seg['utilization'].get('NPU_0', 0) - trad['utilization'].get('NPU_0', 0),
        'DSP利用率': seg['utilization'].get('DSP_0', 0) - trad['utilization'].get('DSP_0', 0),
        'System利用率': seg['system_utilization'] - trad['system_utilization'],
        '完成实例': ((seg['stats']['completed_instances'] - trad['stats']['completed_instances']) 
                    / trad['stats']['completed_instances'] * 100) if trad['stats']['completed_instances'] > 0 else 0,
        '等待时间': ((trad['trace_stats']['average_wait_time'] - seg['trace_stats']['average_wait_time']) 
                    / trad['trace_stats']['average_wait_time'] * 100) if trad['trace_stats']['average_wait_time'] > 0 else 0
    }
    
    for metric, value in improvements.items():
        if metric == '等待时间':
            print(f"  {metric}: {value:+.1f}% (减少)")
        else:
            print(f"  {metric}: {value:+.1f}%")
    
    return results, tracers


def generate_visualization():
    """生成优化前后的可视化对比"""
    print("\n\n=== 生成可视化 ===\n")
    
    # 重新执行以生成可视化
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tasks = prepare_tasks_with_segmentation()
    
    for mode_name, segment_mode in [("segment", True)]:
        tracer = ScheduleTracer(queue_manager)
        launcher = EnhancedTaskLauncher(queue_manager, tracer)
        
        # 注册所有任务
        for task in tasks:
            launcher.register_task(task)
        
        duration = 200.0
        plan = launcher.create_launch_plan(duration, "eager")
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=segment_mode)
        
        # 创建可视化
        visualizer = ScheduleVisualizer(tracer)
        
        # 打印甘特图
        print(f"\n{mode_name.upper()} 模式执行时间线:\n")
        
        # 确保显示完整的200ms时间线
        original_end_time = tracer.end_time
        if tracer.end_time is None or tracer.end_time < duration:
            tracer.end_time = duration
        
        visualizer.print_gantt_chart(width=100)
        
        # 恢复原始end_time
        tracer.end_time = original_end_time
        
        # 显示统计信息
        trace_stats = tracer.get_statistics()
        system_util = calculate_system_utilization(tracer, duration)
        
        print(f"\n统计信息:")
        print(f"  执行数: {stats['total_segments_executed']}")
        print(f"  时间跨度: {trace_stats['time_span']:.1f}ms")
        print(f"  资源利用率: NPU={tracer.get_resource_utilization().get('NPU_0', 0):.1f}%, "
              f"DSP={tracer.get_resource_utilization().get('DSP_0', 0):.1f}%, "
              f"System={system_util:.1f}%")
        
        # 统计任务执行次数
        task_counts = {}
        for exec in tracer.executions:
            if '#' in exec.task_id:
                base_task_id = exec.task_id.split('#')[0]
                if '_seg0' in exec.task_id or '_seg' not in exec.task_id:
                    task_counts[base_task_id] = task_counts.get(base_task_id, 0) + 1
        
        print(f"\n任务执行次数:")
        for task_id in sorted(task_counts.keys()):
            task = next((t for t in tasks if t.task_id == task_id), None)
            if task:
                expected = int(duration / (1000.0 / task.fps_requirement))
                actual = task_counts.get(task_id, 0)
                status = "✓" if actual >= expected else "✗"
                print(f"  {task_id}: {actual}/{expected} (FPS要求: {task.fps_requirement}) {status}")
        
        # 生成图片
        original_start_time = tracer.start_time
        original_end_time = tracer.end_time
        
        if tracer.start_time is None or tracer.start_time > 0:
            tracer.start_time = 0
        if tracer.end_time is None or tracer.end_time < duration:
            tracer.end_time = duration
            
        visualizer.plot_resource_timeline(f"segmented_tasks_{mode_name}.png", figsize=(16, 6), dpi=100)
        
        # 恢复原始时间
        tracer.start_time = original_start_time
        tracer.end_time = original_end_time
        
        # 保存追踪数据
        visualizer.export_chrome_tracing(f"segmented_tasks_{mode_name}.json")
        
        print(f"\n生成文件:")
        print(f"  - segmented_tasks_{mode_name}.png")
        print(f"  - segmented_tasks_{mode_name}.json")


def check_task_fps_requirements():
    """检查并报告未满足FPS要求的任务"""
    print("\n\n=== FPS要求满足情况分析 ===\n")
    
    tasks = create_real_tasks()
    duration = 200.0
    
    print("任务FPS要求:")
    for task in tasks:
        expected_instances = int(duration / (1000.0 / task.fps_requirement))
        print(f"  {task.task_id} ({task.name}): {task.fps_requirement} FPS → {expected_instances} 实例/200ms")
    
    print("\n分析T7和T8执行不足的原因:")
    print("1. 资源竞争: 单NPU+单DSP的资源有限")
    print("2. 优先级影响: T7和T8优先级为LOW，容易被高优先级任务抢占")
    print("3. 调度策略: eager策略可能导致资源利用不均衡")
    
    print("\n解决方案:")
    print("1. 使用段级调度提高资源利用率")
    print("2. 优化任务优先级分配")
    print("3. 使用更智能的发射策略（如遗传算法优化）")


def main():
    """主测试函数"""
    print("🚀 真实任务段级优化测试（使用 FORCED_SEGMENTATION）\n")
    print("系统配置：单NPU (60 GFLOPS) + 单DSP (40 GFLOPS)")
    print("=" * 115)
    
    # 1. 分析分段策略
    analyze_segmented_tasks()
    
    # 2. 基准测试 - 确保所有任务都注册和执行
    baseline_results, tracers = test_single_npu_dsp_baseline()
    
    # 3. 检查FPS要求满足情况
    check_task_fps_requirements()
    
    # 4. 生成可视化
    generate_visualization()
    
    # 总结
    print("\n\n" + "=" * 115)
    print("📊 优化效果总结")
    print("=" * 115)
    
    print("\n关键发现：")
    print("1. FORCED_SEGMENTATION 策略让T2和T3的NPU段被有效切分")
    print("2. 段级模式充分利用了分段带来的调度灵活性")
    print("3. System利用率展示了整体系统的繁忙程度")
    print("4. 低优先级任务（T7、T8）在资源受限时可能无法满足FPS要求")
    
    print("\n优化建议：")
    print("- 使用多资源（如2个NPU）来满足所有任务的FPS要求")
    print("- 调整任务优先级或使用更智能的调度策略")
    print("- 考虑任务的时间特性，优化发射时机")


if __name__ == "__main__":
    main()
