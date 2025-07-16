#!/usr/bin/env python3
"""
测试 hybrid_task 场景的调度优化
配置：单DSP + 单NPU，带宽各40GB/s
重点关注FPS达标和延迟要求
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enhanced_launcher import EnhancedTaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority, SegmentationStrategy
from core.evaluator import PerformanceEvaluator
from scenario.hybrid_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
import numpy as np


def print_task_requirements(tasks):
    """打印任务要求概览"""
    print("\n📋 任务要求概览:")
    print("=" * 100)
    print(f"{'ID':<6} {'名称':<15} {'优先级':<10} {'FPS要求':<10} {'延迟要求(ms)':<15} {'分段策略':<20}")
    print("-" * 100)
    
    for task in tasks:
        print(f"{task.task_id:<6} {task.name:<15} {task.priority.name:<10} "
              f"{task.fps_requirement:<10.0f} {task.latency_requirement:<15.1f} "
              f"{task.segmentation_strategy.value:<20}")


def analyze_task_demands(tasks, time_window=1000.0):
    """分析任务的资源需求"""
    print("\n📊 资源需求分析 (带宽=40GB/s, 时间窗口=1000ms):")
    print("=" * 100)
    
    total_npu_demand = 0.0
    total_dsp_demand = 0.0
    
    for task in tasks:
        # 计算在时间窗口内需要的实例数
        instances_needed = task.fps_requirement * (time_window / 1000.0)
        
        # 获取分段后的执行时间
        segments = task.apply_segmentation()
        if not segments:
            segments = task.segments
        
        npu_time_per_instance = 0.0
        dsp_time_per_instance = 0.0
        
        for seg in segments:
            duration = seg.get_duration(40.0)  # 40GB/s带宽
            if seg.resource_type == ResourceType.NPU:
                npu_time_per_instance += duration
            elif seg.resource_type == ResourceType.DSP:
                dsp_time_per_instance += duration
        
        npu_demand = npu_time_per_instance * instances_needed
        dsp_demand = dsp_time_per_instance * instances_needed
        
        total_npu_demand += npu_demand
        total_dsp_demand += dsp_demand
        
        if npu_demand > 0 or dsp_demand > 0:
            print(f"{task.task_id} ({task.name}): "
                  f"NPU={npu_demand:.1f}ms, DSP={dsp_demand:.1f}ms "
                  f"({instances_needed:.1f}实例)")
    
    print(f"\n总需求: NPU={total_npu_demand:.1f}ms, DSP={total_dsp_demand:.1f}ms")
    print(f"理论利用率: NPU={total_npu_demand/10:.1f}%, DSP={total_dsp_demand/10:.1f}%")
    
    if total_npu_demand > time_window or total_dsp_demand > time_window:
        print("\n⚠️ 警告: 资源需求超过可用时间，部分任务可能无法满足FPS要求！")


def test_scheduling_modes(time_window=1000.0):
    """测试不同的调度模式"""
    print(f"\n\n🔬 调度模式对比测试 (时间窗口: {time_window}ms)")
    print("=" * 100)
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备任务
    tasks = create_real_tasks()
    
    results = {}
    
    # 测试两种模式
    modes = [
        ("传统模式", False),
        ("段级模式", True)
    ]
    
    for mode_name, segment_mode in modes:
        print(f"\n\n{'='*50}")
        print(f"执行 {mode_name} (segment_mode={segment_mode})")
        print('='*50)
        
        # 重置环境
        queue_manager = ResourceQueueManager()
        queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
        queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
        
        tracer = ScheduleTracer(queue_manager)
        
        if segment_mode:
            launcher = EnhancedTaskLauncher(queue_manager, tracer)
        else:
            launcher = TaskLauncher(queue_manager, tracer)
        
        # 注册任务
        for task in tasks:
            launcher.register_task(task)
        
        # 创建发射计划
        plan = launcher.create_launch_plan(time_window, "eager")
        
        # 执行
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, time_window, segment_mode=segment_mode)
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(time_window, plan.events)
        
        results[mode_name] = {
            'stats': stats,
            'metrics': metrics,
            'tracer': tracer,
            'evaluator': evaluator
        }
        
        # 打印结果摘要
        print(f"\n{mode_name}结果:")
        print(f"  完成实例: {stats.get('completed_instances', 0)}/{stats.get('total_instances', 0)}")
        # 执行段数统计
        if 'total_segments_executed' in stats:
            print(f"  执行段数: {stats['total_segments_executed']}")
        print(f"  平均延迟: {metrics.avg_latency:.1f}ms")
        print(f"  最大延迟: {metrics.max_latency:.1f}ms")
        print(f"  NPU利用率: {metrics.avg_npu_utilization:.1f}%")
        print(f"  DSP利用率: {metrics.avg_dsp_utilization:.1f}%")
    
    return results


def analyze_latency_performance(results):
    """分析延迟性能"""
    print("\n\n📈 延迟性能分析")
    print("=" * 100)
    
    for mode_name, data in results.items():
        evaluator = data['evaluator']
        
        print(f"\n{mode_name}:")
        print(f"{'任务ID':<8} {'任务名':<15} {'FPS要求':<10} {'实际FPS':<10} "
              f"{'延迟要求':<12} {'平均延迟':<12} {'最大延迟':<12} {'满足率':<10}")
        print("-" * 100)
        
        for task_id, metrics in evaluator.task_metrics.items():
            # 获取对应的任务对象
            task = next((t for t in evaluator.tasks.values() if t.task_id == task_id), None)
            if not task:
                continue
            
            fps_status = "✓" if metrics.fps_satisfaction else "✗"
            latency_status = "✓" if metrics.latency_satisfaction_rate > 0.9 else "✗"
            
            print(f"{task_id:<8} {task.name:<15} {metrics.fps_requirement:<10.0f} "
                  f"{metrics.achieved_fps:<9.1f}{fps_status} "
                  f"{metrics.latency_requirement:<12.1f} {metrics.avg_latency:<12.1f} "
                  f"{metrics.max_latency:<12.1f} {metrics.latency_satisfaction_rate:<9.1%}{latency_status}")


def print_detailed_task_analysis(results, task_id):
    """打印特定任务的详细分析"""
    print(f"\n\n🔍 任务 {task_id} 详细分析")
    print("=" * 80)
    
    for mode_name, data in results.items():
        evaluator = data['evaluator']
        tracer = data['tracer']
        
        if task_id not in evaluator.task_metrics:
            continue
            
        metrics = evaluator.task_metrics[task_id]
        task = next((t for t in evaluator.tasks.values() if t.task_id == task_id), None)
        if not task:
            continue
        
        print(f"\n{mode_name}:")
        print(f"  任务: {task.name}")
        print(f"  实例数: {metrics.instance_count}")
        print(f"  FPS: 要求={metrics.fps_requirement}, 实际={metrics.achieved_fps:.1f}")
        print(f"  延迟: 要求={metrics.latency_requirement:.1f}ms")
        
        if metrics.latencies:
            print(f"    平均={metrics.avg_latency:.1f}ms")
            print(f"    最大={metrics.max_latency:.1f}ms")
            print(f"    最小={min(metrics.latencies):.1f}ms")
            print(f"    标准差={np.std(metrics.latencies):.1f}ms")
            
            # 显示延迟分布
            print(f"  延迟分布:")
            bins = [0, 25, 50, 75, 100, 150, 200, float('inf')]
            bin_labels = ['0-25', '25-50', '50-75', '75-100', '100-150', '150-200', '>200']
            bin_counts = [0] * (len(bins) - 1)
            
            for latency in metrics.latencies:
                for i in range(len(bins) - 1):
                    if bins[i] <= latency < bins[i+1]:
                        bin_counts[i] += 1
                        break
            
            for label, count in zip(bin_labels, bin_counts):
                if count > 0:
                    percentage = (count / len(metrics.latencies)) * 100
                    print(f"    {label}ms: {count} ({percentage:.1f}%)")


def visualize_execution(results, time_range=(0, 200)):
    """可视化执行时间线"""
    print(f"\n\n📊 执行时间线可视化")
    print("=" * 100)
    
    for mode_name, data in results.items():
        tracer = data['tracer']
        visualizer = ScheduleVisualizer(tracer)
        
        print(f"\n{mode_name}:")
        # 直接使用默认的甘特图显示
        visualizer.print_gantt_chart(width=80)
        
        # 生成图片
        png_filename = f"hybrid_task_{mode_name.replace(' ', '_')}.png"
        visualizer.plot_resource_timeline(png_filename)
        print(f"  ✓ 生成甘特图: {png_filename}")
        
        # 生成Chrome Tracing JSON文件
        json_filename = f"hybrid_task_{mode_name.replace(' ', '_')}_trace.json"
        visualizer.export_chrome_tracing(json_filename)
        print(f"  ✓ 生成Chrome Tracing文件: {json_filename}")
    
    print("\n💡 提示：在Chrome浏览器中打开 chrome://tracing 并加载JSON文件查看详细时间线")


def main():
    """主函数"""
    print("=" * 100)
    print("Hybrid Task 调度优化测试")
    print("配置: 单DSP + 单NPU, 带宽各40GB/s")
    print("=" * 100)
    
    # 1. 创建任务并分析
    tasks = create_real_tasks()
    print_task_requirements(tasks)
    
    # 2. 分析资源需求
    analyze_task_demands(tasks)
    
    # 3. 执行调度测试
    results = test_scheduling_modes(time_window=1000.0)
    
    # 4. 分析延迟性能
    analyze_latency_performance(results)
    
    # 5. 分析关键任务的详细性能
    critical_tasks = ["T11", "T12", "T14"]  # Stereo4x 和 Skywater 系列
    for task_id in critical_tasks:
        print_detailed_task_analysis(results, task_id)
    
    # 6. 可视化前200ms的执行
    visualize_execution(results, time_range=(0, 200))
    
    # 7. 总结
    print("\n\n" + "=" * 100)
    print("📊 优化效果总结")
    print("=" * 100)
    
    # 计算改进
    old_metrics = results['传统模式']['metrics']
    new_metrics = results['段级模式']['metrics']
    
    print("\n性能改进:")
    print(f"  平均延迟: {old_metrics.avg_latency:.1f}ms → {new_metrics.avg_latency:.1f}ms "
          f"(改善 {((old_metrics.avg_latency - new_metrics.avg_latency) / old_metrics.avg_latency * 100):.1f}%)")
    print(f"  最大延迟: {old_metrics.max_latency:.1f}ms → {new_metrics.max_latency:.1f}ms")
    print(f"  NPU利用率: {old_metrics.avg_npu_utilization:.1f}% → {new_metrics.avg_npu_utilization:.1f}%")
    print(f"  DSP利用率: {old_metrics.avg_dsp_utilization:.1f}% → {new_metrics.avg_dsp_utilization:.1f}%")
    
    # 统计延迟要求满足情况
    trad_satisfied = 0
    seg_satisfied = 0
    
    for task_id in results['传统模式']['evaluator'].task_metrics:
        if results['传统模式']['evaluator'].task_metrics[task_id].latency_satisfaction_rate > 0.9:
            trad_satisfied += 1
        if results['段级模式']['evaluator'].task_metrics[task_id].latency_satisfaction_rate > 0.9:
            seg_satisfied += 1
    
    total_tasks = len(results['传统模式']['evaluator'].task_metrics)
    print(f"\n延迟要求满足情况:")
    print(f"  传统模式: {trad_satisfied}/{total_tasks} 任务满足延迟要求")
    print(f"  段级模式: {seg_satisfied}/{total_tasks} 任务满足延迟要求")
    
    print("\n关键发现:")
    print("1. 段级调度能够显著改善任务延迟，特别是对于有严格延迟要求的任务")
    print("2. 通过更灵活的调度，可以在相同资源下满足更多任务的性能要求")
    print("3. 某些高负载任务可能仍需要额外资源才能完全满足要求")


if __name__ == "__main__":
    main()
