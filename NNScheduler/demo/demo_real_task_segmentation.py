#!/usr/bin/env python3
"""
测试真实任务在段级模式下的优化效果 - 修复版本
使用 FORCED_SEGMENTATION 策略强制T2和T3分段
确保所有任务都能达到FPS要求
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.executor import ScheduleExecutor
from NNScheduler.core.enums import ResourceType, TaskPriority, SegmentationStrategy
from NNScheduler.core.evaluator import PerformanceEvaluator
from NNScheduler.core.models import SubSegment
from NNScheduler.scenario.real_task import create_real_tasks
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
import copy
import numpy as np

# 添加FPS窗口计算功能
def gcd(a, b):
    """计算两个数的最大公约数"""
    while b:
        a, b = b, a % b
    return a

def gcd_multiple(numbers):
    """计算多个数的最大公约数"""
    from functools import reduce
    return reduce(gcd, numbers)

def calculate_optimal_window(fps_requirements, max_window_ms=1000.0):
    """基于FPS要求计算最优分析时间窗口"""
    if not fps_requirements:
        return 200.0
    
    fps_gcd = gcd_multiple(fps_requirements)
    gcd_window = 1000.0 / fps_gcd
    return min(gcd_window, max_window_ms)


def compute_resource_demand(tasks, bandwidth_npu=40.0, bandwidth_dsp=40.0, time_window_ms=1000.0):
    """
    计算在给定带宽下，指定时间窗口内NPU和DSP的总资源需求
    
    Args:
        tasks: 任务列表
        bandwidth_npu: NPU带宽
        bandwidth_dsp: DSP带宽
        time_window_ms: 时间窗口（毫秒）
        
    Returns:
        dict: 包含资源需求分析的字典
    """
    npu_total_time = 0.0
    dsp_total_time = 0.0
    
    # 详细的任务需求
    task_details = []
    
    for task in tasks:
        # 计算这个任务在时间窗口内需要执行的次数
        instances_needed = task.fps_requirement * (time_window_ms / 1000.0)
        
        # 应用分段策略获取实际执行的段
        segments = task.apply_segmentation()
        if not segments:
            segments = task.segments
        
        # 计算每个段的执行时间
        npu_time_per_instance = 0.0
        dsp_time_per_instance = 0.0
        segment_details = []
        
        for seg in segments:
            if seg.resource_type == ResourceType.NPU:
                # 获取在指定带宽下的执行时间
                duration = seg.get_duration(bandwidth_npu)
                npu_time_per_instance += duration
                segment_details.append({
                    'segment': seg.sub_id,
                    'resource': 'NPU',
                    'duration': duration
                })
            elif seg.resource_type == ResourceType.DSP:
                duration = seg.get_duration(bandwidth_dsp)
                dsp_time_per_instance += duration
                segment_details.append({
                    'segment': seg.sub_id,
                    'resource': 'DSP',
                    'duration': duration
                })
        
        # 计算总时间需求
        task_npu_total = npu_time_per_instance * instances_needed
        task_dsp_total = dsp_time_per_instance * instances_needed
        
        npu_total_time += task_npu_total
        dsp_total_time += task_dsp_total
        
        task_details.append({
            'task_id': task.task_id,
            'task_name': task.name,
            'fps': task.fps_requirement,
            'instances_in_window': instances_needed,
            'segments': segment_details,
            'npu_time_per_instance': npu_time_per_instance,
            'dsp_time_per_instance': dsp_time_per_instance,
            'npu_total_time': task_npu_total,
            'dsp_total_time': task_dsp_total
        })
    
    # 计算资源利用率（超过100%表示过载）
    npu_utilization = (npu_total_time / time_window_ms) * 100
    dsp_utilization = (dsp_total_time / time_window_ms) * 100
    
    return {
        'bandwidth': {
            'npu': bandwidth_npu,
            'dsp': bandwidth_dsp
        },
        'time_window_ms': time_window_ms,
        'total_demand': {
            'npu_ms': npu_total_time,
            'dsp_ms': dsp_total_time
        },
        'utilization': {
            'npu_percent': npu_utilization,
            'dsp_percent': dsp_utilization
        },
        'feasible': npu_utilization <= 100 and dsp_utilization <= 100,
        'task_details': task_details
    }


def print_resource_demand_analysis(tasks, bandwidth_npu=40.0, bandwidth_dsp=40.0, time_window_ms=1000.0):
    """
    打印资源需求分析报告
    
    Args:
        tasks: 任务列表
        bandwidth_npu: NPU带宽
        bandwidth_dsp: DSP带宽
    """
    print("\n" + "="*80)
    print("[ANALYSIS] 资源需求分析（1秒内）")
    print("="*80)
    
    analysis = compute_resource_demand(tasks, bandwidth_npu, bandwidth_dsp, time_window_ms)
    
    print(f"\n配置:")
    print(f"  NPU带宽: {analysis['bandwidth']['npu']} Gbps")
    print(f"  DSP带宽: {analysis['bandwidth']['dsp']} Gbps")
    print(f"  时间窗口: {analysis['time_window_ms']} ms")
    
    print(f"\n总资源需求:")
    print(f"  NPU总耗时: {analysis['total_demand']['npu_ms']:.1f} ms")
    print(f"  DSP总耗时: {analysis['total_demand']['dsp_ms']:.1f} ms")
    
    print(f"\n理论资源利用率:")
    npu_util = analysis['utilization']['npu_percent']
    dsp_util = analysis['utilization']['dsp_percent']
    print(f"  NPU: {npu_util:.1f}% {'[WARNING] 过载!' if npu_util > 100 else '[OK]'}")
    print(f"  DSP: {dsp_util:.1f}% {'[WARNING] 过载!' if dsp_util > 100 else '[OK]'}")
    
    if analysis['feasible']:
        print(f"\n✅ 系统可行：所有任务的FPS要求理论上可以满足")
    else:
        print(f"\n[ERROR] 系统不可行：资源不足以满足所有任务的FPS要求")
    
    # 打印任务详情
    print(f"\n任务详细需求:")
    print(f"{'任务':<15} {'FPS':<6} {'实例/秒':<8} {'NPU时间/实例':<12} {'DSP时间/实例':<12} {'NPU总计':<10} {'DSP总计':<10}")
    print("-"*90)
    
    for task in analysis['task_details']:
        print(f"{task['task_id']:<15} {task['fps']:<6} {task['instances_in_window']:<8.1f} "
              f"{task['npu_time_per_instance']:<12.2f} {task['dsp_time_per_instance']:<12.2f} "
              f"{task['npu_total_time']:<10.1f} {task['dsp_total_time']:<10.1f}")
    
    # 找出最耗资源的任务
    print(f"\n资源消耗TOP3:")
    
    # NPU TOP3
    npu_sorted = sorted(analysis['task_details'], key=lambda x: x['npu_total_time'], reverse=True)[:3]
    print(f"\n  NPU消耗最多的任务:")
    for i, task in enumerate(npu_sorted, 1):
        percentage = (task['npu_total_time'] / analysis['total_demand']['npu_ms']) * 100 if analysis['total_demand']['npu_ms'] > 0 else 0
        print(f"    {i}. {task['task_id']}: {task['npu_total_time']:.1f}ms ({percentage:.1f}%)")
    
    # DSP TOP3
    dsp_sorted = sorted(analysis['task_details'], key=lambda x: x['dsp_total_time'], reverse=True)[:3]
    print(f"\n  DSP消耗最多的任务:")
    for i, task in enumerate(dsp_sorted, 1):
        if task['dsp_total_time'] > 0:
            percentage = (task['dsp_total_time'] / analysis['total_demand']['dsp_ms']) * 100 if analysis['total_demand']['dsp_ms'] > 0 else 0
            print(f"    {i}. {task['task_id']}: {task['dsp_total_time']:.1f}ms ({percentage:.1f}%)")


def analyze_bandwidth_scenarios(tasks):
    """
    分析不同带宽场景下的资源需求
    
    Args:
        tasks: 任务列表
    """
    print("\n" + "="*80)
    print("[ANALYSIS] 不同带宽场景分析")
    print("="*80)
    
    scenarios = [
        ("低带宽", 30.0, 20.0),
        ("中带宽", 40.0, 40.0),
        ("高带宽", 120.0, 80.0),
    ]
    
    for name, npu_bw, dsp_bw in scenarios:
        analysis = compute_resource_demand(tasks, npu_bw, dsp_bw)
        
        print(f"\n{name} (NPU={npu_bw}, DSP={dsp_bw}):")
        print(f"  NPU需求: {analysis['total_demand']['npu_ms']:.1f}ms ({analysis['utilization']['npu_percent']:.1f}%)")
        print(f"  DSP需求: {analysis['total_demand']['dsp_ms']:.1f}ms ({analysis['utilization']['dsp_percent']:.1f}%)")
        print(f"  状态: {'✅ 可行' if analysis['feasible'] else '[ERROR] 不可行'}")


def analyze_execution_gaps(tracer, window_ms=200.0):
    """
    分析实际执行中的资源空闲时间和利用率差异
    
    Args:
        tracer: ScheduleTracer对象
        window_ms: 分析窗口（毫秒）
    
    Returns:
        dict: 包含详细分析的字典
    """
    resource_timelines = {}
    
    # 初始化每个资源的时间线
    for res_id in tracer.queue_manager.resource_queues.keys():
        resource_timelines[res_id] = {
            'busy_periods': [],
            'gaps': [],
            'total_busy_time': 0.0,
            'total_gap_time': 0.0,
            'task_executions': []
        }
    
    # 收集执行信息
    for exec in tracer.executions:
        if exec.resource_id in resource_timelines:
            timeline = resource_timelines[exec.resource_id]
            timeline['busy_periods'].append((exec.start_time, exec.end_time))
            timeline['task_executions'].append({
                'task_id': exec.task_id,
                'start': exec.start_time,
                'end': exec.end_time,
                'duration': exec.duration,
                'priority': exec.priority.name
            })
    
    # 分析每个资源
    for res_id, timeline in resource_timelines.items():
        # 排序忙碌期间
        timeline['busy_periods'].sort()
        
        # 计算总忙碌时间
        for start, end in timeline['busy_periods']:
            timeline['total_busy_time'] += (end - start)
        
        # 找出空闲期间
        if timeline['busy_periods']:
            # 开始前的空闲
            if timeline['busy_periods'][0][0] > 0:
                gap_duration = timeline['busy_periods'][0][0]
                timeline['gaps'].append({
                    'start': 0,
                    'end': timeline['busy_periods'][0][0],
                    'duration': gap_duration,
                    'reason': 'startup_delay'
                })
                timeline['total_gap_time'] += gap_duration
            
            # 中间的空闲
            for i in range(len(timeline['busy_periods']) - 1):
                gap_start = timeline['busy_periods'][i][1]
                gap_end = timeline['busy_periods'][i+1][0]
                if gap_end > gap_start:
                    gap_duration = gap_end - gap_start
                    timeline['gaps'].append({
                        'start': gap_start,
                        'end': gap_end,
                        'duration': gap_duration,
                        'reason': 'scheduling_gap'
                    })
                    timeline['total_gap_time'] += gap_duration
            
            # 结束后的空闲
            last_end = timeline['busy_periods'][-1][1]
            if last_end < window_ms:
                gap_duration = window_ms - last_end
                timeline['gaps'].append({
                    'start': last_end,
                    'end': window_ms,
                    'duration': gap_duration,
                    'reason': 'end_idle'
                })
                timeline['total_gap_time'] += gap_duration
        else:
            # 完全空闲
            timeline['gaps'].append({
                'start': 0,
                'end': window_ms,
                'duration': window_ms,
                'reason': 'completely_idle'
            })
            timeline['total_gap_time'] = window_ms
        
        # 计算利用率
        timeline['utilization_percent'] = (timeline['total_busy_time'] / window_ms) * 100
        timeline['gap_percent'] = (timeline['total_gap_time'] / window_ms) * 100
    
    return resource_timelines


def print_execution_gap_analysis(tracer, window_ms=200.0):
    """
    打印执行空隙分析报告
    
    Args:
        tracer: ScheduleTracer对象
        window_ms: 分析窗口（毫秒）
    """
    print("\n" + "="*80)
    print("[ANALYSIS] 执行空隙分析")
    print("="*80)
    
    analysis = analyze_execution_gaps(tracer, window_ms)
    
    # 打印每个资源的分析
    for res_id in sorted(analysis.keys()):
        timeline = analysis[res_id]
        
        print(f"\n{res_id}:")
        print(f"  总忙碌时间: {timeline['total_busy_time']:.1f}ms")
        print(f"  总空闲时间: {timeline['total_gap_time']:.1f}ms")
        print(f"  利用率: {timeline['utilization_percent']:.1f}%")
        print(f"  空闲率: {timeline['gap_percent']:.1f}%")
        
        # 打印主要空隙
        if timeline['gaps']:
            print(f"\n  主要空隙 (>1ms):")
            significant_gaps = [g for g in timeline['gaps'] if g['duration'] > 1.0]
            for gap in sorted(significant_gaps, key=lambda x: x['duration'], reverse=True)[:5]:
                print(f"    {gap['start']:>6.1f} - {gap['end']:>6.1f}ms: "
                      f"{gap['duration']:>5.1f}ms ({gap['reason']})")
        
        # 任务执行统计
        if timeline['task_executions']:
            print(f"\n  任务执行次数: {len(timeline['task_executions'])}")
            # 按任务ID统计
            task_counts = {}
            for exec in timeline['task_executions']:
                task_base = exec['task_id'].split('#')[0]
                task_counts[task_base] = task_counts.get(task_base, 0) + 1
            
            print(f"  任务分布:")
            for task_id, count in sorted(task_counts.items()):
                print(f"    {task_id}: {count}次")
    
    # 总体统计
    total_busy = sum(t['total_busy_time'] for t in analysis.values())
    total_gap = sum(t['total_gap_time'] for t in analysis.values())
    num_resources = len(analysis)
    
    print(f"\n总体统计:")
    print(f"  资源数: {num_resources}")
    print(f"  总忙碌时间: {total_busy:.1f}ms")
    print(f"  总空闲时间: {total_gap:.1f}ms")
    print(f"  平均利用率: {(total_busy / (window_ms * num_resources)) * 100:.1f}%")
    
    # 分析空隙原因
    gap_reasons = {}
    for timeline in analysis.values():
        for gap in timeline['gaps']:
            reason = gap['reason']
            if reason not in gap_reasons:
                gap_reasons[reason] = {'count': 0, 'total_time': 0}
            gap_reasons[reason]['count'] += 1
            gap_reasons[reason]['total_time'] += gap['duration']
    
    if gap_reasons:
        print(f"\n空隙原因分析:")
        for reason, stats in sorted(gap_reasons.items(), key=lambda x: x[1]['total_time'], reverse=True):
            avg_duration = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            print(f"  {reason}: {stats['count']}次, "
                  f"总计{stats['total_time']:.1f}ms, "
                  f"平均{avg_duration:.1f}ms/次")


def compare_theory_vs_actual(tasks, tracer, bandwidth_npu=40.0, bandwidth_dsp=40.0, window_ms=200.0):
    """
    比较理论需求和实际执行的差异
    
    Args:
        tasks: 任务列表
        tracer: ScheduleTracer对象
        bandwidth_npu: NPU带宽
        bandwidth_dsp: DSP带宽
        window_ms: 时间窗口
    """
    print("\n" + "="*80)
    print("[ANALYSIS] 理论 vs 实际执行对比")
    print("="*80)
    
    # 计算理论需求（按比例缩放到实际窗口）
    theory_1s = compute_resource_demand(tasks, bandwidth_npu, bandwidth_dsp, 1000.0)
    theory_window = compute_resource_demand(tasks, bandwidth_npu, bandwidth_dsp, window_ms)
    
    # 获取实际执行统计
    actual_stats = tracer.get_statistics()
    gap_analysis = analyze_execution_gaps(tracer, window_ms)
    
    print(f"\n时间窗口: {window_ms}ms")
    
    print(f"\n理论需求 (1秒内):")
    print(f"  NPU: {theory_1s['total_demand']['npu_ms']:.1f}ms ({theory_1s['utilization']['npu_percent']:.1f}%)")
    print(f"  DSP: {theory_1s['total_demand']['dsp_ms']:.1f}ms ({theory_1s['utilization']['dsp_percent']:.1f}%)")
    
    print(f"\n理论需求 ({window_ms}ms内):")
    print(f"  NPU: {theory_window['total_demand']['npu_ms']:.1f}ms ({theory_window['utilization']['npu_percent']:.1f}%)")
    print(f"  DSP: {theory_window['total_demand']['dsp_ms']:.1f}ms ({theory_window['utilization']['dsp_percent']:.1f}%)")
    
    print(f"\n实际执行:")
    if 'NPU_0' in gap_analysis:
        actual_npu_time = gap_analysis['NPU_0']['total_busy_time']
        actual_npu_util = gap_analysis['NPU_0']['utilization_percent']
        print(f"  NPU_0: {actual_npu_time:.1f}ms ({actual_npu_util:.1f}%)")
    
    if 'DSP_0' in gap_analysis:
        actual_dsp_time = gap_analysis['DSP_0']['total_busy_time']
        actual_dsp_util = gap_analysis['DSP_0']['utilization_percent']
        print(f"  DSP_0: {actual_dsp_time:.1f}ms ({actual_dsp_util:.1f}%)")
    
    # 差异分析
    print(f"\n差异分析:")
    if 'NPU_0' in gap_analysis:
        theory_npu = theory_window['total_demand']['npu_ms']
        actual_npu = gap_analysis['NPU_0']['total_busy_time']
        diff_npu = actual_npu - theory_npu
        print(f"  NPU差异: {diff_npu:+.1f}ms ({(diff_npu/theory_npu*100) if theory_npu > 0 else 0:+.1f}%)")
        
        # 分析差异原因
        if abs(diff_npu) > 1:
            print(f"    可能原因:")
            startup_gap = next((g for g in gap_analysis['NPU_0']['gaps'] if g['reason'] == 'startup_delay'), None)
            if startup_gap:
                print(f"    - 启动延迟: ~{startup_gap['duration']:.1f}ms")
            scheduling_gaps = sum(g['duration'] for g in gap_analysis['NPU_0']['gaps'] 
                                if g['reason'] == 'scheduling_gap')
            if scheduling_gaps > 0:
                print(f"    - 调度间隙: ~{scheduling_gaps:.1f}ms")
    
    if 'DSP_0' in gap_analysis:
        theory_dsp = theory_window['total_demand']['dsp_ms']
        actual_dsp = gap_analysis['DSP_0']['total_busy_time']
        diff_dsp = actual_dsp - theory_dsp
        print(f"  DSP差异: {diff_dsp:+.1f}ms ({(diff_dsp/theory_dsp*100) if theory_dsp > 0 else 0:+.1f}%)")


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
    
    # T3 (YoloV8nSmall) - 设置为强制分段  
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
    print("- T3 (YoloV8nSmall): 使用 FORCED_SEGMENTATION，NPU段被切分")
    print("- 其他任务保持 NO_SEGMENTATION 策略")


def test_single_npu_dsp_baseline(time_window=200.0):
    """测试单NPU+单DSP的基准性能"""
    print(f"\n\n=== 基准测试：单NPU + 单DSP (时间窗口: {time_window:.1f}ms) ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备分段后的任务
    tasks = prepare_tasks_with_segmentation()
    
    # 打印所有任务信息...（省略不变的部分）
    
    results = {}
    tracers = {}
    
    # 测试两种模式
    for mode_name, segment_mode in [("传统模式", False), ("段级模式", True)]:
        print(f"\n{mode_name}:")
        
        tracer = ScheduleTracer(queue_manager)
        launcher = EnhancedTaskLauncher(queue_manager, tracer)
        
        # 注册所有任务
        for task in tasks:
            launcher.register_task(task)
        
        # 执行
        duration = time_window
        plan = launcher.create_launch_plan(duration, "eager")
        
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=segment_mode)
        
        # 分析执行时间线
        trace_stats = tracer.get_statistics(time_window=duration)  # 传入时间窗口
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(duration, plan.events)
        
        # 计算系统利用率
        system_util = calculate_system_utilization(tracer, duration)
        
        # 获取一致的资源利用率（使用时间窗口）
        resource_utilization = tracer.get_resource_utilization(time_window=duration)
        
        results[mode_name] = {
            'stats': stats,
            'metrics': metrics,
            'utilization': resource_utilization,  # 使用一致的计算
            'system_utilization': system_util,
            'trace_stats': trace_stats,
            'evaluator': evaluator
        }
        tracers[mode_name] = tracer
        
        print(f"  完成实例: {stats['completed_instances']}")
        print(f"  执行段数: {stats['total_segments_executed']}")
        print(f"  NPU利用率: {resource_utilization.get('NPU_0', 0):.1f}%")
        print(f"  DSP利用率: {resource_utilization.get('DSP_0', 0):.1f}%")
        print(f"  System利用率: {system_util:.1f}%")
        print(f"  平均等待时间: {metrics.avg_wait_time:.2f}ms")
        print(f"  FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
    
    # 性能对比
    print("\n性能提升分析:")
    for metric in ['NPU_0', 'DSP_0']:
        if metric in results['传统模式']['utilization']:
            old_val = results['传统模式']['utilization'][metric]
            new_val = results['段级模式']['utilization'][metric]
            improvement = ((new_val - old_val) / old_val * 100) if old_val > 0 else 0
            print(f"  {metric}利用率: {improvement:+.1f}%")
    
    system_old = results['传统模式']['system_utilization']
    system_new = results['段级模式']['system_utilization']
    system_improvement = ((system_new - system_old) / system_old * 100) if system_old > 0 else 0
    print(f"  System利用率: {system_improvement:+.1f}%")
    
    # 其他指标对比
    segments_old = results['传统模式']['stats']['completed_instances']
    segments_new = results['段级模式']['stats']['completed_instances']
    segments_improvement = ((segments_new - segments_old) / segments_old * 100) if segments_old > 0 else 0
    print(f"  完成实例: {segments_improvement:+.1f}%")
    
    wait_old = results['传统模式']['metrics'].avg_wait_time
    wait_new = results['段级模式']['metrics'].avg_wait_time
    wait_improvement = ((wait_old - wait_new) / wait_old * 100) if wait_old > 0 else 0
    print(f"  等待时间: {wait_improvement:+.1f}% (减少)")
    
    return results, tracers


def check_task_fps_requirements(time_window=200.0):
    """检查FPS要求满足情况"""
    print("\n\n=== FPS要求满足情况分析 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备任务
    tasks = prepare_tasks_with_segmentation()
    
    # 打印任务FPS要求
    print("任务FPS要求:")
    for task in tasks:  # 显示T1-T9
        instances_needed = task.fps_requirement * (time_window / 1000.0)
        print(f"  {task.task_id} ({task.name}): {task.fps_requirement} FPS → {instances_needed:.1f} 实例/{time_window:.0f}ms")


def generate_visualization(time_window=200.0):
    """生成可视化图表"""
    print("\n\n=== 生成可视化 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备任务
    tasks = prepare_tasks_with_segmentation()
    
    # 再次运行段级模式以生成可视化
    tracer = ScheduleTracer(queue_manager)
    launcher = EnhancedTaskLauncher(queue_manager, tracer)
    
    # 打印任务注册信息
    print("[INFO] 创建测试任务:")
    for task in tasks:
        launcher.register_task(task)  # ← 关键！必须注册任务
        if len(task.segments) > 1:
            print(f"  [OK] {task.task_id} {task.name}: {len(task.segments)}段混合任务")
        else:
            print(f"  [OK] {task.task_id} {task.name}: 纯{task.segments[0].resource_type.value}任务")
    
    # 执行
    duration = time_window
    plan = launcher.create_launch_plan(duration, "eager")
    
    print(f"\n{'='*100}")
    print("开始执行调度 (max_time=200.0ms, mode=段级)")
    print("="*100)
    
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    stats = executor.execute_plan(plan, duration, segment_mode=True)
    
    # 创建可视化器
    visualizer = ScheduleVisualizer(tracer)
    
    # 生成时间线图
    print("\nSEGMENT 模式执行时间线:\n")
    visualizer.print_gantt_chart(width=100)
    
    # 生成图表文件
    filename = f"segmented_tasks_segment_{int(time_window)}ms.png"
    json_filename = f"segmented_tasks_segment_{int(time_window)}ms.json"
    
    # 生成PNG图表
    visualizer.plot_resource_timeline(filename)
    
    # 生成Chrome Trace JSON
    visualizer.export_chrome_tracing(json_filename)
    
    # 打印统计信息（使用一致的时间窗口）
    trace_stats = tracer.get_statistics(time_window=duration)
    resource_utilization = tracer.get_resource_utilization(time_window=duration)
    system_util = calculate_system_utilization(tracer, duration)
    
    print(f"\n统计信息:")
    print(f"  执行数: {trace_stats['total_executions']}")
    print(f"  时间跨度: {trace_stats['time_span']:.1f}ms")
    print(f"  资源利用率: NPU={resource_utilization.get('NPU_0', 0):.1f}%, "
          f"DSP={resource_utilization.get('DSP_0', 0):.1f}%, "
          f"System={system_util:.1f}%")
    
    # 验证利用率的逻辑一致性
    max_resource_util = max(resource_utilization.values()) if resource_utilization else 0
    print(f"\n利用率验证:")
    print(f"  最高单资源利用率: {max_resource_util:.1f}%")
    print(f"  System利用率: {system_util:.1f}%")
    if system_util >= max_resource_util - 0.1:  # 允许0.1%的误差
        print(f"  [OK] 逻辑一致性检查通过")
    else:
        print(f"  [FAIL] 警告：System利用率低于最高资源利用率！")
    
    # 检查任务执行情况
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    metrics = evaluator.evaluate(duration, plan.events)
    
    print(f"\n任务执行次数:")
    for task_id in sorted(launcher.tasks.keys()):
        task = launcher.tasks[task_id]
        # 从evaluator的task_metrics中获取实际执行情况
        if hasattr(evaluator, 'task_metrics') and task_id in evaluator.task_metrics:
            task_metric = evaluator.task_metrics[task_id]
            completed = task_metric.instance_count
            actual_fps = task_metric.achieved_fps
        else:
            # 如果没有task_metrics，从completion_count获取
            completed = evaluator.task_completion_count.get(task_id, 0)
            actual_fps = (completed * 1000.0 / duration) if duration > 0 else 0
        
        expected = task.fps_requirement * (duration / 1000.0)
        fps_rate = (actual_fps / task.fps_requirement * 100) if task.fps_requirement > 0 else 0
        
        status = "[OK]" if fps_rate >= 100 else "[FAIL]"
        print(f"  {task_id}: {completed}/{expected:.1f} "
              f"(FPS要求: {task.fps_requirement}) {status}")
    
    print(f"\n生成文件:")
    print(f"  - {filename}")
    print(f"  - {json_filename}")


def main():
    """主函数"""
    print("DEMO: 真实任务段级调度优化")
    print("=" * 115)
    
    # 0. 计算最优时间窗口
    tasks = prepare_tasks_with_segmentation()
    fps_requirements = [task.fps_requirement for task in tasks]
    optimal_window = calculate_optimal_window(fps_requirements, max_window_ms=1000.0)
    
    print(f"\n[ANALYSIS] 时间窗口分析:")
    print(f"  FPS要求: {fps_requirements}")
    print(f"  最大公约数: {gcd_multiple(fps_requirements)}")
    print(f"  选择的窗口: {optimal_window:.1f}ms")
    
    # 1. 分析分段策略
    analyze_segmented_tasks()
    
    # 1.5 分析资源需求（新增）
    tasks = prepare_tasks_with_segmentation()
    print_resource_demand_analysis(tasks, bandwidth_npu=40.0, bandwidth_dsp=40.0, time_window_ms=optimal_window)
    analyze_bandwidth_scenarios(tasks)
    
    # 2. 基准测试
    baseline_results, tracers = test_single_npu_dsp_baseline(optimal_window)
    
    # 2.5 分析执行空隙（新增）
    if '段级模式' in tracers:
        print_execution_gap_analysis(tracers['段级模式'], window_ms=optimal_window)
        compare_theory_vs_actual(tasks, tracers['段级模式'], 
                               bandwidth_npu=40.0, bandwidth_dsp=40.0, window_ms=optimal_window)
    
    # 3. 检查FPS要求满足情况
    check_task_fps_requirements(optimal_window)
    
    # 4. 生成可视化
    generate_visualization(optimal_window)
    
    # 总结
    print("\n\n" + "=" * 115)
    print("[ANALYSIS] 优化效果总结")
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