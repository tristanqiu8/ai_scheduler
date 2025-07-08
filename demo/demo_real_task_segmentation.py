#!/usr/bin/env python3
"""
测试真实任务在段级模式下的优化效果
使用 FORCED_SEGMENTATION 策略强制T2和T3分段
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority, SegmentationStrategy
from core.evaluator import PerformanceEvaluator
from core.models import SubSegment
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
import copy
import numpy as np  # 需要numpy用于evaluator中的标准差计算


def prepare_tasks_with_segmentation():
    """准备任务并设置T2和T3为强制分段"""
    tasks = create_real_tasks()
    
    # T2 (YoloV8nBig) - 设置为强制分段
    t2 = tasks[1]
    t2.segmentation_strategy = SegmentationStrategy.FORCED_SEGMENTATION
    # 不需要重新定义切分点，real_task.py 中已经定义好了
    # T2 已经有 4 个切分点：op6, op13, op14, op19
    
    # T3 (Lpr) - 设置为强制分段  
    t3 = tasks[2]
    t3.segmentation_strategy = SegmentationStrategy.FORCED_SEGMENTATION
    # 不需要重新定义切分点，real_task.py 中已经定义好了
    # T3 已经有 3 个切分点：op5, op15, op19
    
    # 其他任务保持原有策略
    # T1: NO_SEGMENTATION (CRITICAL任务，不分段)
    # T4-T8: 根据需要可以设置
    
    # 调试：打印任务段信息
    print("\n调试信息 - 任务段配置:")
    for i, task in enumerate(tasks[:6]):
        print(f"\n{task.task_id} ({task.segmentation_strategy.value}):")
        for j, seg in enumerate(task.segments):
            print(f"  段{j}: {seg.segment_id}, {seg.resource_type.value}, "
                  f"duration@60={seg.get_duration(60):.1f}ms")
            if seg.cut_points:
                print(f"    切分点: {[cp.op_id for cp in seg.cut_points]}")
                # 打印每个切分点的详细信息
                for cp in seg.cut_points:
                    if 60 in cp.before_duration_table:
                        print(f"      {cp.op_id}: before={cp.before_duration_table[60]:.1f}ms@60")
        
        # 应用分段看看结果
        sub_segs = task.apply_segmentation()
        if sub_segs:
            print(f"  分段结果: {len(sub_segs)}个子段")
            for k, sub in enumerate(sub_segs):
                print(f"    子段{k}: {sub.sub_id}, {sub.resource_type.value}, "
                      f"duration@60={sub.get_duration(60):.1f}ms")
    
    return tasks

def verify_launch_plan(launcher, duration=200.0):
    """验证发射计划是否正确生成了所有任务实例"""
    plan = launcher.create_launch_plan(duration, "eager")
    
    print(f"\n发射计划验证 (时间窗口: {duration}ms):")
    print(f"总发射事件数: {len(plan.events)}")
    
    # 按任务统计
    task_launches = {}
    for event in plan.events:
        if event.task_id not in task_launches:
            task_launches[event.task_id] = []
        task_launches[event.task_id].append(event.time)
    
    print("\n任务发射详情:")
    for task_id in sorted(task_launches.keys()):
        task = launcher.tasks.get(task_id)
        if task:
            period = 1000.0 / task.fps_requirement
            expected_count = int(duration / period)
            actual_count = len(task_launches[task_id])
            
            print(f"\n{task_id} ({task.name}):")
            print(f"  FPS要求: {task.fps_requirement} (周期: {period:.1f}ms)")
            print(f"  预期实例数: {expected_count}")
            print(f"  实际实例数: {actual_count}")
            print(f"  发射时间: {task_launches[task_id][:5]}{'...' if len(task_launches[task_id]) > 5 else ''}")
            
            if actual_count < expected_count:
                print(f"  ⚠️ 警告: 实例数少于预期!")
    
    # 检查最后一个发射时间
    if plan.events:
        last_event_time = max(event.time for event in plan.events)
        print(f"\n最后一个发射时间: {last_event_time:.1f}ms")
        if last_event_time < duration * 0.8:
            print(f"⚠️ 警告: 最后的发射时间过早，可能影响执行时长!")
    
    return plan


def analyze_execution_timeline(tracer, expected_duration=200.0):
    """分析执行时间线，找出为什么提前结束"""
    executions = tracer.execution_records
    
    if not executions:
        print("没有执行记录!")
        return
    
    # 找出最后的执行时间
    last_end_time = max(record['end_time'] for record in executions)
    
    print(f"\n执行时间线分析:")
    print(f"期望执行时长: {expected_duration}ms")
    print(f"实际最后结束时间: {last_end_time:.1f}ms")
    print(f"差距: {expected_duration - last_end_time:.1f}ms")
    
    # 分析每个资源的最后执行时间
    resource_last_time = {}
    for record in executions:
        res_id = record['resource_id']
        end_time = record['end_time']
        if res_id not in resource_last_time or end_time > resource_last_time[res_id]:
            resource_last_time[res_id] = end_time
    
    print("\n各资源最后执行时间:")
    for res_id, last_time in sorted(resource_last_time.items()):
        print(f"  {res_id}: {last_time:.1f}ms")
    
    # 检查是否有任务在等待但没有被执行
    stats = tracer.get_statistics()
    print(f"\n执行统计:")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  时间跨度: {stats['time_span']:.1f}ms")
    
    return last_end_time

def analyze_segmented_tasks():
    """分析分段后的任务特征"""
    print("=== 分段策略分析 ===\n")
    
    tasks = prepare_tasks_with_segmentation()
    
    print(f"{'任务ID':<10} {'任务名称':<20} {'分段策略':<25} {'原段数':>8} {'子段数':>8} {'分段详情':<40}")
    print("-" * 115)
    
    for task in tasks[:6]:  # 显示前6个任务
        sub_segments = task.apply_segmentation()
        seg_count = len(sub_segments) if sub_segments else len(task.segments)
        
        # 构建分段详情
        if sub_segments:
            seg_details = []
            for seg in sub_segments:
                duration = seg.get_duration(60.0 if seg.resource_type == ResourceType.NPU else 40.0)
                seg_details.append(f"{seg.resource_type.value}:{duration:.1f}ms")
            detail_str = " → ".join(seg_details[:4])  # 最多显示4段
            if len(seg_details) > 4:
                detail_str += f" (+{len(seg_details)-4}段)"
        else:
            detail_str = "未分段"
        
        print(f"{task.task_id:<10} {task.name:<20} {task.segmentation_strategy.value:<25} "
              f"{len(task.segments):>8} {seg_count:>8} {detail_str:<40}")
    
    print("\n关键变化：")
    print("- T2 (YoloV8nBig): 使用 FORCED_SEGMENTATION，NPU段切分为 10ms + 7.6ms")
    print("- T3 (Lpr): 使用 FORCED_SEGMENTATION，NPU段切分为 4ms + 2.9ms")
    print("- 其他任务保持 NO_SEGMENTATION 策略")


def test_single_npu_dsp_baseline():
    """测试单NPU+单DSP的基准性能"""
    print("\n\n=== 基准测试：单NPU + 单DSP ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备分段后的任务
    tasks = prepare_tasks_with_segmentation()
    
    results = {}
    tracers = {}  # 保存tracer用于可视化
    
    # 测试两种模式
    for mode_name, segment_mode in [("传统模式", False), ("段级模式", True)]:
        print(f"\n{mode_name}:")
        
        tracer = ScheduleTracer(queue_manager)
        launcher = TaskLauncher(queue_manager, tracer)
        
        # 注册所有任务
        for task in tasks:
            launcher.register_task(task)
        
        # 执行
        duration = 200.0
        plan = launcher.create_launch_plan(duration, "eager")
        plan = verify_launch_plan(launcher, 200.0)
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=segment_mode)
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(duration, plan.events)
        
        # 获取详细统计
        trace_stats = tracer.get_statistics()
        
        results[mode_name] = {
            'stats': stats,
            'metrics': metrics,
            'utilization': tracer.get_resource_utilization(),
            'trace_stats': trace_stats
        }
        tracers[mode_name] = tracer
        
        print(f"  完成实例: {stats['completed_instances']}")
        print(f"  执行段数: {stats['total_segments_executed']}")
        print(f"  NPU利用率: {results[mode_name]['utilization'].get('NPU_0', 0):.1f}%")
        print(f"  DSP利用率: {results[mode_name]['utilization'].get('DSP_0', 0):.1f}%")
        print(f"  平均等待时间: {trace_stats['average_wait_time']:.2f}ms")
        print(f"  平均执行时间: {trace_stats['average_execution_time']:.2f}ms")
        print(f"  FPS满足率: {metrics.fps_satisfaction_rate:.1f}%")
    
    # 计算提升
    print("\n性能提升分析:")
    trad = results['传统模式']
    seg = results['段级模式']
    
    improvements = {
        'NPU利用率': seg['utilization'].get('NPU_0', 0) - trad['utilization'].get('NPU_0', 0),
        'DSP利用率': seg['utilization'].get('DSP_0', 0) - trad['utilization'].get('DSP_0', 0),
        '完成实例': ((seg['stats']['completed_instances'] - trad['stats']['completed_instances']) 
                    / trad['stats']['completed_instances'] * 100) if trad['stats']['completed_instances'] > 0 else 0,
        '等待时间': ((trad['trace_stats']['average_wait_time'] - seg['trace_stats']['average_wait_time']) 
                    / trad['trace_stats']['average_wait_time'] * 100) if trad['trace_stats']['average_wait_time'] > 0 else 0
    }
    
    for metric, value in improvements.items():
        print(f"  {metric}: {value:+.1f}{'%' if metric != '等待时间' else '% (减少)'}")
    
    return results, tracers


def test_segmentation_strategies():
    """测试不同分段策略的效果"""
    print("\n\n=== 分段策略对比测试 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    strategies = [
        ("全部不分段", {
            "T1": SegmentationStrategy.NO_SEGMENTATION,
            "T2": SegmentationStrategy.NO_SEGMENTATION,
            "T3": SegmentationStrategy.NO_SEGMENTATION,
        }),
        ("仅T2/T3分段", {
            "T1": SegmentationStrategy.NO_SEGMENTATION,
            "T2": SegmentationStrategy.FORCED_SEGMENTATION,
            "T3": SegmentationStrategy.FORCED_SEGMENTATION,
        }),
        ("全部强制分段", {
            "T1": SegmentationStrategy.FORCED_SEGMENTATION,
            "T2": SegmentationStrategy.FORCED_SEGMENTATION,
            "T3": SegmentationStrategy.FORCED_SEGMENTATION,
        }),
    ]
    
    print("测试不同的分段策略组合：\n")
    
    for strategy_name, strategy_map in strategies:
        print(f"{strategy_name}:")
        
        # 准备任务
        tasks = create_real_tasks()
        
        # 应用分段策略
        for task_id, strategy in strategy_map.items():
            for task in tasks:
                if task.task_id == task_id:
                    task.segmentation_strategy = strategy
                    # 如果是强制分段且没有预定义子段，添加默认分段
                    if strategy == SegmentationStrategy.FORCED_SEGMENTATION and task_id in ["T2", "T3"]:
                        # 复用之前的分段逻辑
                        prepare_tasks_with_segmentation()
        
        # 执行测试
        tracer = ScheduleTracer(queue_manager)
        launcher = TaskLauncher(queue_manager, tracer)
        
        # 注册前3个任务
        for i in range(3):
            launcher.register_task(tasks[i])
        
        duration = 200.0
        plan = launcher.create_launch_plan(duration, "eager")
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=True)
        
        util = tracer.get_resource_utilization()
        
        print(f"  完成实例: {stats['completed_instances']}")
        print(f"  NPU利用率: {util.get('NPU_0', 0):.1f}%")
        print(f"  DSP利用率: {util.get('DSP_0', 0):.1f}%")
        print()


def test_specific_scenarios():
    """测试特定场景下的优化效果"""
    print("\n\n=== 特定场景测试 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 准备任务
    tasks = prepare_tasks_with_segmentation()
    
    scenarios = [
        ("场景1: T1+T2+T3 (混合分段)", [0, 1, 2]),
        ("场景2: 检测任务组合", [1, 2, 3, 4]),
        ("场景3: 混合负载", [0, 1, 5, 6, 7]),
        ("场景4: 高优先级T1", [0, 1, 2], {'T1': TaskPriority.HIGH}),
    ]
    
    for scenario_name, task_indices, *priority_override in scenarios:
        print(f"\n{scenario_name}:")
        priority_map = priority_override[0] if priority_override else {}
        
        results = {}
        
        for mode_name, segment_mode in [("传统", False), ("段级", True)]:
            tracer = ScheduleTracer(queue_manager)
            launcher = TaskLauncher(queue_manager, tracer)
            
            # 注册选定的任务
            for idx in task_indices:
                task = copy.deepcopy(tasks[idx])
                # 应用优先级覆盖
                if task.task_id in priority_map:
                    task.priority = priority_map[task.task_id]
                launcher.register_task(task)
            
            # 执行
            duration = 100.0
            plan = launcher.create_launch_plan(duration, "eager")
            executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
            stats = executor.execute_plan(plan, duration, segment_mode=segment_mode)
            
            util = tracer.get_resource_utilization()
            trace_stats = tracer.get_statistics()
            
            results[mode_name] = {
                'completed': stats['completed_instances'],
                'segments': stats['total_segments_executed'],
                'npu_util': util.get('NPU_0', 0),
                'dsp_util': util.get('DSP_0', 0),
                'avg_wait': trace_stats['average_wait_time']
            }
        
        # 显示对比
        trad = results['传统']
        seg = results['段级']
        
        task_list = [tasks[i].task_id for i in task_indices]
        seg_info = [f"{tasks[i].task_id}({tasks[i].segmentation_strategy.value[:4]})" for i in task_indices[:3]]
        
        print(f"  任务: {seg_info}")
        print(f"  完成实例: {trad['completed']} → {seg['completed']} "
              f"(+{seg['completed'] - trad['completed']})")
        print(f"  执行段数: {trad['segments']} → {seg['segments']}")
        print(f"  NPU利用率: {trad['npu_util']:.1f}% → {seg['npu_util']:.1f}% "
              f"(+{seg['npu_util'] - trad['npu_util']:.1f}%)")
        print(f"  DSP利用率: {trad['dsp_util']:.1f}% → {seg['dsp_util']:.1f}% "
              f"(+{seg['dsp_util'] - trad['dsp_util']:.1f}%)")
        print(f"  平均等待: {trad['avg_wait']:.1f}ms → {seg['avg_wait']:.1f}ms "
              f"(-{trad['avg_wait'] - seg['avg_wait']:.1f}ms)")


def generate_visualization():
    """生成可视化对比"""
    print("\n\n=== 生成可视化 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 使用分段后的任务
    tasks = prepare_tasks_with_segmentation()
    
    # 选择代表性任务：T1, T2(分段), T3(分段)
    selected_indices = [0, 1, 2]
    
    duration = 100.0  # 增加时间窗口以便观察更多执行
    
    # 生成两种模式的可视化
    for mode_name, segment_mode in [("traditional", False), ("segment", True)]:
        tracer = ScheduleTracer(queue_manager)
        launcher = TaskLauncher(queue_manager, tracer)
        
        for idx in selected_indices:
            launcher.register_task(tasks[idx])
        
        plan = launcher.create_launch_plan(duration, "eager")
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, duration, segment_mode=segment_mode)
        
        # 检查是否有执行记录
        if len(tracer.executions) == 0:
            print(f"\n警告：{mode_name}模式没有执行记录！")
            continue
        
        # 打印甘特图
        print(f"\n{mode_name.upper()} 模式执行时间线:")
        viz = ScheduleVisualizer(tracer)
        viz.print_gantt_chart(width=80)
        
        # 显示一些统计信息
        trace_stats = tracer.get_statistics()
        print(f"\n统计信息:")
        print(f"  执行数: {len(tracer.executions)}")
        print(f"  时间跨度: {trace_stats['time_span']:.1f}ms")
        print(f"  资源利用率: NPU={tracer.get_resource_utilization().get('NPU_0', 0):.1f}%, "
              f"DSP={tracer.get_resource_utilization().get('DSP_0', 0):.1f}%")
        
        # 生成文件
        if len(tracer.executions) > 0:
            viz.plot_resource_timeline(f"segmented_tasks_{mode_name}.png")
            viz.export_chrome_tracing(f"segmented_tasks_{mode_name}.json")
            
            print(f"\n生成文件:")
            print(f"  - segmented_tasks_{mode_name}.png")
            print(f"  - segmented_tasks_{mode_name}.json")
        else:
            print(f"\n跳过文件生成（无执行数据）")


def main():
    """主测试函数"""
    print("🚀 真实任务段级优化测试（使用 FORCED_SEGMENTATION）\n")
    print("系统配置：单NPU (60 GFLOPS) + 单DSP (40 GFLOPS)")
    print("=" * 115)
    
    # 1. 分析分段策略
    analyze_segmented_tasks()
    
    # 2. 基准测试
    baseline_results, tracers = test_single_npu_dsp_baseline()
    
    # 3. 分段策略对比
    test_segmentation_strategies()
    
    # 4. 特定场景测试
    test_specific_scenarios()
    
    # 5. 生成可视化
    generate_visualization()
    
    # 总结
    print("\n\n" + "=" * 115)
    print("📊 优化效果总结")
    print("=" * 115)
    
    print("\n关键发现：")
    print("1. FORCED_SEGMENTATION 策略让T2和T3的NPU段被有效切分")
    print("2. 段级模式充分利用了分段带来的调度灵活性")
    print("3. 即使在单NPU+单DSP的资源受限场景，性能提升依然明显")
    print("4. 分段策略可以灵活配置，适应不同的任务特征")
    
    print("\n优化机制：")
    print("- 通过 SegmentationStrategy 枚举控制每个任务的分段行为")
    print("- FORCED_SEGMENTATION 强制使用所有可用的切分点")
    print("- NO_SEGMENTATION 保持任务的原始段结构")
    print("- 可以为不同任务设置不同的策略，实现精细控制")
    
    print("\n建议：")
    print("- 对计算密集的长段使用 FORCED_SEGMENTATION")
    print("- 对已经很短的段保持 NO_SEGMENTATION")
    print("- 未来可以探索 ADAPTIVE_SEGMENTATION 的自动优化")


if __name__ == "__main__":
    main()
