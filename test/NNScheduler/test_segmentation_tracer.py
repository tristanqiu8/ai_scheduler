#!/usr/bin/env python3
"""
测试 ScheduleTracer 对分段任务的追踪能力
验证分段执行的正确记录和可视化
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.enums import ResourceType, TaskPriority, SegmentationStrategy
from NNScheduler.core.task import NNTask
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer


def test_basic_segmentation_tracing():
    """测试基本的分段任务追踪"""
    print("=== 测试分段任务追踪 ===\n")
    
    # 创建资源
    manager = ResourceQueueManager()
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(manager)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建一个分段任务
    task = NNTask(
        "YOLO_V8",
        "YoloV8-Segmented",
        priority=TaskPriority.NORMAL,
        segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION
    )
    
    # 添加NPU主段
    task.add_segment(ResourceType.NPU, {60: 12.0}, "npu_main")
    
    # 添加切分点
    task.add_cut_points_to_segment("npu_main", [
        ("layer_10", {60: 3.0}, 0.1),   # 25%
        ("layer_20", {60: 6.0}, 0.1),   # 50%
        ("layer_30", {60: 9.0}, 0.1),   # 75%
    ])
    
    # 选择切分配置（切成3段）
    task.set_preset_cut_configurations("npu_main", [
        [],                    # 不切
        ["layer_20"],         # 2段
        ["layer_10", "layer_30"], # 3段
    ])
    task.select_cut_configuration("npu_main", 2)
    
    # 应用分段
    sub_segments = task.apply_segmentation()
    
    print(f"任务 {task.task_id} 分段情况:")
    print(f"  原始段: 1个NPU段 (12ms @60带宽)")
    print(f"  分段后: {len(sub_segments)}个子段")
    
    # 模拟执行并追踪
    current_time = 0.0
    
    # 第一次执行实例
    print("\n第一次执行实例 (t=0ms开始):")
    for i, sub_seg in enumerate(sub_segments):
        # 记录入队
        task_id = f"{task.task_id}#0_seg{i}"
        tracer.record_enqueue(
            task_id,
            "NPU_0" if i % 2 == 0 else "NPU_1",  # 交替使用NPU
            task.priority,
            current_time,
            [sub_seg]
        )
        
        # 记录执行
        start = current_time + i * 0.5  # 轻微错开
        duration = sub_seg.get_duration(60.0)
        end = start + duration
        
        tracer.record_execution(
            task_id,
            "NPU_0" if i % 2 == 0 else "NPU_1",
            start,
            end,
            60.0,
            sub_seg.sub_id
        )
        
        print(f"  段{i} ({sub_seg.sub_id}): {start:.1f}-{end:.1f}ms "
              f"在 {'NPU_0' if i % 2 == 0 else 'NPU_1'}")
    
    # 第二次执行实例（展示周期性）
    print("\n第二次执行实例 (t=20ms开始):")
    current_time = 20.0
    
    for i, sub_seg in enumerate(sub_segments):
        task_id = f"{task.task_id}#1_seg{i}"
        
        # 这次全部在NPU_0上执行（展示不同的调度策略）
        tracer.record_enqueue(task_id, "NPU_0", task.priority, current_time, [sub_seg])
        
        # 顺序执行
        if i == 0:
            start = current_time
        else:
            # 等待前一段完成
            start = last_end
        
        duration = sub_seg.get_duration(60.0)
        end = start + duration
        last_end = end
        
        tracer.record_execution(
            task_id,
            "NPU_0",
            start,
            end,
            60.0,
            sub_seg.sub_id
        )
        
        print(f"  段{i} ({sub_seg.sub_id}): {start:.1f}-{end:.1f}ms 在 NPU_0")
    
    # 显示甘特图
    print("\n文本甘特图:")
    print("-" * 80)
    visualizer.print_gantt_chart(width=80)
    
    # 生成可视化文件
    visualizer.plot_resource_timeline("segmentation_trace_basic.png")
    print("\n✓ 生成图表: segmentation_trace_basic.png")
    
    # 显示统计
    stats = tracer.get_statistics()
    print(f"\n执行统计:")
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  平均执行时间: {stats['average_execution_time']:.2f}ms")


def test_complex_segmentation_scenario():
    """测试复杂的分段场景：多任务、多资源、依赖关系"""
    print("\n\n=== 测试复杂分段场景 ===\n")
    
    # 创建资源
    manager = ResourceQueueManager()
    for i in range(2):
        manager.add_resource(f"NPU_{i}", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(manager)
    visualizer = ScheduleVisualizer(tracer)
    
    # 场景：3个任务，不同的分段策略
    tasks = []
    
    # 任务1：大任务，分4段
    task1 = NNTask("BIG_TASK", "BigSegmented", priority=TaskPriority.NORMAL)
    task1.add_segment(ResourceType.NPU, {60: 20.0}, "big_npu")
    task1.add_cut_points_to_segment("big_npu", [
        ("cut1", {60: 5.0}, 0.1),
        ("cut2", {60: 10.0}, 0.1),
        ("cut3", {60: 15.0}, 0.1),
    ])
    task1.set_preset_cut_configurations("big_npu", [["cut1", "cut2", "cut3"]])
    task1.select_cut_configuration("big_npu", 0)
    tasks.append(task1)
    
    # 任务2：中等任务，分2段
    task2 = NNTask("MED_TASK", "MediumSegmented", priority=TaskPriority.HIGH)
    task2.add_segment(ResourceType.NPU, {60: 10.0}, "med_npu")
    task2.add_cut_points_to_segment("med_npu", [("mid", {60: 5.0}, 0.1)])
    task2.set_preset_cut_configurations("med_npu", [["mid"]])
    task2.select_cut_configuration("med_npu", 0)
    tasks.append(task2)
    
    # 任务3：混合任务，NPU段分2段，DSP不分
    task3 = NNTask("MIX_TASK", "MixedSegmented", priority=TaskPriority.CRITICAL)
    task3.add_segment(ResourceType.NPU, {60: 8.0}, "mix_npu")
    task3.add_segment(ResourceType.DSP, {40: 3.0}, "mix_dsp")
    task3.add_cut_points_to_segment("mix_npu", [("split", {60: 4.0}, 0.1)])
    task3.set_preset_cut_configurations("mix_npu", [["split"]])
    task3.select_cut_configuration("mix_npu", 0)
    tasks.append(task3)
    
    # 执行调度模拟
    print("任务配置:")
    for task in tasks:
        sub_segs = task.apply_segmentation()
        print(f"  {task.task_id} ({task.priority.name}): {len(sub_segs)}个子段")
    
    # 模拟调度执行
    current_time = 0.0
    
    # 按优先级排序任务
    sorted_tasks = sorted(tasks, key=lambda t: t.priority.value)
    
    for task in sorted_tasks:
        sub_segments = task.apply_segmentation()
        print(f"\n调度 {task.task_id}:")
        
        for i, sub_seg in enumerate(sub_segments):
            # 选择资源
            if sub_seg.resource_type == ResourceType.NPU:
                # 使用轮询分配NPU
                resource_id = f"NPU_{i % 2}"
            else:
                resource_id = "DSP_0"
            
            # 计算开始时间（简化：立即开始或等待资源）
            start_time = current_time + i * 2.0
            duration = sub_seg.get_duration(60.0 if sub_seg.resource_type == ResourceType.NPU else 40.0)
            end_time = start_time + duration
            
            # 记录追踪
            seg_task_id = f"{task.task_id}_seg{i}"
            tracer.record_enqueue(
                seg_task_id,
                resource_id,
                task.priority,
                start_time,
                [sub_seg]
            )
            
            tracer.record_execution(
                seg_task_id,
                resource_id,
                start_time,
                end_time,
                60.0 if sub_seg.resource_type == ResourceType.NPU else 40.0,
                sub_seg.sub_id
            )
            
            print(f"  段{i} ({sub_seg.sub_id}): {start_time:.1f}-{end_time:.1f}ms "
                  f"在 {resource_id}")
        
        current_time += 5.0  # 任务间间隔
    
    # 生成报告
    print("\n" + "="*80)
    visualizer.print_gantt_chart(width=80)
    
    # 生成可视化
    visualizer.plot_resource_timeline("segmentation_trace_complex.png")
    visualizer.export_chrome_tracing("segmentation_trace_complex.json")
    
    print("\n生成的文件:")
    print("  ✓ segmentation_trace_complex.png - Matplotlib甘特图")
    print("  ✓ segmentation_trace_complex.json - Chrome Tracing文件")
    
    # 详细统计
    stats = tracer.get_statistics()
    print(f"\n详细统计:")
    print(f"  总任务段数: {stats['total_executions']}")
    print(f"  时间跨度: {stats['time_span']:.1f}ms")
    
    print(f"\n资源利用率:")
    for res_id in ["NPU_0", "NPU_1", "DSP_0"]:
        util = stats['resource_utilization'].get(res_id, 0)
        print(f"  {res_id}: {util:.1f}%")
    
    print(f"\n优先级分布:")
    for priority, count in stats['tasks_by_priority'].items():
        print(f"  {priority}: {count}个任务段")


def test_segmentation_performance_impact():
    """测试分段对性能的影响"""
    print("\n\n=== 测试分段性能影响 ===\n")
    
    manager = ResourceQueueManager()
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    # 对同一任务测试不同分段策略
    base_task = NNTask("PERF_TEST", "PerformanceTest")
    base_task.add_segment(ResourceType.NPU, {60: 24.0}, "main")
    
    # 添加多个切分点
    base_task.add_cut_points_to_segment("main", [
        ("p1", {60: 4.0}, 0.1),
        ("p2", {60: 8.0}, 0.1),
        ("p3", {60: 12.0}, 0.1),
        ("p4", {60: 16.0}, 0.1),
        ("p5", {60: 20.0}, 0.1),
    ])
    
    # 测试不同的分段数
    configs = [
        ([], "不分段"),
        (["p3"], "2段"),
        (["p2", "p4"], "3段"),
        (["p1", "p3", "p5"], "4段"),
        (["p1", "p2", "p3", "p4"], "5段"),
    ]
    
    print("对比不同分段策略的执行时间:\n")
    print(f"{'策略':<10} {'子段数':<8} {'总时间':<10} {'开销':<10} {'开销比例':<10}")
    print("-" * 50)
    
    for cut_points, desc in configs:
        # 设置切分配置
        base_task.set_preset_cut_configurations("main", [cut_points])
        base_task.select_cut_configuration("main", 0)
        
        # 应用分段
        sub_segments = base_task.apply_segmentation()
        
        # 计算总时间
        total_time = 0.0
        total_overhead = 0.0
        
        for sub_seg in sub_segments:
            duration = sub_seg.get_duration(60.0)
            total_time += duration
            total_overhead += sub_seg.cut_overhead
        
        overhead_ratio = (total_overhead / 24.0) * 100 if 24.0 > 0 else 0
        
        print(f"{desc:<10} {len(sub_segments):<8} {total_time:<10.2f} "
              f"{total_overhead:<10.2f} {overhead_ratio:<10.1f}%")
    
    print("\n结论：分段数增加会带来额外开销，需要权衡并行收益与开销")


def test_timeline_analysis():
    """测试时间线分析功能"""
    print("\n\n=== 测试时间线分析 ===\n")
    
    manager = ResourceQueueManager()
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(manager)
    
    # 模拟一个任务的多个分段执行
    task_id_base = "TIMELINE_TASK"
    segments = [
        ("seg0", 0.0, 3.0),
        ("seg1", 3.1, 6.0),
        ("seg2", 10.0, 13.0),  # 有间隔
        ("seg3", 13.1, 15.0),
    ]
    
    for seg_id, start, end in segments:
        tracer.record_execution(
            f"{task_id_base}_{seg_id}",
            "NPU_0",
            start,
            end,
            60.0,
            seg_id
        )
    
    # 获取任务时间线
    timeline = tracer.get_timeline()
    
    print("NPU_0 执行时间线:")
    for exec in timeline["NPU_0"]:
        print(f"  {exec.start_time:>5.1f} - {exec.end_time:>5.1f}ms: "
              f"{exec.task_id} (段: {exec.segment_id})")
    
    # 分析间隔
    print("\n执行间隔分析:")
    execs = timeline["NPU_0"]
    for i in range(1, len(execs)):
        gap = execs[i].start_time - execs[i-1].end_time
        if gap > 0.1:  # 显著间隔
            print(f"  {execs[i-1].end_time:.1f} - {execs[i].start_time:.1f}ms: "
                  f"空闲 {gap:.1f}ms")
    
    # 计算实际利用率
    total_busy = sum(e.duration for e in execs)
    time_span = execs[-1].end_time - execs[0].start_time
    actual_util = (total_busy / time_span * 100) if time_span > 0 else 0
    
    print(f"\n时间线分析:")
    print(f"  时间跨度: {time_span:.1f}ms")
    print(f"  总忙碌时间: {total_busy:.1f}ms")
    print(f"  实际利用率: {actual_util:.1f}%")


if __name__ == "__main__":
    # 运行所有测试
    test_basic_segmentation_tracing()
    test_complex_segmentation_scenario()
    test_segmentation_performance_impact()
    test_timeline_analysis()
    
    print("\n\n✅ 所有分段追踪测试完成！")
