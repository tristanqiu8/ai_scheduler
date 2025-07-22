#!/usr/bin/env python3
"""
测试任务分段功能与发射器的集成
重点测试NPU任务的分段执行
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enums import ResourceType, TaskPriority, SegmentationStrategy
from core.task import NNTask
from viz.schedule_visualizer import ScheduleVisualizer


def create_segmentable_task():
    """创建一个可分段的NPU任务"""
    task = NNTask(
        "YOLO_SEG", 
        "YoloV8n-Segmentable",
        priority=TaskPriority.NORMAL,
        segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION
    )
    
    # 添加主NPU段，执行时间较长
    main_segment = task.add_segment(
        ResourceType.NPU,
        {20: 23.494, 40: 13.684, 60: 9.123, 120: 7.411},
        "npu_main"
    )
    
    # 添加DSP后处理段
    task.add_segment(
        ResourceType.DSP,
        {40: 3.423},
        "dsp_postprocess"
    )
    
    # 为NPU主段添加切分点
    task.add_cut_points_to_segment("npu_main", [
        ("conv_6", {20: 4.699, 40: 2.737, 60: 1.825, 120: 1.482}, 0.1),   # 20%处
        ("conv_13", {20: 9.398, 40: 5.474, 60: 3.649, 120: 2.964}, 0.1),  # 40%处
        ("conv_20", {20: 14.096, 40: 8.210, 60: 5.474, 120: 4.447}, 0.1), # 60%处
        ("conv_27", {20: 18.795, 40: 10.947, 60: 7.298, 120: 5.929}, 0.1),# 80%处
    ])
    
    # 设置预定义的切分配置
    task.set_preset_cut_configurations("npu_main", [
        [],                                      # 配置0: 不切分
        ["conv_13"],                            # 配置1: 切成2段 (中间切)
        ["conv_6", "conv_20"],                  # 配置2: 切成3段
        ["conv_6", "conv_13", "conv_20"],       # 配置3: 切成4段
        ["conv_6", "conv_13", "conv_20", "conv_27"], # 配置4: 切成5段
    ])
    
    # 设置性能要求
    task.set_performance_requirements(fps=10, latency=100)
    
    return task


def test_segmentation_with_launcher():
    """测试分段任务的发射和执行"""
    print("=== 测试分段任务发射 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建可分段任务
    seg_task = create_segmentable_task()
    
    # 测试不同的分段配置
    for config_idx in [0, 2, 4]:  # 测试不分段、3段、5段
        print(f"\n{'='*60}")
        print(f"测试配置 {config_idx}")
        print(f"{'='*60}")
        
        # 选择分段配置
        seg_task.select_cut_configuration("npu_main", config_idx)
        sub_segments = seg_task.apply_segmentation()
        
        print(f"\n分段情况:")
        print(f"  原始段数: {len(seg_task.segments)}")
        print(f"  分段后子段数: {len(sub_segments)}")
        
        # 打印子段详情
        for i, sub_seg in enumerate(sub_segments):
            duration_60 = sub_seg.get_duration(60.0)
            print(f"  子段{i} ({sub_seg.sub_id}): {sub_seg.resource_type.value}, "
                  f"{duration_60:.2f}ms @60带宽")
        
        # 重新创建launcher和tracer（清空状态）
        tracer = ScheduleTracer(queue_manager)
        launcher = TaskLauncher(queue_manager, tracer)
        
        # 注册任务
        launcher.register_task(seg_task)
        
        # 创建发射计划
        time_window = 200.0
        plan = launcher.create_launch_plan(time_window, "eager")
        
        print(f"\n发射计划: {len(plan.events)}次发射")
        for event in plan.events[:3]:
            print(f"  {event.time:.1f}ms: {event.task_id}#{event.instance_id}")
        
        # 模拟执行
        print(f"\n模拟执行:")
        simulate_segmented_execution(
            launcher, queue_manager, tracer, plan, 
            seg_task, sub_segments, time_window
        )
        
        # 统计
        stats = tracer.get_statistics()
        print(f"\n执行统计:")
        print(f"  总执行段数: {stats['total_executions']}")
        print(f"  资源利用率:")
        for res_id in ["NPU_0", "NPU_1", "DSP_0"]:
            util = stats['resource_utilization'].get(res_id, 0)
            print(f"    {res_id}: {util:.1f}%")
        
        # 生成可视化
        filename = f"segmentation_test_config{config_idx}.png"
        visualizer.plot_resource_timeline(filename)
        print(f"\n  ✓ 生成甘特图: {filename}")


def simulate_segmented_execution(launcher, queue_manager, tracer, plan, 
                                task, sub_segments, max_time):
    """模拟分段任务的执行"""
    current_time = 0.0
    event_idx = 0
    
    # 追踪每个任务实例的执行进度
    task_progress = {}  # (task_id, instance) -> current_segment_index
    
    while current_time < max_time and event_idx < len(plan.events):
        # 处理发射事件
        while event_idx < len(plan.events) and plan.events[event_idx].time <= current_time:
            event = plan.events[event_idx]
            
            # 发射任务（所有子段）
            for i, sub_seg in enumerate(sub_segments):
                queue = queue_manager.find_best_queue(sub_seg.resource_type)
                if queue:
                    # 创建子段任务ID
                    sub_task_id = f"{event.task_id}#{event.instance_id}_seg{i}"
                    
                    # 加入队列
                    queue.enqueue(
                        sub_task_id,
                        task.priority,
                        current_time,  # 直接使用当前时间，子段没有start_time
                        [sub_seg]  # 单个子段
                    )
                    
                    # 记录入队
                    tracer.record_enqueue(
                        sub_task_id,
                        queue.resource_id,
                        task.priority,
                        current_time,  # 直接使用当前时间
                        [sub_seg]
                    )
            
            print(f"  {current_time:.1f}ms: 发射 {event.task_id}#{event.instance_id} "
                  f"({len(sub_segments)}个子段)")
            event_idx += 1
        
        # 执行资源上的任务
        for queue_id, queue in queue_manager.resource_queues.items():
            queue.advance_time(current_time)
            
            if not queue.is_busy():
                next_task = queue.get_next_task()
                if next_task and next_task.ready_time <= current_time:
                    # 获取子段
                    sub_seg = next_task.sub_segments[0] if next_task.sub_segments else None
                    if sub_seg:
                        # 计算执行时间
                        duration = sub_seg.get_duration(queue.bandwidth)
                        end_time = current_time + duration
                        
                        # 记录执行
                        tracer.record_execution(
                            next_task.task_id,
                            queue_id,
                            current_time,
                            end_time,
                            queue.bandwidth,
                            sub_seg.sub_id
                        )
                        
                        # 更新资源状态
                        queue.busy_until = end_time
                        queue.current_task = next_task.task_id
                        
                        # 从队列移除
                        queue.dequeue_task(next_task.task_id, next_task.priority)
        
        # 时间推进
        current_time += 0.5


def test_segmentation_dependencies():
    """测试分段任务之间的依赖关系"""
    print("\n\n=== 测试分段任务依赖 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建两个可分段任务，B依赖A
    task_a = create_segmentable_task()
    task_a.task_id = "TASK_A"
    task_a.select_cut_configuration("npu_main", 2)  # 3段
    
    task_b = create_segmentable_task()
    task_b.task_id = "TASK_B"
    task_b.select_cut_configuration("npu_main", 1)  # 2段
    task_b.add_dependency("TASK_A")
    
    # 注册任务
    launcher.register_task(task_a)
    launcher.register_task(task_b)
    
    # 创建发射计划
    plan = launcher.create_launch_plan(100.0, "eager")
    
    print("任务配置:")
    print(f"  TASK_A: {len(task_a.apply_segmentation())}个子段")
    print(f"  TASK_B: {len(task_b.apply_segmentation())}个子段，依赖TASK_A")
    
    print("\n发射计划:")
    for event in plan.events[:5]:
        print(f"  {event.time:.1f}ms: {event.task_id}#{event.instance_id}")
    
    print("\n注意：由于B依赖A，B的发射会被延迟直到A完成")


def test_mixed_segmentation_workload():
    """测试混合工作负载：分段和非分段任务"""
    print("\n\n=== 测试混合工作负载 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建任务组合
    tasks = []
    
    # 1. 可分段的大任务
    seg_task = create_segmentable_task()
    seg_task.task_id = "BIG_SEG"
    seg_task.priority = TaskPriority.NORMAL
    seg_task.select_cut_configuration("npu_main", 3)  # 4段
    tasks.append(seg_task)
    
    # 2. 不分段的小任务（高优先级）
    from core.task import create_npu_task
    small_task = create_npu_task(
        "SMALL_HIGH", 
        "SmallHighPri",
        {60: 2.0},
        priority=TaskPriority.HIGH
    )
    small_task.set_performance_requirements(fps=50, latency=20)
    tasks.append(small_task)
    
    # 3. 中等任务（关键）
    medium_task = create_npu_task(
        "MED_CRIT",
        "MediumCritical", 
        {60: 5.0},
        priority=TaskPriority.CRITICAL
    )
    medium_task.set_performance_requirements(fps=25, latency=40)
    tasks.append(medium_task)
    
    # 注册所有任务
    for task in tasks:
        launcher.register_task(task)
    
    # 创建发射计划
    time_window = 100.0
    plan = launcher.create_launch_plan(time_window, "balanced")
    
    print("工作负载配置:")
    for task in tasks:
        if hasattr(task, 'apply_segmentation'):
            segments = task.apply_segmentation()
            print(f"  {task.task_id}: {task.priority.name}, "
                  f"{len(segments)}个子段, FPS={task.fps_requirement}")
        else:
            print(f"  {task.task_id}: {task.priority.name}, "
                  f"不分段, FPS={task.fps_requirement}")
    
    # 模拟执行
    print("\n模拟执行100ms...")
    current_time = 0.0
    event_idx = 0
    
    while current_time < time_window and event_idx < len(plan.events):
        # 处理发射
        while event_idx < len(plan.events) and plan.events[event_idx].time <= current_time:
            event = plan.events[event_idx]
            task = launcher.tasks[event.task_id]
            
            # 处理分段任务
            if hasattr(task, 'apply_segmentation'):
                sub_segments = task.apply_segmentation()
                for sub_seg in sub_segments:
                    queue = queue_manager.find_best_queue(sub_seg.resource_type)
                    if queue:
                        sub_id = f"{event.task_id}#{event.instance_id}_{sub_seg.sub_id}"
                        queue.enqueue(sub_id, task.priority, current_time, [sub_seg])
                        tracer.record_enqueue(sub_id, queue.resource_id, 
                                            task.priority, current_time, [sub_seg])
            else:
                # 非分段任务
                queue = queue_manager.find_best_queue(ResourceType.NPU)
                if queue:
                    queue.enqueue(f"{event.task_id}#{event.instance_id}", 
                                task.priority, current_time, task.segments)
                    tracer.record_enqueue(f"{event.task_id}#{event.instance_id}",
                                        queue.resource_id, task.priority, 
                                        current_time, task.segments)
            
            event_idx += 1
        
        # 执行队列中的任务
        for queue_id, queue in queue_manager.resource_queues.items():
            queue.advance_time(current_time)
            
            if not queue.is_busy():
                next_task = queue.get_next_task()
                if next_task and next_task.ready_time <= current_time:
                    # 执行
                    seg = next_task.sub_segments[0] if next_task.sub_segments else None
                    if seg:
                        duration = seg.get_duration(queue.bandwidth)
                        end_time = current_time + duration
                        
                        tracer.record_execution(
                            next_task.task_id,
                            queue_id,
                            current_time,
                            end_time,
                            queue.bandwidth,
                            seg.sub_id if hasattr(seg, 'sub_id') else None
                        )
                        
                        queue.busy_until = end_time
                        queue.dequeue_task(next_task.task_id, next_task.priority)
        
        current_time += 0.5
    
    # 显示结果
    print("\n执行结果:")
    visualizer.print_gantt_chart(width=80)
    
    # 生成可视化
    visualizer.plot_resource_timeline("mixed_workload_segmentation.png")
    print("\n✓ 生成甘特图: mixed_workload_segmentation.png")
    
    # 分析
    stats = tracer.get_statistics()
    print(f"\n性能分析:")
    print(f"  平均等待时间: {stats['average_wait_time']:.2f}ms")
    print(f"  平均执行时间: {stats['average_execution_time']:.2f}ms")
    print(f"  总执行段数: {stats['total_executions']}")


if __name__ == "__main__":
    # 运行所有测试
    test_segmentation_with_launcher()
    test_segmentation_dependencies()
    test_mixed_segmentation_workload()
    
    print("\n\n✅ 所有分段测试完成！")
