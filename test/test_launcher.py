#!/usr/bin/env python3
"""
测试任务发射器功能
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
from core.enums import ResourceType, TaskPriority
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer


def test_launcher_strategies():
    """测试不同的发射策略"""
    print("=== 测试任务发射器 ===\n")
    
    # 创建资源和组件
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建并注册任务
    tasks = create_real_tasks()
    for task in tasks[:6]:  # 使用前6个任务进行测试
        launcher.register_task(task)
        
    # 测试不同策略
    strategies = ["eager", "lazy", "balanced"]
    time_window = 100.0  # 100ms窗口
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"测试 {strategy.upper()} 策略")
        print(f"{'='*60}")
        
        # 创建发射计划
        plan = launcher.create_launch_plan(time_window, strategy)
        
        # 打印计划摘要
        print(f"\n发射计划摘要:")
        print(f"  总发射次数: {len(plan.events)}")
        
        # 按任务统计
        task_launches = {}
        for event in plan.events:
            if event.task_id not in task_launches:
                task_launches[event.task_id] = []
            task_launches[event.task_id].append(event.time)
            
        print(f"\n  任务发射时间表:")
        for task_id in sorted(task_launches.keys()):
            times = task_launches[task_id]
            task = launcher.tasks[task_id]
            print(f"    {task_id} ({task.name}): {len(times)}次发射")
            print(f"      时间点: {[f'{t:.1f}' for t in times[:5]]}", end="")
            if len(times) > 5:
                print(f" ... (共{len(times)}次)")
            else:
                print()
                
        # 分析发射密度
        print(f"\n  发射时间分布:")
        time_buckets = [0] * 10  # 10个时间段
        bucket_size = time_window / 10
        
        for event in plan.events:
            bucket = int(event.time / bucket_size)
            if bucket < 10:
                time_buckets[bucket] += 1
                
        for i, count in enumerate(time_buckets):
            start = i * bucket_size
            end = (i + 1) * bucket_size
            bar = '█' * (count // 2)
            print(f"    {start:3.0f}-{end:3.0f}ms: {bar} ({count})")


def test_dependency_handling():
    """测试依赖处理"""
    print("\n\n=== 测试依赖处理 ===\n")
    
    # 创建组件
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建有依赖关系的任务
    from core.task import create_npu_task
    
    # Task A: 独立任务
    task_a = create_npu_task(
        "A", "Independent",
        {60: 10.0},  # duration_table 作为位置参数
        priority=TaskPriority.NORMAL
    )
    task_a.set_performance_requirements(fps=10, latency=100)
    
    # Task B: 依赖于A
    task_b = create_npu_task(
        "B", "Depends on A",
        {60: 15.0},  # duration_table 作为位置参数
        priority=TaskPriority.NORMAL
    )
    task_b.add_dependency("A")  # 使用 add_dependency 方法
    task_b.set_performance_requirements(fps=10, latency=100)
    
    # Task C: 依赖于B
    task_c = create_npu_task(
        "C", "Depends on B",
        {60: 20.0},  # duration_table 作为位置参数
        priority=TaskPriority.NORMAL
    )
    task_c.add_dependency("B")  # 使用 add_dependency 方法
    task_c.set_performance_requirements(fps=10, latency=100)
    
    # 注册任务
    for task in [task_a, task_b, task_c]:
        launcher.register_task(task)
        
    # 创建发射计划
    plan = launcher.create_launch_plan(200.0, "eager")
    
    print("依赖关系:")
    print("  A -> B -> C")
    print("\n发射计划:")
    
    for event in plan.events[:10]:  # 显示前10个事件
        print(f"  {event.time:6.1f}ms: {event.task_id} (实例#{event.instance_id})")
        
    # 模拟执行和完成通知
    print("\n模拟执行:")
    
    # A完成后，B才能执行
    launcher.notify_task_completion("A", 0, 10.0)
    print("  10.0ms: A#0 完成，B#0 可以执行")
    
    launcher.notify_task_completion("B", 0, 25.0)
    print("  25.0ms: B#0 完成，C#0 可以执行")


def test_launch_execution():
    """测试实际发射执行"""
    print("\n\n=== 测试发射执行与可视化 ===\n")
    
    # 创建组件
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建简单任务集
    from core.task import create_npu_task, create_dsp_task
    
    tasks = [
        create_npu_task("T1", "高频任务", {60: 5.0}, priority=TaskPriority.HIGH),
        create_dsp_task("T2", "音频处理", {40: 8.0}, priority=TaskPriority.NORMAL),
        create_npu_task("T3", "低频任务", {60: 10.0}, priority=TaskPriority.LOW),
    ]
    
    # 设置不同的FPS要求
    tasks[0].set_performance_requirements(fps=30, latency=50)   # 高频
    tasks[1].set_performance_requirements(fps=20, latency=50)   # 中频
    tasks[2].set_performance_requirements(fps=5, latency=200)   # 低频
    
    for task in tasks:
        launcher.register_task(task)
        
    # 创建均衡发射计划
    time_window = 100.0
    plan = launcher.create_launch_plan(time_window, "balanced")
    
    print("执行发射计划...")
    
    # 改进的执行模拟
    current_time = 0.0
    event_idx = 0
    
    # 跟踪任务实例的段执行状态
    task_segment_status = {}  # (task_id, instance) -> current_segment_index
    
    while current_time < time_window and event_idx < len(plan.events):
        # 检查是否有发射事件
        while event_idx < len(plan.events) and plan.events[event_idx].time <= current_time:
            event = plan.events[event_idx]
            launcher._launch_task(event.task_id, event.instance_id, current_time)
            print(f"  {current_time:.1f}ms: 发射 {event.task_id}#{event.instance_id}")
            
            # 初始化任务段状态
            task_key = (event.task_id, event.instance_id)
            task_segment_status[task_key] = 0
            
            event_idx += 1
            
        # 模拟资源执行（使用实际duration）
        for queue_id, queue in queue_manager.resource_queues.items():
            queue.advance_time(current_time)
            
            if not queue.is_busy():
                next_task = queue.get_next_task()
                if next_task and next_task.ready_time <= current_time:
                    # 获取任务的实际段信息
                    task_base_id = next_task.task_id.split('#')[0]
                    task = launcher.tasks.get(task_base_id)
                    
                    if task and next_task.sub_segments:
                        # 使用子段的实际duration
                        sub_seg = next_task.sub_segments[0]
                        duration = sub_seg.get_duration(queue.bandwidth)
                        end_time = current_time + duration
                        
                        # 记录执行
                        tracer.record_execution(
                            next_task.task_id,
                            queue_id,
                            current_time,
                            end_time,
                            queue.bandwidth,
                            sub_seg.sub_id if hasattr(sub_seg, 'sub_id') else None
                        )
                        
                        print(f"    {current_time:.1f}ms: {queue_id} 执行 {next_task.task_id} "
                              f"(duration: {duration:.1f}ms)")
                        
                        # 标记资源忙碌
                        queue.busy_until = end_time
                        queue.current_task = next_task.task_id
                        
                        # 从队列移除
                        queue.dequeue_task(next_task.task_id, next_task.priority)
        
        # 时间推进
        current_time += 1.0
        
        # 更新资源状态
        for queue in queue_manager.resource_queues.values():
            queue.advance_time(current_time)
            
    # 显示结果
    print("\n执行结果:")
    visualizer.print_gantt_chart(width=80)
    
    # 生成可视化文件
    print("\n生成可视化文件...")
    visualizer.plot_resource_timeline("launcher_test.png")
    visualizer.export_chrome_tracing("launcher_test.json")
    
    print("\n✅ 测试完成！")
    print("  - launcher_test.png: Matplotlib甘特图")
    print("  - launcher_test.json: Chrome追踪文件")


if __name__ == "__main__":
    # 运行所有测试
    test_launcher_strategies()
    test_dependency_handling()
    test_launch_execution()
