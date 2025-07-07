#!/usr/bin/env python3
"""
调试为什么任务发射后没有执行
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ResourceType, TaskPriority, NNTask,
    ResourceQueueManager, ScheduleTracer, 
    TaskLauncher, ScheduleExecutor
)
from viz.schedule_visualizer import ScheduleVisualizer


def debug_simple_execution():
    """调试最简单的执行场景"""
    print("="*80)
    print("调试简单执行场景")
    print("="*80)
    
    # 1. 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    print("\n✓ 资源创建完成")
    print(f"  资源队列: {list(queue_manager.resource_queues.keys())}")
    
    # 2. 创建追踪器和发射器
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 3. 创建最简单的任务
    simple_task = NNTask("SIMPLE", "Simple Task", priority=TaskPriority.HIGH)
    simple_task.add_segment(ResourceType.NPU, {60: 5.0}, "main")
    simple_task.set_performance_requirements(fps=10, latency=100)
    
    print("\n✓ 任务创建完成")
    print(f"  任务: {simple_task.task_id}")
    print(f"  段数: {len(simple_task.segments)}")
    print(f"  第一段: {simple_task.segments[0].resource_type.value}, duration={simple_task.segments[0].duration_table}")
    
    # 4. 注册任务
    launcher.register_task(simple_task)
    
    # 5. 创建发射计划
    plan = launcher.create_launch_plan(50.0, "eager")
    
    print(f"\n✓ 发射计划创建完成")
    print(f"  事件数: {len(plan.events)}")
    for event in plan.events:
        print(f"  - {event.time}ms: {event.task_id}#{event.instance_id}")
    
    # 6. 创建执行器
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    # 添加调试输出
    print("\n开始调试执行过程...")
    
    # 手动执行第一步看看发生了什么
    if plan.events:
        first_event = plan.events[0]
        print(f"\n处理第一个发射事件: {first_event.task_id} at {first_event.time}ms")
        
        # 检查任务是否在executor.tasks中
        print(f"  executor.tasks包含: {list(executor.tasks.keys())}")
        
        # 手动调用处理
        executor._reset_state()
        executor._handle_launch_event(first_event)
        
        # 检查任务实例是否创建
        print(f"\n  任务实例创建: {list(executor.task_instances.keys())}")
        
        # 检查资源队列状态
        print("\n  资源队列状态:")
        for res_id, queue in queue_manager.resource_queues.items():
            print(f"    {res_id}:")
            print(f"      忙碌: {queue.is_busy()}")
            print(f"      当前任务: {queue.current_task}")
            
            # 检查各优先级队列
            for priority in TaskPriority:
                pq = queue.priority_queues[priority]
                if pq:
                    print(f"      {priority.name}队列: {len(pq)}个任务")
                    # 打印队列中的任务
                    for task in pq:
                        print(f"        - {task.task_id}")
        
        # 尝试调度
        print("\n  尝试调度就绪任务...")
        executor._schedule_ready_segments()
        
        # 再次检查队列状态
        print("\n  调度后的队列状态:")
        for res_id, queue in queue_manager.resource_queues.items():
            print(f"    {res_id}: 忙碌={queue.is_busy()}, 当前任务={queue.current_task}")
    
    # 7. 完整执行
    print("\n\n执行完整的调度计划...")
    executor._reset_state()
    stats = executor.execute_plan(plan, 50.0)
    
    print(f"\n执行统计:")
    print(f"  总实例数: {stats['total_instances']}")
    print(f"  完成实例: {stats['completed_instances']}")
    print(f"  执行段数: {stats['total_segments_executed']}")
    
    # 8. 显示结果
    visualizer = ScheduleVisualizer(tracer)
    print("\n甘特图:")
    visualizer.print_gantt_chart(width=60)
    
    # 检查tracer中的执行记录
    print(f"\n追踪器记录:")
    print(f"  执行记录数: {len(tracer.executions)}")
    stats = tracer.get_statistics()
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  总执行次数: {stats['total_executions']}")


def debug_launcher_issue():
    """专门调试launcher中的问题"""
    print("\n\n" + "="*80)
    print("调试Launcher问题")
    print("="*80)
    
    # 检查_launch_task方法
    from core.models import SubSegment, ResourceSegment
    
    # 创建一个测试段
    test_segment = ResourceSegment(
        resource_type=ResourceType.NPU,
        duration_table={60: 5.0},
        start_time=0.0,
        segment_id="test_seg"
    )
    
    print("\n测试SubSegment创建:")
    print(f"  原始段: {test_segment.segment_id}")
    print(f"  duration_table: {test_segment.duration_table}")
    
    # 正确的创建方式
    try:
        sub_seg = SubSegment(
            sub_id=test_segment.segment_id,
            resource_type=test_segment.resource_type,
            duration_table=test_segment.duration_table,
            cut_overhead=0.0,
            original_segment_id=test_segment.segment_id
        )
        print("  ✓ SubSegment创建成功")
        print(f"    sub_id: {sub_seg.sub_id}")
        print(f"    duration: {sub_seg.get_duration(60)}")
    except Exception as e:
        print(f"  ✗ SubSegment创建失败: {e}")


if __name__ == "__main__":
    debug_simple_execution()
    debug_launcher_issue()
