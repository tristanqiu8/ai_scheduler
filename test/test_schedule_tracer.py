#!/usr/bin/env python3
"""
测试 ScheduleTracer 功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enums import ResourceType, TaskPriority
from core.models import SubSegment


def test_schedule_tracer():
    """测试调度追踪器"""
    print("=== 测试 ScheduleTracer ===\n")
    
    # 创建资源队列管理器（静态带宽）
    manager = ResourceQueueManager()
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 创建追踪器
    tracer = ScheduleTracer(manager)
    
    # 创建测试任务
    tasks = [
        {
            "id": "TASK_1",
            "priority": TaskPriority.CRITICAL,
            "ready_time": 0.0,
            "segments": [SubSegment("t1_seg", ResourceType.NPU, {60: 5.0}, 0.0, "main")]
        },
        {
            "id": "TASK_2",
            "priority": TaskPriority.HIGH,
            "ready_time": 2.0,
            "segments": [SubSegment("t2_seg", ResourceType.NPU, {60: 8.0}, 0.0, "main")]
        },
        {
            "id": "TASK_3",
            "priority": TaskPriority.NORMAL,
            "ready_time": 0.0,
            "segments": [SubSegment("t3_seg", ResourceType.DSP, {40: 10.0}, 0.0, "main")]
        },
        {
            "id": "TASK_4",
            "priority": TaskPriority.NORMAL,
            "ready_time": 5.0,
            "segments": [SubSegment("t4_seg", ResourceType.NPU, {60: 6.0}, 0.0, "main")]
        }
    ]
    
    # 分配任务并记录
    print("任务分配:")
    for task in tasks:
        # 找到最佳队列
        queue = manager.find_best_queue(
            ResourceType.NPU if "NPU" in task["segments"][0].resource_type.value else ResourceType.DSP
        )
        
        if queue:
            # 记录入队
            tracer.record_enqueue(
                task["id"],
                queue.resource_id,
                task["priority"],
                task["ready_time"],
                task["segments"]
            )
            
            # 实际入队
            queue.enqueue(task["id"], task["priority"], task["ready_time"], task["segments"])
            print(f"  {task['id']} -> {queue.resource_id}")
    
    # 模拟执行
    print("\n模拟执行:")
    current_time = 0.0
    max_time = 30.0
    
    while current_time < max_time:
        for resource_id, queue in manager.resource_queues.items():
            queue.advance_time(current_time)
            
            if not queue.is_busy():
                next_task = queue.get_next_task()
                if next_task and next_task.ready_time <= current_time:
                    # 执行任务
                    start_time = current_time
                    end_time = queue.execute_task(next_task, start_time)
                    
                    # 记录执行
                    segment = next_task.get_current_segment()
                    tracer.record_execution(
                        next_task.task_id,
                        resource_id,
                        start_time,
                        end_time,
                        queue.bandwidth,
                        segment.sub_id if segment else None
                    )
                    
                    print(f"  {start_time:.1f}-{end_time:.1f}ms: {resource_id} 执行 {next_task.task_id}")
                    
                    # 移除已完成的任务
                    if not next_task.has_remaining_segments():
                        queue.dequeue_task(next_task.task_id, next_task.priority)
        
        # 时间步进
        current_time += 0.5
    
    # 显示结果
    tracer.print_gantt_chart(width=60)
    
    # 显示统计
    print("\n调度统计:")
    stats = tracer.get_statistics()
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  平均等待时间: {stats['average_wait_time']:.2f}ms")
    print(f"  平均执行时间: {stats['average_execution_time']:.2f}ms")
    
    # 导出Chrome追踪文件
    tracer.export_chrome_tracing("test_trace.json")
    print("\n已导出Chrome追踪文件: test_trace.json")


if __name__ == "__main__":
    test_schedule_tracer()
