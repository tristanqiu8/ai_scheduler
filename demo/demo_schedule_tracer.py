#!/usr/bin/env python3
"""
演示 ScheduleTracer 的完整功能
包括甘特图和Chrome Tracing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enums import ResourceType, TaskPriority
from core.models import SubSegment


def demo_schedule_tracer():
    """演示调度追踪器的所有功能"""
    print("=== ScheduleTracer 功能演示 ===\n")
    
    # 1. 创建资源系统
    manager = ResourceQueueManager()
    
    # 添加多个资源
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    print("系统资源配置:")
    for rid, queue in sorted(manager.resource_queues.items()):
        print(f"  {rid}: {queue.resource_type.value}, 带宽={queue.bandwidth}")
    
    # 2. 创建追踪器
    tracer = ScheduleTracer(manager)
    
    # 3. 创建多样化的任务
    print("\n创建测试任务:")
    
    # CRITICAL任务 - 最高优先级
    critical_task = {
        "id": "CRITICAL_DETECT",
        "priority": TaskPriority.CRITICAL,
        "ready_time": 0.0,
        "resource_type": ResourceType.NPU,
        "segments": [
            SubSegment("detect_main", ResourceType.NPU, {60: 4.0}, 0.0, "detect")
        ]
    }
    
    # HIGH优先级任务 - 视频处理
    high_tasks = [
        {
            "id": "VIDEO_PROC_1",
            "priority": TaskPriority.HIGH,
            "ready_time": 0.0,
            "resource_type": ResourceType.DSP,
            "segments": [
                SubSegment("video_dsp", ResourceType.DSP, {40: 8.0}, 0.0, "video")
            ]
        },
        {
            "id": "VIDEO_PROC_2",
            "priority": TaskPriority.HIGH,
            "ready_time": 5.0,
            "resource_type": ResourceType.DSP,
            "segments": [
                SubSegment("video_dsp2", ResourceType.DSP, {40: 6.0}, 0.0, "video")
            ]
        }
    ]
    
    # NORMAL优先级任务 - 常规处理
    normal_tasks = []
    for i in range(3):
        task = {
            "id": f"NORMAL_TASK_{i+1}",
            "priority": TaskPriority.NORMAL,
            "ready_time": i * 2.0,
            "resource_type": ResourceType.NPU,
            "segments": [
                SubSegment(f"normal_{i}", ResourceType.NPU, {60: 5.0 + i}, 0.0, "normal")
            ]
        }
        normal_tasks.append(task)
    
    # LOW优先级任务 - 后台任务
    low_task = {
        "id": "BACKGROUND_CLEAN",
        "priority": TaskPriority.LOW,
        "ready_time": 0.0,
        "resource_type": ResourceType.NPU,
        "segments": [
            SubSegment("cleanup", ResourceType.NPU, {60: 10.0}, 0.0, "background")
        ]
    }
    
    # 合并所有任务
    all_tasks = [critical_task] + high_tasks + normal_tasks + [low_task]
    
    # 4. 分配任务到队列
    print("\n任务分配结果:")
    for task in all_tasks:
        # 根据资源类型找最佳队列
        queue = manager.find_best_queue(task["resource_type"])
        
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
            queue.enqueue(
                task["id"],
                task["priority"],
                task["ready_time"],
                task["segments"]
            )
            
            print(f"  {task['id']:<20} ({task['priority'].name:<8}) -> {queue.resource_id}")
    
    # 5. 模拟执行
    print("\n开始模拟执行...")
    print("-" * 60)
    
    current_time = 0.0
    max_time = 40.0
    time_step = 0.1
    
    while current_time < max_time:
        any_activity = False
        
        for resource_id, queue in sorted(manager.resource_queues.items()):
            queue.advance_time(current_time)
            
            # 如果资源空闲，尝试调度新任务
            if not queue.is_busy():
                next_task = queue.get_next_task()
                
                if next_task and next_task.ready_time <= current_time:
                    # 执行任务
                    start_time = current_time
                    end_time = queue.execute_task(next_task, start_time)
                    
                    # 获取段信息
                    segment = next_task.get_current_segment()
                    
                    # 记录执行
                    tracer.record_execution(
                        next_task.task_id,
                        resource_id,
                        start_time,
                        end_time,
                        queue.bandwidth,
                        segment.sub_id if segment else None
                    )
                    
                    print(f"{start_time:>6.1f} - {end_time:>6.1f}ms: {resource_id} 执行 "
                          f"{next_task.task_id} ({next_task.priority.name})")
                    
                    # 如果任务完成，从队列移除
                    if not next_task.has_remaining_segments():
                        queue.dequeue_task(next_task.task_id, next_task.priority)
                    
                    any_activity = True
        
        # 时间推进
        if not any_activity:
            # 找下一个事件时间
            next_event_time = max_time
            
            # 检查忙碌结束时间
            for queue in manager.resource_queues.values():
                if queue.is_busy() and queue.busy_until < next_event_time:
                    next_event_time = queue.busy_until
            
            # 检查任务就绪时间
            for queue in manager.resource_queues.values():
                for priority_queue in queue.priority_queues.values():
                    for task in priority_queue:
                        if task.ready_time > current_time and task.ready_time < next_event_time:
                            next_event_time = task.ready_time
            
            current_time = min(next_event_time, max_time)
        else:
            current_time += time_step
    
    print("-" * 60)
    print("模拟执行完成！\n")
    
    # 6. 显示甘特图
    tracer.print_gantt_chart(width=80)
    
    # 7. 显示详细统计
    print("\n" + "="*60)
    print("调度统计分析")
    print("="*60)
    
    stats = tracer.get_statistics()
    
    print(f"\n基本统计:")
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  时间跨度: {stats['time_span']:.1f}ms")
    
    print(f"\n性能指标:")
    print(f"  平均等待时间: {stats['average_wait_time']:.2f}ms")
    print(f"  平均执行时间: {stats['average_execution_time']:.2f}ms")
    
    print(f"\n任务优先级分布:")
    for priority, count in sorted(stats['tasks_by_priority'].items()):
        print(f"  {priority}: {count} 个任务")
    
    # 8. 导出Chrome Tracing
    output_file = "demo_schedule_trace.json"
    tracer.export_chrome_tracing(output_file)
    print(f"\n✅ 已导出Chrome Tracing文件: {output_file}")
    print("   在Chrome浏览器中打开 chrome://tracing 并加载此文件查看详细时间线")
    
    # 9. 显示任务执行详情
    print("\n" + "="*60)
    print("任务执行时间线详情")
    print("="*60)
    
    timeline = tracer.get_timeline()
    for resource_id in sorted(timeline.keys()):
        executions = timeline[resource_id]
        print(f"\n{resource_id}:")
        for exec in executions:
            print(f"  {exec.start_time:>6.1f} - {exec.end_time:>6.1f}ms: "
                  f"{exec.task_id:<20} (优先级: {exec.priority.name:<8}, "
                  f"带宽: {exec.bandwidth}, 时长: {exec.duration:.1f}ms)")


if __name__ == "__main__":
    demo_schedule_tracer()
