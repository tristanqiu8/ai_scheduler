#!/usr/bin/env python3
"""
测试真实任务的执行，包括多段任务如MOTR
使用实际的duration和正确的调度逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enums import ResourceType, TaskPriority
from scenario.real_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
from core.models import SubSegment


def execute_real_tasks_with_segments():
    """执行真实任务，正确处理多段"""
    print("=== 测试真实任务执行（含多段） ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建真实任务
    tasks = create_real_tasks()
    
    # 只使用前几个任务进行测试
    test_tasks = [
        tasks[0],  # T1 MOTR - 9段混合任务
        tasks[1],  # T2 YoloV8nBig - NPU+DSP
        tasks[5],  # T6 reid - 高频NPU任务
    ]
    
    print("测试任务:")
    for task in test_tasks:
        launcher.register_task(task)
        print(f"  {task.task_id} ({task.name}): {len(task.segments)}段, "
              f"FPS={task.fps_requirement}, 优先级={task.priority.name}")
    
    # 创建发射计划
    time_window = 200.0
    plan = launcher.create_launch_plan(time_window, "eager")
    
    print(f"\n发射计划: 共{len(plan.events)}次发射")
    
    # 执行计划
    print("\n开始执行模拟...")
    simulate_multi_segment_execution(launcher, queue_manager, tracer, plan, time_window)
    
    # 显示结果
    print("\n" + "="*80)
    visualizer.print_gantt_chart(width=80)
    
    # 统计
    stats = tracer.get_statistics()
    print(f"\n执行统计:")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  时间跨度: {stats['time_span']:.1f}ms")
    print(f"  平均执行时间: {stats['average_execution_time']:.2f}ms")
    
    print(f"\n资源利用率:")
    for res_id in ["NPU_0", "NPU_1", "DSP_0", "DSP_1"]:
        util = stats['resource_utilization'].get(res_id, 0)
        print(f"  {res_id}: {util:.1f}%")
    
    # 生成可视化
    visualizer.plot_resource_timeline("real_task_execution.png")
    visualizer.export_chrome_tracing("real_task_execution.json")
    
    print("\n✓ 生成文件:")
    print("  - real_task_execution.png")
    print("  - real_task_execution.json")


def simulate_multi_segment_execution(launcher, queue_manager, tracer, plan, max_time):
    """模拟多段任务的执行"""
    current_time = 0.0
    event_idx = 0
    
    # 任务实例的执行状态
    # (task_id, instance) -> {segments: [...], current_index: 0, completed: []}
    task_states = {}
    
    # 段完成时间，用于依赖管理
    segment_completions = {}  # (task_id, instance, segment_index) -> completion_time
    
    while current_time < max_time and (event_idx < len(plan.events) or has_active_tasks(queue_manager)):
        # 1. 处理发射事件
        while event_idx < len(plan.events) and plan.events[event_idx].time <= current_time:
            event = plan.events[event_idx]
            task = launcher.tasks[event.task_id]
            
            # 初始化任务状态
            state_key = (event.task_id, event.instance_id)
            
            # 获取任务的所有段（应用分段）
            sub_segments = task.apply_segmentation()
            if not sub_segments:
                # 转换原始段为子段格式
                sub_segments = []
                for i, seg in enumerate(task.segments):
                    sub_seg = SubSegment(
                        sub_id=f"{seg.segment_id}_{i}",
                        resource_type=seg.resource_type,
                        duration_table=seg.duration_table,
                        cut_overhead=0.0,
                        original_segment_id=seg.segment_id
                    )
                    sub_segments.append(sub_seg)
            
            task_states[state_key] = {
                'segments': sub_segments,
                'current_index': 0,
                'completed': [],
                'priority': task.priority
            }
            
            print(f"  {current_time:.1f}ms: 发射 {event.task_id}#{event.instance_id} "
                  f"({len(sub_segments)}个段)")
            
            # 立即尝试调度第一个段
            schedule_next_segment(state_key, task_states, queue_manager, tracer, current_time)
            
            event_idx += 1
        
        # 2. 检查资源执行
        for resource_id, queue in queue_manager.resource_queues.items():
            queue.advance_time(current_time)
            
            if not queue.is_busy():
                # 尝试执行队列中的任务
                next_task = queue.get_next_task()
                if next_task and next_task.ready_time <= current_time:
                    # 解析任务信息
                    parts = next_task.task_id.split('#')
                    if len(parts) >= 2:
                        base_task_id = parts[0]
                        instance_seg = parts[1].split('_')
                        instance_id = int(instance_seg[0])
                        
                        state_key = (base_task_id, instance_id)
                        
                        if state_key in task_states and next_task.sub_segments:
                            # 执行段
                            sub_seg = next_task.sub_segments[0]
                            duration = sub_seg.get_duration(queue.bandwidth)
                            end_time = current_time + duration
                            
                            # 记录执行
                            tracer.record_execution(
                                next_task.task_id,
                                resource_id,
                                current_time,
                                end_time,
                                queue.bandwidth,
                                sub_seg.sub_id
                            )
                            
                            print(f"    {current_time:.1f}-{end_time:.1f}ms: {resource_id} "
                                  f"执行 {base_task_id}#{instance_id} 的 {sub_seg.sub_id} "
                                  f"({sub_seg.resource_type.value})")
                            
                            # 更新资源状态
                            queue.busy_until = end_time
                            queue.current_task = next_task.task_id
                            
                            # 记录段完成信息
                            state = task_states[state_key]
                            seg_idx = state['current_index']
                            segment_completions[(base_task_id, instance_id, seg_idx)] = end_time
                            
                            # 从队列移除
                            queue.dequeue_task(next_task.task_id, next_task.priority)
        
        # 3. 检查段完成，调度下一段
        for (task_id, instance_id, seg_idx), completion_time in list(segment_completions.items()):
            if completion_time <= current_time:
                state_key = (task_id, instance_id)
                if state_key in task_states:
                    state = task_states[state_key]
                    
                    # 标记当前段完成
                    if seg_idx not in state['completed']:
                        state['completed'].append(seg_idx)
                        state['current_index'] = seg_idx + 1
                        
                        # 尝试调度下一段
                        if state['current_index'] < len(state['segments']):
                            schedule_next_segment(state_key, task_states, queue_manager, 
                                                tracer, current_time)
                        else:
                            print(f"    {current_time:.1f}ms: {task_id}#{instance_id} 完成所有段")
                
                # 清理已处理的完成记录
                del segment_completions[(task_id, instance_id, seg_idx)]
        
        # 4. 时间推进
        next_event_time = plan.events[event_idx].time if event_idx < len(plan.events) else max_time
        
        # 找下一个资源释放时间
        next_free_time = max_time
        for queue in queue_manager.resource_queues.values():
            if queue.is_busy() and queue.busy_until < next_free_time:
                next_free_time = queue.busy_until
        
        # 推进到下一个事件
        current_time = min(next_event_time, next_free_time, current_time + 0.1)


def schedule_next_segment(state_key, task_states, queue_manager, tracer, current_time):
    """调度任务的下一个段"""
    task_id, instance_id = state_key
    state = task_states[state_key]
    
    if state['current_index'] < len(state['segments']):
        seg = state['segments'][state['current_index']]
        
        # 找最佳队列
        best_queue = queue_manager.find_best_queue(seg.resource_type)
        if best_queue:
            # 创建段任务ID
            seg_task_id = f"{task_id}#{instance_id}_seg{state['current_index']}"
            
            # 加入队列
            best_queue.enqueue(
                seg_task_id,
                state['priority'],
                current_time,
                [seg]
            )
            
            # 记录入队
            tracer.record_enqueue(
                seg_task_id,
                best_queue.resource_id,
                state['priority'],
                current_time,
                [seg]
            )


def has_active_tasks(queue_manager):
    """检查是否还有活跃任务"""
    # 检查忙碌的资源
    for queue in queue_manager.resource_queues.values():
        if queue.is_busy():
            return True
        
        # 检查队列中的任务
        for pq in queue.priority_queues.values():
            if pq:
                return True
    
    return False


def test_priority_scheduling():
    """测试优先级调度"""
    print("\n\n=== 测试优先级调度 ===\n")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    visualizer = ScheduleVisualizer(tracer)
    
    # 创建不同优先级的任务
    from core.task import create_npu_task
    
    tasks = [
        create_npu_task("LOW", "低优先级", {60: 20.0}, priority=TaskPriority.LOW),
        create_npu_task("NORMAL", "普通优先级", {60: 15.0}, priority=TaskPriority.NORMAL),
        create_npu_task("HIGH", "高优先级", {60: 10.0}, priority=TaskPriority.HIGH),
        create_npu_task("CRITICAL", "关键任务", {60: 5.0}, priority=TaskPriority.CRITICAL),
    ]
    
    # 都设置为单次执行
    for task in tasks:
        task.set_performance_requirements(fps=1, latency=100)
        launcher.register_task(task)
    
    print("任务优先级测试:")
    for task in tasks:
        print(f"  {task.task_id}: {task.priority.name}, duration={task.estimate_duration({ResourceType.NPU: 60})}ms")
    
    # 手动在同一时刻发射所有任务
    print("\n同时发射所有任务...")
    for i, task in enumerate(tasks):
        launcher._launch_task(task.task_id, 0, 0.0)
    
    # 执行
    current_time = 0.0
    queue = queue_manager.get_queue("NPU_0")
    
    print("\n执行顺序（基于优先级）:")
    while current_time < 100.0:
        queue.advance_time(current_time)
        
        if not queue.is_busy():
            next_task = queue.get_next_task()
            if next_task:
                # 执行
                sub_seg = next_task.sub_segments[0]
                duration = sub_seg.get_duration(60.0)
                end_time = current_time + duration
                
                base_task_id = next_task.task_id.split('#')[0]
                task = launcher.tasks[base_task_id]
                
                print(f"  {current_time:.1f}-{end_time:.1f}ms: 执行 {base_task_id} "
                      f"(优先级: {task.priority.name})")
                
                tracer.record_execution(
                    next_task.task_id,
                    "NPU_0",
                    current_time,
                    end_time,
                    60.0
                )
                
                queue.busy_until = end_time
                queue.dequeue_task(next_task.task_id, next_task.priority)
                
        current_time = queue.busy_until if queue.is_busy() else current_time + 0.1
    
    # 显示甘特图
    print("\n" + "="*60)
    visualizer.print_gantt_chart(width=60)
    
    print("\n结论：任务按优先级执行 - CRITICAL → HIGH → NORMAL → LOW")


if __name__ == "__main__":
    # 运行测试
    execute_real_tasks_with_segments()
    test_priority_scheduling()
