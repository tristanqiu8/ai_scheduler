#!/usr/bin/env python3
"""
resource_queue.py 的单元测试
测试资源队列的各种功能
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from NNScheduler.core.resource_queue import ResourceQueue, ResourceQueueManager, QueuedTask
from NNScheduler.core.bandwidth_manager import BandwidthManager
from NNScheduler.core.enums import TaskPriority, ResourceType
from NNScheduler.core.models import SubSegment
from NNScheduler.core.task import NNTask, create_npu_task, create_mixed_task


def test_basic_queue_operations():
    """测试基本队列操作"""
    print("=== 测试基本队列操作 ===\n")
    
    # 创建资源队列
    queue = ResourceQueue("NPU_0", ResourceType.NPU, bandwidth=40.0)
    
    print(f"创建的队列: {queue.resource_id}")
    print(f"资源类型: {queue.resource_type.value}")
    print(f"固定带宽: {queue.bandwidth}")
    
    # 测试入队
    print("\n测试任务入队:")
    
    # 创建子段
    sub_segments = [
        SubSegment("seg_0", ResourceType.NPU, {40: 5.0, 120: 2.5}, 0.0, "main"),
        SubSegment("seg_1", ResourceType.NPU, {40: 3.0, 120: 1.5}, 0.0, "main"),
    ]
    
    # 入队不同优先级的任务
    queue.enqueue("T1", TaskPriority.HIGH, 0.0, sub_segments)
    queue.enqueue("T2", TaskPriority.NORMAL, 0.0, sub_segments[:1])
    queue.enqueue("T3", TaskPriority.HIGH, 0.0, sub_segments[:1])
    queue.enqueue("T4", TaskPriority.CRITICAL, 0.0, sub_segments)
    
    # 检查队列长度
    queue_lengths = queue.get_queue_length()
    print("各优先级队列长度:")
    for priority, length in queue_lengths.items():
        if length > 0:
            print(f"  {priority.name}: {length}")
    
    # 测试获取下一个任务（按优先级）
    print("\n按优先级获取任务:")
    for i in range(4):
        next_task = queue.get_next_task()
        if next_task:
            print(f"  第{i+1}个: {next_task.task_id} (优先级: {next_task.priority.name})")
            # 执行任务
            end_time = queue.execute_task(next_task, queue.current_time)
            print(f"    执行时间: {queue.current_time:.1f} - {end_time:.1f}ms")
            # 从队列移除
            queue.dequeue_task(next_task.task_id, next_task.priority)
            # 推进时间
            queue.advance_time(end_time)
    
    # 测试资源利用率
    print(f"\n资源利用率: {queue.get_utilization(queue.current_time):.1f}%")
    print(f"总执行任务数: {queue.total_tasks_executed}")
    print(f"总忙碌时间: {queue.total_busy_time:.1f}ms")


def test_priority_and_fifo():
    """测试优先级和FIFO顺序"""
    print("\n=== 测试优先级和FIFO顺序 ===\n")
    
    queue = ResourceQueue("NPU_0", ResourceType.NPU, bandwidth=40.0)
    
    # 创建简单子段
    sub_seg = [SubSegment("seg", ResourceType.NPU, {40: 2.0}, 0.0, "main")]
    
    # 按顺序入队相同优先级的任务
    print("入队3个NORMAL优先级任务:")
    for i in range(3):
        queue.enqueue(f"N{i+1}", TaskPriority.NORMAL, 0.0, sub_seg)
        print(f"  入队 N{i+1}")
    
    # 入队一个HIGH优先级任务
    print("\n入队1个HIGH优先级任务:")
    queue.enqueue("H1", TaskPriority.HIGH, 0.0, sub_seg)
    print("  入队 H1")
    
    # 再入队NORMAL任务
    print("\n再入队1个NORMAL优先级任务:")
    queue.enqueue("N4", TaskPriority.NORMAL, 0.0, sub_seg)
    print("  入队 N4")
    
    # 获取执行顺序
    print("\n执行顺序（应该是 H1 -> N1 -> N2 -> N3 -> N4）:")
    execution_order = []
    while True:
        task = queue.get_next_task()
        if not task:
            break
        execution_order.append(task.task_id)
        queue.dequeue_task(task.task_id, task.priority)
    
    print("  实际顺序:", " -> ".join(execution_order))
    
    # 验证FIFO
    normal_tasks = [t for t in execution_order if t.startswith('N')]
    print(f"\nNORMAL任务的FIFO顺序: {' -> '.join(normal_tasks)}")
    assert normal_tasks == ['N1', 'N2', 'N3', 'N4'], "FIFO顺序错误！"
    print("✅ FIFO顺序正确")


def test_resource_busy_state():
    """测试资源忙碌状态"""
    print("\n=== 测试资源忙碌状态 ===\n")
    
    queue = ResourceQueue("DSP_0", ResourceType.DSP, bandwidth=40.0)
    
    # 创建不同时长的子段
    short_seg = [SubSegment("short", ResourceType.DSP, {40: 5.0}, 0.0, "main")]
    long_seg = [SubSegment("long", ResourceType.DSP, {40: 20.0}, 0.0, "main")]
    
    # 入队任务
    queue.enqueue("T1", TaskPriority.HIGH, 0.0, short_seg)
    queue.enqueue("T2", TaskPriority.HIGH, 0.0, long_seg)
    queue.enqueue("T3", TaskPriority.NORMAL, 0.0, short_seg)
    
    print("初始状态:")
    print(f"  当前时间: {queue.current_time:.1f}ms")
    print(f"  资源忙碌: {queue.is_busy()}")
    
    # 执行第一个任务
    task1 = queue.get_next_task()
    end_time1 = queue.execute_task(task1, 0.0)
    queue.dequeue_task(task1.task_id, task1.priority)
    
    print(f"\n执行 {task1.task_id} 后:")
    print(f"  忙碌直到: {queue.busy_until:.1f}ms")
    print(f"  资源忙碌: {queue.is_busy()}")
    print(f"  当前任务: {queue.current_task}")
    
    # 时间推进到任务中间
    queue.advance_time(2.5)
    print(f"\n时间推进到 {queue.current_time:.1f}ms:")
    print(f"  资源忙碌: {queue.is_busy()}")
    
    # 时间推进到任务结束
    queue.advance_time(5.0)
    print(f"\n时间推进到 {queue.current_time:.1f}ms:")
    print(f"  资源忙碌: {queue.is_busy()}")
    print(f"  当前任务: {queue.current_task}")
    
    # 获取下次可用时间
    print(f"\n下次可用时间: {queue.get_next_available_time():.1f}ms")


def test_multi_segment_task():
    """测试多段任务的执行"""
    print("\n=== 测试多段任务执行 ===\n")
    
    queue = ResourceQueue("NPU_0", ResourceType.NPU, bandwidth=40.0)
    
    # 创建多段任务
    segments = [
        SubSegment("seg_0", ResourceType.NPU, {40: 5.0}, 0.0, "main"),
        SubSegment("seg_1", ResourceType.NPU, {40: 3.0}, 0.0, "main"),
        SubSegment("seg_2", ResourceType.NPU, {40: 2.0}, 0.0, "main"),
    ]
    
    # 入队任务
    queue.enqueue("T1", TaskPriority.NORMAL, 0.0, segments)
    
    print(f"任务T1有 {len(segments)} 个段")
    
    # 逐段执行
    task = queue.get_next_task()
    segment_times = []
    
    while task and task.has_remaining_segments():
        current_seg = task.get_current_segment()
        start_time = queue.current_time
        end_time = queue.execute_task(task, start_time)
        
        segment_times.append({
            'segment': current_seg.sub_id,
            'start': start_time,
            'end': end_time,
            'duration': end_time - start_time
        })
        
        print(f"\n执行段 {current_seg.sub_id}:")
        print(f"  时间: {start_time:.1f} - {end_time:.1f}ms")
        print(f"  时长: {end_time - start_time:.1f}ms")
        print(f"  剩余段数: {len(task.sub_segments) - task.current_segment_index}")
        
        queue.advance_time(end_time)
        
        # 如果任务还有剩余段，重新获取（模拟调度器行为）
        if not task.has_remaining_segments():
            queue.dequeue_task(task.task_id, task.priority)
            task = None
    
    # 打印执行摘要
    print("\n执行摘要:")
    total_time = sum(s['duration'] for s in segment_times)
    print(f"  总执行时间: {total_time:.1f}ms")
    print(f"  段数: {len(segment_times)}")


def test_resource_queue_manager():
    """测试资源队列管理器"""
    print("\n=== 测试资源队列管理器 ===\n")
    
    # 创建管理器
    manager = ResourceQueueManager()
    
    # 添加资源
    manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    print("添加的资源:")
    for res_id, queue in manager.resource_queues.items():
        print(f"  {res_id}: {queue.resource_type.value}, 带宽={queue.bandwidth}")
    
    # 创建任务并分配到队列
    tasks = [
        ("T1", TaskPriority.HIGH, ResourceType.NPU),
        ("T2", TaskPriority.NORMAL, ResourceType.NPU),
        ("T3", TaskPriority.HIGH, ResourceType.DSP),
        ("T4", TaskPriority.NORMAL, ResourceType.NPU),
    ]
    
    sub_seg = [SubSegment("seg", ResourceType.NPU, {40: 5.0, 60: 3.5}, 0.0, "main")]
    
    print("\n分配任务到最佳资源:")
    for task_id, priority, res_type in tasks:
        # 找到最佳队列
        best_queue = manager.find_best_queue(res_type)
        if best_queue:
            # 调整子段的资源类型
            task_segments = [SubSegment("seg", res_type, {40: 5.0, 60: 3.5}, 0.0, "main")]
            best_queue.enqueue(task_id, priority, 0.0, task_segments)
            print(f"  {task_id} -> {best_queue.resource_id}")
    
    # 获取NPU队列
    print("\nNPU队列状态:")
    npu_queues = manager.get_queues_by_type(ResourceType.NPU)
    for queue in npu_queues:
        lengths = queue.get_queue_length()
        total = sum(lengths.values())
        print(f"  {queue.resource_id}: {total} 个任务")
    
    # 获取全局统计
    stats = manager.get_global_stats()
    print("\n全局统计:")
    print(f"  总执行任务数: {stats['total_tasks_executed']}")
    print(f"  队列长度: {stats['queue_lengths']}")


def test_dynamic_bandwidth():
    """测试动态带宽功能"""
    print("\n=== 测试动态带宽功能 ===\n")
    
    # 创建带宽管理器
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    
    # 创建支持动态带宽的队列管理器
    manager = ResourceQueueManager(bandwidth_manager=bw_manager)
    
    # 添加资源（带宽设为0，因为会动态分配）
    npu0 = manager.add_resource("NPU_0", ResourceType.NPU, 0.0)
    npu1 = manager.add_resource("NPU_1", ResourceType.NPU, 0.0)
    dsp0 = manager.add_resource("DSP_0", ResourceType.DSP, 0.0)
    
    print("资源配置（动态带宽模式）:")
    print(f"  系统总带宽: {bw_manager.total_system_bandwidth}")
    print(f"  资源: NPU_0, NPU_1, DSP_0")
    
    # 创建具有不同带宽-时长映射的子段
    npu_seg = SubSegment(
        "npu_seg",
        ResourceType.NPU,
        {30: 12.0, 40: 9.0, 60: 6.0, 120: 3.0},  # 带宽越高，执行越快
        0.0,
        "main"
    )
    
    dsp_seg = SubSegment(
        "dsp_seg",
        ResourceType.DSP,
        {30: 8.0, 40: 6.0, 60: 4.0, 120: 2.0},
        0.0,
        "main"
    )
    
    # 场景1：单个NPU任务
    print("\n场景1: 只有NPU_0执行任务")
    npu0.enqueue("T1", TaskPriority.NORMAL, 0.0, [npu_seg])
    task1 = npu0.get_next_task()
    
    # 执行时NPU_0应该获得全部120带宽
    end_time1 = npu0.execute_task(task1, 0.0)
    print(f"  NPU_0执行T1:")
    print(f"    分配的带宽: {npu0.last_used_bandwidth:.1f}")
    print(f"    执行时间: 0.0 - {end_time1:.1f}ms")
    print(f"    预期时间(120带宽): {npu_seg.get_duration(120):.1f}ms")
    
    # 场景2：两个NPU并行
    print("\n场景2: NPU_0和NPU_1并行执行")
    npu0.advance_time(10.0)
    npu1.advance_time(10.0)
    
    npu0.enqueue("T2", TaskPriority.NORMAL, 10.0, [npu_seg])
    npu1.enqueue("T3", TaskPriority.NORMAL, 10.0, [npu_seg])
    
    task2 = npu0.get_next_task()
    task3 = npu1.get_next_task()
    
    # 同时执行，应该各获得60带宽
    end_time2 = npu0.execute_task(task2, 10.0)
    end_time3 = npu1.execute_task(task3, 10.0)
    
    print(f"  NPU_0执行T2:")
    print(f"    分配的带宽: {npu0.last_used_bandwidth:.1f}")
    print(f"    执行时间: 10.0 - {end_time2:.1f}ms")
    print(f"  NPU_1执行T3:")
    print(f"    分配的带宽: {npu1.last_used_bandwidth:.1f}")
    print(f"    执行时间: 10.0 - {end_time3:.1f}ms")
    print(f"  预期时间(60带宽): {npu_seg.get_duration(60):.1f}ms")
    
    # 场景3：NPU和DSP混合
    print("\n场景3: NPU和DSP同时执行")
    npu0.advance_time(20.0)
    dsp0.advance_time(20.0)
    
    npu0.enqueue("T4", TaskPriority.NORMAL, 20.0, [npu_seg])
    dsp0.enqueue("T5", TaskPriority.NORMAL, 20.0, [dsp_seg])
    
    task4 = npu0.get_next_task()
    task5 = dsp0.get_next_task()
    
    # NPU和DSP平分带宽
    end_time4 = npu0.execute_task(task4, 20.0)
    end_time5 = dsp0.execute_task(task5, 20.0)
    
    print(f"  NPU_0执行T4:")
    print(f"    分配的带宽: {npu0.last_used_bandwidth:.1f}")
    print(f"    执行时间: 20.0 - {end_time4:.1f}ms")
    print(f"  DSP_0执行T5:")
    print(f"    分配的带宽: {dsp0.last_used_bandwidth:.1f}")
    print(f"    执行时间: 20.0 - {end_time5:.1f}ms")
    
    # 获取系统状态
    system_status = bw_manager.get_system_status(25.0)
    print(f"\n时间25ms时的系统状态:")
    print(f"  活跃资源: {system_status['active_resources']['list']}")
    print(f"  每单元带宽: {system_status['bandwidth_per_unit']:.1f}")


def test_ready_time_constraint():
    """测试就绪时间约束"""
    print("\n=== 测试就绪时间约束 ===\n")
    
    queue = ResourceQueue("NPU_0", ResourceType.NPU, bandwidth=40.0)
    sub_seg = [SubSegment("seg", ResourceType.NPU, {40: 5.0}, 0.0, "main")]
    
    # 入队不同就绪时间的任务
    queue.enqueue("T1", TaskPriority.HIGH, 0.0, sub_seg)    # 立即就绪
    queue.enqueue("T2", TaskPriority.HIGH, 10.0, sub_seg)   # 10ms后就绪
    queue.enqueue("T3", TaskPriority.HIGH, 5.0, sub_seg)    # 5ms后就绪
    
    print("入队的任务:")
    print("  T1: ready_time=0ms")
    print("  T2: ready_time=10ms")
    print("  T3: ready_time=5ms")
    
    # 在不同时间点检查可执行任务
    time_points = [0, 4, 6, 11]
    
    for time in time_points:
        queue.advance_time(time)
        print(f"\n时间 {time}ms:")
        
        # 临时保存队列状态
        temp_tasks = []
        
        # 获取所有就绪任务
        while True:
            task = queue.get_next_task()
            if not task:
                break
            print(f"  可执行: {task.task_id} (ready_time={task.ready_time}ms)")
            temp_tasks.append(task)
            queue.dequeue_task(task.task_id, task.priority)
        
        # 恢复队列
        for task in temp_tasks:
            queue.priority_queues[task.priority].append(task)


def test_real_scenario():
    """测试真实场景"""
    print("\n=== 测试真实场景 ===\n")
    
    # 创建带宽管理器和队列管理器
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    manager = ResourceQueueManager(bandwidth_manager=bw_manager)
    
    # 添加资源
    manager.add_resource("NPU_0", ResourceType.NPU, 0.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 0.0)
    
    # 创建真实任务
    print("创建真实任务:")
    
    # MOTR任务 - 混合资源
    motr = create_mixed_task("T1", "MOTR", [
        (ResourceType.NPU, {40: 0.410, 120: 0.249}, "npu_s0"),
        (ResourceType.DSP, {40: 1.2}, "dsp_s0"),
        (ResourceType.NPU, {40: 9.333, 120: 5.147}, "npu_s1"),
    ], priority=TaskPriority.CRITICAL)
    
    # YOLO任务 - 纯NPU
    yolo = create_npu_task("T2", "YOLO", {40: 12.71, 120: 6.35}, 
                          priority=TaskPriority.NORMAL)
    
    print(f"  {motr.task_id}: {motr.name} (段数: {len(motr.segments)})")
    print(f"  {yolo.task_id}: {yolo.name} (段数: {len(yolo.segments)})")
    
    # 将任务段转换为子段并入队
    print("\n任务入队:")
    
    # MOTR的段
    motr_segments = motr.apply_segmentation()
    # 将NPU段入队到NPU_0
    npu_segments = [s for s in motr_segments if s.resource_type == ResourceType.NPU]
    if npu_segments:
        manager.resource_queues["NPU_0"].enqueue(motr.task_id, motr.priority, 0.0, npu_segments)
        print(f"  {motr.task_id} NPU段 -> NPU_0")
    
    # 将DSP段入队到DSP_0
    dsp_segments = [s for s in motr_segments if s.resource_type == ResourceType.DSP]
    if dsp_segments:
        manager.resource_queues["DSP_0"].enqueue(motr.task_id, motr.priority, 0.0, dsp_segments)
        print(f"  {motr.task_id} DSP段 -> DSP_0")
    
    # YOLO入队
    yolo_segments = yolo.apply_segmentation()
    manager.resource_queues["NPU_0"].enqueue(yolo.task_id, yolo.priority, 0.0, yolo_segments)
    print(f"  {yolo.task_id} -> NPU_0")
    
    # 模拟执行
    print("\n模拟执行:")
    current_time = 0.0
    max_time = 30.0
    events = []
    
    while current_time < max_time:
        executed = False
        
        # 检查每个资源队列
        for res_id, queue in manager.resource_queues.items():
            if not queue.is_busy() or queue.busy_until <= current_time:
                task = queue.get_next_task()
                if task and task.ready_time <= current_time:
                    # 执行任务
                    start_time = current_time
                    end_time = queue.execute_task(task, start_time)
                    
                    seg = task.sub_segments[task.current_segment_index - 1]
                    events.append({
                        'resource': res_id,
                        'task': task.task_id,
                        'segment': seg.sub_id,
                        'start': start_time,
                        'end': end_time,
                        'bandwidth': queue.last_used_bandwidth
                    })
                    
                    print(f"  {start_time:.1f}-{end_time:.1f}ms: "
                          f"{res_id}执行{task.task_id}.{seg.sub_id} "
                          f"(带宽: {queue.last_used_bandwidth:.0f})")
                    
                    # 如果任务完成，从队列移除
                    if not task.has_remaining_segments():
                        queue.dequeue_task(task.task_id, task.priority)
                    
                    executed = True
        
        # 推进时间
        if not executed:
            current_time += 0.5
        
        manager.advance_all_queues(current_time)
    
    # 打印执行统计
    print("\n执行统计:")
    for res_id, queue in manager.resource_queues.items():
        util = queue.get_utilization(current_time)
        print(f"  {res_id}: 利用率={util:.1f}%, 执行任务数={queue.total_tasks_executed}")


def main():
    """运行所有测试"""
    print("开始测试 resource_queue.py\n")
    
    test_basic_queue_operations()
    print("\n" + "="*60)
    
    test_priority_and_fifo()
    print("\n" + "="*60)
    
    test_resource_busy_state()
    print("\n" + "="*60)
    
    test_multi_segment_task()
    print("\n" + "="*60)
    
    test_resource_queue_manager()
    print("\n" + "="*60)
    
    test_dynamic_bandwidth()
    print("\n" + "="*60)
    
    test_ready_time_constraint()
    print("\n" + "="*60)
    
    test_real_scenario()
    
    print("\n\n✅ 所有测试完成！")


if __name__ == "__main__":
    main()
