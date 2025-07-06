#!/usr/bin/env python3
"""
静态带宽方案示例
每个资源有固定的带宽限制，从头到尾保持不变
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.enums import ResourceType, TaskPriority
from core.models import SubSegment
from core.task import create_npu_task, create_mixed_task


def setup_static_bandwidth_system():
    """设置静态带宽系统"""
    # 创建资源队列管理器（不使用动态带宽管理器）
    manager = ResourceQueueManager()
    
    # 添加资源，每个资源有固定的带宽
    # NPU资源
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)  # NPU_0: 60带宽
    manager.add_resource("NPU_1", ResourceType.NPU, 60.0)  # NPU_1: 60带宽
    
    # DSP资源
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)  # DSP_0: 40带宽
    manager.add_resource("DSP_1", ResourceType.DSP, 40.0)  # DSP_1: 40带宽
    
    print("系统配置（静态带宽）:")
    print("  NPU_0: 60 带宽（固定）")
    print("  NPU_1: 60 带宽（固定）")
    print("  DSP_0: 40 带宽（固定）")
    print("  DSP_1: 40 带宽（固定）")
    print("  总计: 200 带宽")
    
    return manager


def test_static_bandwidth_execution():
    """测试静态带宽下的任务执行"""
    manager = setup_static_bandwidth_system()
    
    print("\n=== 测试静态带宽执行 ===\n")
    
    # 创建任务
    # 注意：duration_table 需要包含对应的带宽值
    task1_segments = [
        SubSegment("conv_0", ResourceType.NPU, 
                  {40: 10.0, 60: 6.67, 120: 3.33},  # 60带宽时需要6.67ms
                  0.0, "conv")
    ]
    
    task2_segments = [
        SubSegment("process_0", ResourceType.DSP,
                  {40: 8.0, 60: 5.33, 120: 2.67},   # 40带宽时需要8ms
                  0.0, "process")
    ]
    
    # 场景1：NPU任务在NPU_0上执行
    print("场景1: NPU任务在NPU_0上执行")
    npu0_queue = manager.get_queue("NPU_0")
    npu0_queue.enqueue("T1", TaskPriority.HIGH, 0.0, task1_segments)
    
    task = npu0_queue.get_next_task()
    if task:
        segment = task.get_current_segment()
        # NPU_0的固定带宽是60
        duration = segment.get_duration(60.0)
        print(f"  任务: {task.task_id}")
        print(f"  资源: NPU_0 (带宽=60)")
        print(f"  执行时间: {duration:.2f}ms")
        
        # 执行任务
        end_time = npu0_queue.execute_task(task, 0.0)
        print(f"  实际结束时间: {end_time:.2f}ms")
    
    # 场景2：DSP任务在DSP_0上执行
    print("\n场景2: DSP任务在DSP_0上执行")
    dsp0_queue = manager.get_queue("DSP_0")
    dsp0_queue.enqueue("T2", TaskPriority.HIGH, 0.0, task2_segments)
    
    task = dsp0_queue.get_next_task()
    if task:
        segment = task.get_current_segment()
        # DSP_0的固定带宽是40
        duration = segment.get_duration(40.0)
        print(f"  任务: {task.task_id}")
        print(f"  资源: DSP_0 (带宽=40)")
        print(f"  执行时间: {duration:.2f}ms")
        
        # 执行任务
        end_time = dsp0_queue.execute_task(task, 0.0)
        print(f"  实际结束时间: {end_time:.2f}ms")
    
    # 场景3：并行执行
    print("\n场景3: NPU_0和NPU_1并行执行")
    print("  NPU_0和NPU_1各自使用自己的60带宽，互不影响")
    
    # 清空队列并获取NPU_1队列
    npu0_queue.clear()
    npu1_queue = manager.get_queue("NPU_1")
    
    # 推进时间到10ms
    npu0_queue.advance_time(10.0)
    npu1_queue.advance_time(10.0)
    
    # 两个NPU同时执行任务（ready_time=10.0）
    npu0_queue.enqueue("T3", TaskPriority.NORMAL, 10.0, task1_segments)
    npu1_queue.enqueue("T4", TaskPriority.NORMAL, 10.0, task1_segments)
    
    # NPU_0执行
    task3 = npu0_queue.get_next_task()
    if task3:
        end_time3 = npu0_queue.execute_task(task3, 10.0)
        npu0_queue.dequeue_task(task3.task_id, task3.priority)
        print(f"  T3在NPU_0: 10.0 - {end_time3:.2f}ms (带宽=60)")
    
    # NPU_1执行
    task4 = npu1_queue.get_next_task()
    if task4:
        end_time4 = npu1_queue.execute_task(task4, 10.0)
        npu1_queue.dequeue_task(task4.task_id, task4.priority)
        print(f"  T4在NPU_1: 10.0 - {end_time4:.2f}ms (带宽=60)")
    
    print(f"  两个任务并行执行，各自使用固定的60带宽")


def demonstrate_bandwidth_impact():
    """演示带宽对执行时间的影响"""
    print("\n=== 带宽对执行时间的影响 ===\n")
    
    # 创建一个对带宽敏感的任务
    segment = SubSegment(
        "compute_heavy",
        ResourceType.NPU,
        {
            20: 20.0,   # 20带宽需要20ms
            40: 10.0,   # 40带宽需要10ms
            60: 6.67,   # 60带宽需要6.67ms
            80: 5.0,    # 80带宽需要5ms
            100: 4.0,   # 100带宽需要4ms
            120: 3.33   # 120带宽需要3.33ms
        },
        0.0,
        "compute"
    )
    
    print("同一个任务在不同带宽资源上的执行时间:")
    print("任务工作量: 固定")
    print("\n带宽  执行时间  相对性能")
    print("-" * 30)
    
    for bandwidth in [20, 40, 60, 80, 100, 120]:
        duration = segment.get_duration(bandwidth)
        relative_perf = 20.0 / duration  # 相对于20带宽的性能
        print(f"{bandwidth:>4}  {duration:>7.2f}ms  {relative_perf:>6.2f}x")
    
    print("\n结论: 带宽越高，执行时间越短，呈反比关系")


if __name__ == "__main__":
    # 测试静态带宽执行
    test_static_bandwidth_execution()
    
    # 演示带宽影响
    demonstrate_bandwidth_impact()
    
    print("\n✅ 静态带宽方案演示完成！")
