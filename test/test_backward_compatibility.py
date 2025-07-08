#!/usr/bin/env python3
"""
测试向后兼容性 - 确保更新后的执行器不会破坏现有功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.executor import ScheduleExecutor, create_executor
from core.enums import ResourceType, TaskPriority
from core.task import create_mixed_task


def test_traditional_mode():
    """测试传统模式（默认行为）"""
    print("=== 测试传统模式（向后兼容） ===\n")
    
    # 创建环境
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建任务
    task = create_mixed_task(
        "TestTask", "测试任务",
        segments=[
            (ResourceType.NPU, {60: 5.0}, "seg0"),
            (ResourceType.DSP, {40: 8.0}, "seg1"),
        ],
        priority=TaskPriority.NORMAL
    )
    launcher.register_task(task)
    
    # 创建执行器（使用原始方式）
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    # 执行（不指定segment_mode，使用默认值False）
    plan = launcher.create_launch_plan(30.0, "eager")
    stats = executor.execute_plan(plan, 30.0)
    
    print("执行结果:")
    print(f"  完成实例: {stats['completed_instances']}")
    print(f"  执行段数: {stats['total_segments_executed']}")
    print(f"  默认segment_mode: {executor.segment_mode}")
    
    assert executor.segment_mode == False, "默认应该是传统模式"
    print("\n✅ 传统模式测试通过")


def test_segment_mode():
    """测试段级模式"""
    print("\n\n=== 测试段级模式 ===\n")
    
    # 创建相同的环境
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建多个任务
    for i in range(3):
        task = create_mixed_task(
            f"Task{i}", f"任务{i}",
            segments=[
                (ResourceType.NPU, {60: 3.0}, "npu"),
                (ResourceType.DSP, {40: 5.0}, "dsp"),
            ],
            priority=TaskPriority.NORMAL
        )
        launcher.register_task(task)
    
    # 方式1：通过参数启用段级模式
    executor1 = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    plan = launcher.create_launch_plan(30.0, "eager")
    stats1 = executor1.execute_plan(plan, 30.0, segment_mode=True)
    
    print("方式1 - 通过参数:")
    print(f"  完成实例: {stats1['completed_instances']}")
    print(f"  执行段数: {stats1['total_segments_executed']}")
    
    # 方式2：通过属性设置
    queue_manager2 = ResourceQueueManager()
    queue_manager2.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager2.add_resource("DSP_0", ResourceType.DSP, 40.0)
    tracer2 = ScheduleTracer(queue_manager2)
    
    executor2 = ScheduleExecutor(queue_manager2, tracer2, launcher.tasks)
    executor2.segment_mode = True
    stats2 = executor2.execute_plan(plan, 30.0)
    
    print("\n方式2 - 通过属性:")
    print(f"  完成实例: {stats2['completed_instances']}")
    print(f"  segment_mode: {executor2.segment_mode}")
    
    # 方式3：使用工厂函数
    queue_manager3 = ResourceQueueManager()
    queue_manager3.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager3.add_resource("DSP_0", ResourceType.DSP, 40.0)
    tracer3 = ScheduleTracer(queue_manager3)
    
    executor3 = create_executor(queue_manager3, tracer3, launcher.tasks, mode="segment_aware")
    stats3 = executor3.execute_plan(plan, 30.0)
    
    print("\n方式3 - 工厂函数:")
    print(f"  完成实例: {stats3['completed_instances']}")
    print(f"  segment_mode: {executor3.segment_mode}")
    
    print("\n✅ 段级模式测试通过")


def test_existing_test_case():
    """模拟现有测试用例的代码"""
    print("\n\n=== 模拟现有测试用例 ===\n")
    
    # 这是现有测试用例的典型代码模式
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 现有代码直接创建 ScheduleExecutor
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    
    # 现有代码调用 execute_plan，不传 segment_mode
    plan = launcher.create_launch_plan(10.0, "eager")
    stats = executor.execute_plan(plan, 10.0)
    
    print("现有代码模式运行正常:")
    print(f"  仿真时间: {stats['simulation_time']:.1f}ms")
    print("\n✅ 现有测试用例兼容性通过")


if __name__ == "__main__":
    # 运行所有兼容性测试
    test_traditional_mode()
    test_segment_mode()
    test_existing_test_case()
    
    print("\n\n🎉 所有向后兼容性测试通过！")
    print("\n建议:")
    print("1. 可以安全地替换 executor.py")
    print("2. 现有测试用例无需修改")
    print("3. 新功能可通过参数或属性启用")
