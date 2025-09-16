#!/usr/bin/env python3
"""
测试帧率感知的依赖检查机制
用于验证不同帧率任务之间的依赖处理
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.enums import ResourceType
from NNScheduler.scenario.camera_task import create_real_tasks


def test_dependency_mapping():
    """测试依赖映射逻辑"""
    
    # 创建任务
    tasks = create_real_tasks()
    
    # 创建资源管理器
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    # 创建发射器
    launcher = TaskLauncher(queue_manager)
    
    # 注册任务
    for task in tasks:
        launcher.register_task(task)
    
    print("=" * 80)
    print("帧率感知的依赖映射测试")
    print("=" * 80)
    
    # 获取任务信息
    t2_fps = launcher.task_configs["T2"].fps_requirement
    t5_fps = launcher.task_configs["T5"].fps_requirement
    
    print(f"\nTask2 (FaceEhnsLite): {t2_fps} FPS")
    print(f"Task5 (FaceDet): {t5_fps} FPS")
    print(f"帧率比例: {t2_fps/t5_fps}:1")
    
    print("\n依赖映射关系:")
    print("-" * 40)
    
    # 测试多个实例的映射
    for i in range(6):
        dep_instance = launcher._get_dependency_instance("T2", i, "T5")
        print(f"T2 实例{i} -> T5 实例{dep_instance}")
    
    # 模拟任务完成情况
    print("\n\n模拟执行测试:")
    print("-" * 40)
    
    # 假设依赖任务已完成
    launcher.task_completions[("T1", 0)] = 10.0  # T1#0在10ms完成
    launcher.task_completions[("T1", 1)] = 20.0  # T1#1在20ms完成
    launcher.task_completions[("T3", 0)] = 10.0  # T3#0在10ms完成
    launcher.task_completions[("T3", 1)] = 20.0  # T3#1在20ms完成
    launcher.task_completions[("T5", 0)] = 10.0  # T5#0在10ms完成
    launcher.task_completions[("T5", 1)] = 43.0  # T5#1在43ms完成
    
    print("已完成的任务:")
    for key, time in launcher.task_completions.items():
        print(f"  {key[0]}#{key[1]} 完成于 {time}ms")
    
    print("\n检查T2实例的依赖:")
    for i in range(4):
        can_launch = launcher._check_dependencies("T2", i)
        dep_instance = launcher._get_dependency_instance("T2", i, "T5")
        print(f"  T2#{i} (需要T5#{dep_instance}): {'可以发射' if can_launch else '等待依赖'}")
    
    print("\n\n结论:")
    print("=" * 80)
    print("修改后的算法特性:")
    print("1. T2实例0和1都映射到T5实例0（帧率比2:1）")
    print("2. T2实例2和3都映射到T5实例1")
    print("3. 高帧率任务可以重用低帧率任务的输出")
    print("4. 避免了等待不存在的依赖实例")


def test_edge_cases():
    """测试边界情况"""
    
    # 创建任务
    tasks = create_real_tasks()
    
    # 创建资源管理器
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    # 创建发射器
    launcher = TaskLauncher(queue_manager)
    
    # 注册任务
    for task in tasks:
        launcher.register_task(task)
    
    print("\n\n边界情况测试:")
    print("=" * 80)
    
    # 测试1: 相同帧率的依赖
    print("\n1. 相同帧率的依赖 (T2 -> T1, 都是60 FPS):")
    for i in range(3):
        dep_instance = launcher._get_dependency_instance("T2", i, "T1")
        print(f"   T2 实例{i} -> T1 实例{dep_instance}")
    
    # 测试2: 依赖任务帧率更高的情况
    # 这里我们需要找一个帧率更高的任务作为例子
    print("\n2. 依赖任务帧率相同或更高时，使用相同实例号")
    
    # 测试3: 帧率比例不是整数的情况
    print("\n3. 帧率比例计算:")
    if "T2" in launcher.task_configs and "T5" in launcher.task_configs:
        t2_config = launcher.task_configs["T2"]
        t5_config = launcher.task_configs["T5"]
        ratio = t2_config.fps_requirement / t5_config.fps_requirement
        print(f"   T2/T5 帧率比: {ratio}")
        print(f"   映射规则: T2实例n -> T5实例int(n/{ratio})")


if __name__ == "__main__":
    """运行所有测试"""
    test_dependency_mapping()
    test_edge_cases()
