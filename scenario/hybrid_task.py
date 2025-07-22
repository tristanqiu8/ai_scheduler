#!/usr/bin/env python3
"""
真实任务定义 - 使用精简后的接口（无 start_time）
"""

from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.task import NNTask, create_npu_task, create_dsp_task, create_mixed_task
from scenario.model_repo import get_model


def create_real_tasks():
    """创建测试任务集"""
    
    tasks = []
    
    print("\n📋 创建测试任务:")
    
    fps_table = {"Parsing": 60,
                 "ReID": 25,
                 "MOTR": 25,
                 "qim": 25,
                 "pose2d": 25,
                 "tk_template": 5,
                 "tk_search": 25,
                 "GrayMask": 10, 
                 "Yolov8nBig": 8,
                 "Yolov8nSmall": 8,
                 "Stereo4x": 10,
                 "Skywater": 10,
                 "PeakDetector": 10,
                 "Skywater_Big1": 10,
                 "Skywater_Big2": 10,
                 "Skywater_Big3": 10,
                 "BonusTask": 10,
                 }
    
    # 任务1: 3A Parsing
    task1 = NNTask(
        "T1", "Parsing",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    # 从model_lib获取模型定义并应用
    task1.apply_model(get_model("parsing"))
    task1.set_performance_requirements(fps=fps_table[task1.name], latency=1000.0/fps_table[task1.name])
    tasks.append(task1)
    print("  ✓ T1 Parsing: 3A中频NPU+DSP任务")
    
    # 任务2: 重识别（高频任务）
    task2 = NNTask(
        "T2", "ReID",
        priority=TaskPriority.LOW,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task2.apply_model(get_model("reid"))
    task2.set_performance_requirements(fps=fps_table[task2.name], latency=50.0)
    tasks.append(task2)
    print("  ✓ T2 ReID: 高频NPU任务")
    
    # 任务3: MOTR - 多目标跟踪（关键任务）
    task3 = NNTask(
        "T3", "MOTR",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task3.apply_model(get_model("motr"))
    task3.set_performance_requirements(fps=fps_table[task3.name], latency=1000.0/fps_table[task3.name])
    tasks.append(task3)
    print("  ✓ T3 MOTR: 9段混合任务 (4 DSP + 5 NPU)")
    
    # 任务4: motr post处理 - qim
    task4 = NNTask(
        "T4", "qim",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task4.apply_model(get_model("qim"))
    task4.set_performance_requirements(fps=fps_table[task4.name], latency=1000.0/fps_table[task4.name])
    task4.add_dependency("T3")  # 依赖MOTR
    tasks.append(task4)
    print("  ✓ T4 qim: DSP+NPU混合任务 (依赖T3)")
    
    # 任务5: 2D姿态估计
    task5 = NNTask(
        "T5", "pose2d",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task5.apply_model(get_model("pose2d"))
    task5.set_performance_requirements(fps=fps_table[task5.name], latency=1000.0/fps_table[task5.name])
    task5.add_dependency("T3")  # 依赖MOTR的检测结果
    tasks.append(task5)
    print("  ✓ T5 pose2d: NPU任务 (依赖T3)")
    
    # 任务6: 模板匹配
    task6 = NNTask(
        "T6", "tk_template",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task6.apply_model(get_model("tk_template"))
    task6.set_performance_requirements(fps=fps_table[task6.name], latency=1000.0/fps_table[task6.name])
    tasks.append(task6)
    print("  ✓ T6 tk_temp: 纯NPU任务")
    
    # 任务7: 搜索任务
    task7 = NNTask(
        "T7", "tk_search",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task7.apply_model(get_model("tk_search"))
    task7.set_performance_requirements(fps=fps_table[task7.name], latency=1000.0/fps_table[task7.name])
    tasks.append(task7)
    print("  ✓ T7 tk_search: 纯NPU任务")
    
    # 任务8：灰度Mask
    task8 = NNTask(
        "T8", "GrayMask",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task8.apply_model(get_model("graymask"))
    task8.set_performance_requirements(fps=fps_table[task8.name], latency=1000.0/fps_table[task8.name])
    tasks.append(task8)
    print("  ✓ T8 GrayMask: 纯NPU任务")
        
    # 任务9: YOLOv8n 大模型
    task9 = NNTask(
        "T9", "Yolov8nBig",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task9.apply_model(get_model("yolov8n_big"))
    task9.set_performance_requirements(fps=fps_table[task9.name], latency=1000.0/fps_table[task9.name])
    tasks.append(task9)
    print("  ✓ T9 YoloV8nBig: 可分段NPU任务")
    
    # 任务10: YOLOv8n 小模型
    task10 = NNTask(
        "T10", "Yolov8nSmall",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task10.apply_model(get_model("yolov8n_small"))
    task10.set_performance_requirements(fps=fps_table[task10.name], latency=1000.0/fps_table[task10.name])
    tasks.append(task10)
    print("  ✓ T10 YoloV8nSmall: 可分段NPU任务")
    
    # 任务11: Stereo4x - 双目深度（关键任务）
    task11 = NNTask(
        "T11", "Stereo4x",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task11.apply_model(get_model("stereo4x"))
    task11.set_performance_requirements(fps=fps_table[task11.name], latency=65.0)
    tasks.append(task11)
    print("  ✓ T11 Stereo4x: 8段混合任务 (3 DSP + 5 NPU)")
    
    # 任务12: Skywater 小模型
    task12 = NNTask(
        "T12", "Skywater",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task12.apply_model(get_model("skywater"))
    task12.set_performance_requirements(fps=fps_table[task12.name], latency=100.0)
    tasks.append(task12)
    print("  ✓ T12 Skywater: 可分段NPU+DSP任务")
    
    # 任务13: PeakDetector
    task13 = NNTask(
        "T13", "PeakDetector",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task13.apply_model(get_model("peak_detector"))
    task13.set_performance_requirements(fps=fps_table[task13.name], latency=1000.0/fps_table[task13.name])
    tasks.append(task13)
    print("  ✓ T13 PeakDetector: 纯NPU任务")
    
    # 任务14: Skywater 大模型
    task14 = NNTask(
        "T14", "Skywater_Big1",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task14.apply_model(get_model("skywater_big"))
    task14.set_performance_requirements(fps=fps_table[task14.name], latency=34.0)
    tasks.append(task14)
    print("  ✓ T14 Skywater Mono: 可分段NPU+DSP任务")
    
    # 任务15: Skywater 大模型
    task15 = NNTask(
        "T15", "Skywater_Big2",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task15.apply_model(get_model("skywater_big"))
    task15.set_performance_requirements(fps=fps_table[task15.name], latency=34.0)
    tasks.append(task15)
    print("  ✓ T15 Skywater Mono: 可分段NPU+DSP任务")
    
    # 任务16: Skywater 大模型
    task16 = NNTask(
        "T16", "Skywater_Big3",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task16.apply_model(get_model("skywater_big"))
    task16.set_performance_requirements(fps=fps_table[task16.name], latency=34.0)
    tasks.append(task16)
    print("  ✓ T16 Skywater Mono3: 可分段NPU+DSP任务")

    # 任务17: 模板匹配
    task17 = NNTask(
        "T17", "BonusTask",
        priority=TaskPriority.LOW,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task17.apply_model(get_model("bonus_task"))
    task17.set_performance_requirements(fps=fps_table[task17.name], latency=1000.0/fps_table[task17.name])
    tasks.append(task17)
    print("  ✓ T17 BonusTask: 奖励任务")
    
    return tasks


def print_task_summary(tasks):
    """打印任务摘要"""
    print("\n📊 任务摘要:")
    print("-" * 80)
    print(f"{'ID':<4} {'名称':<12} {'优先级':<10} {'运行时':<12} {'FPS':<6} {'延迟':<8} {'资源':<15} {'依赖':<10}")
    print("-" * 80)
    
    for task in tasks:
        # 获取资源类型
        resources = []
        for seg in task.segments:
            if seg.resource_type.value not in [r for r in resources]:
                resources.append(seg.resource_type.value)
        resource_str = "+".join(resources)
        
        # 获取依赖
        deps = ",".join(task.dependencies) if task.dependencies else "无"
        
        print(f"{task.task_id:<4} {task.name:<12} {task.priority.name:<10} "
              f"{task.runtime_type.value:<12} {task.fps_requirement:<6.0f} "
              f"{task.latency_requirement:<8.0f} {resource_str:<15} {deps:<10}")
    
    # 统计信息
    print("\n📈 统计信息:")
    total_tasks = len(tasks)
    npu_only = sum(1 for t in tasks if t.uses_npu and not t.uses_dsp)
    dsp_only = sum(1 for t in tasks if t.uses_dsp and not t.uses_npu)
    mixed = sum(1 for t in tasks if t.uses_npu and t.uses_dsp)
    
    print(f"  总任务数: {total_tasks}")
    print(f"  纯NPU任务: {npu_only}")
    print(f"  纯DSP任务: {dsp_only}")
    print(f"  混合任务: {mixed}")
    
    # 优先级分布
    priority_dist = {}
    for task in tasks:
        priority_dist[task.priority.name] = priority_dist.get(task.priority.name, 0) + 1
    
    print("\n  优先级分布:")
    for priority, count in priority_dist.items():
        print(f"    {priority}: {count}")


def test_bandwidth_impact():
    """测试带宽对任务执行时间的影响"""
    print("\n🔬 带宽影响分析:")
    
    tasks = create_real_tasks()
    test_bandwidths = [20, 40, 80, 120]
    
    # 选择几个代表性任务
    test_tasks = {
        "T1": "MOTR (混合)",
        "T2": "YOLO (大)",
        "T6": "ReID (高频)",
        "T7": "Pose2D (依赖)"
    }
    
    print("\n不同带宽下的执行时间 (ms):")
    print(f"{'任务':<15}", end="")
    for bw in test_bandwidths:
        print(f"{bw:>8}", end="")
    print("\n" + "-" * 50)
    
    for task_id, desc in test_tasks.items():
        task = next(t for t in tasks if t.task_id == task_id)
        print(f"{desc:<15}", end="")
        
        for bw in test_bandwidths:
            bandwidth_map = {ResourceType.NPU: bw, ResourceType.DSP: bw}
            duration = task.estimate_duration(bandwidth_map)
            print(f"{duration:>8.2f}", end="")
        print()


if __name__ == "__main__":
    print("真实任务定义测试")
    print("=" * 80)
    
    # 创建任务
    tasks = create_real_tasks()
    
    # 打印摘要
    print_task_summary(tasks)
    
    # 测试带宽影响
    test_bandwidth_impact()
    
    print("\n✅ 所有任务创建成功！")
