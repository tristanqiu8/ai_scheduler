#!/usr/bin/env python3
"""
真实任务定义 - 使用精简后的接口
"""

from NNScheduler.core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from NNScheduler.core.task import NNTask, create_npu_task, create_dsp_task, create_mixed_task
from NNScheduler.scenario.model_repo import get_model


def create_real_tasks():
    """创建测试任务集"""
    
    tasks = []
    
    print("\n[INFO] 创建测试任务:")
    Main_Freq = 15
    fps_table = {
        "ML10T_bigmid": Main_Freq,
        "ML10T_midsmall": Main_Freq,
        "AimetlitePlus": Main_Freq * 2,
        "FaceEhnsLite": Main_Freq * 2,
        "Vmask": Main_Freq * 2,
        "Cam_Parsing": Main_Freq,
        "FaceDet": Main_Freq,
        "PD_Depth": Main_Freq,
        "AF_TK": Main_Freq,
        "NNTone": Main_Freq * 2,
        "PD_DNS": Main_Freq * 2,
    }
    
    # 任务1: 光流AimetlitePlus
    task1 = NNTask(
        "T1", "AimetlitePlus",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    # 从model_lib获取模型定义并应用
    task1.apply_model(get_model(task1.name))
    task1.set_performance_requirements(fps=fps_table[task1.name], latency=1000.0/fps_table[task1.name])
    tasks.append(task1)
    print("  [OK] T1 AimetlitePlus: 重型NPU可分段任务")
    
    # 任务2: FaceEhnsLite
    task2 = NNTask(
        "T2", "FaceEhnsLite",
        priority=TaskPriority.LOW,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task2.apply_model(get_model(task2.name))
    task2.set_performance_requirements(fps=fps_table[task2.name], latency=1000.0/fps_table[task2.name])
    task2.add_dependencies(["T1", "T3", "T5"])  # 依赖光流和FD
    tasks.append(task2)
    print("  [OK] T2 FaceEhnsLite: 中型NPU可分段任务")
    
    # 任务3: Vmask
    task3 = NNTask(
        "T3", "Vmask",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task3.apply_model(get_model(task3.name))
    task3.set_performance_requirements(fps=fps_table[task3.name], latency=1000.0/fps_table[task3.name])
    tasks.append(task3)
    print("  [OK] T3 Vmask: 中型NPU可分段任务")
    
    # 任务4: Cam_Parsing
    task4 = NNTask(
        "T4", "Cam_Parsing",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task4.apply_model(get_model(task4.name))
    task4.set_performance_requirements(fps=fps_table[task4.name], latency=1000.0/fps_table[task4.name])
    # task4.add_dependency("T3")  # 依赖MOTR
    tasks.append(task4)
    print("  [OK] T4 Cam Parsing: 轻型NPU任务")
    
    # 任务5: FaceDet
    task5 = NNTask(
        "T5", "FaceDet",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task5.apply_model(get_model(task5.name))
    task5.set_performance_requirements(fps=fps_table[task5.name], latency=1000.0/fps_table[task5.name])
    # task5.add_dependency("T3")  # 依赖MOTR的检测结果
    tasks.append(task5)
    print("  [OK] T5 FaceDet: 小型NPU任务")
    
    # 任务6: AF_TK
    task6 = NNTask(
        "T6", "AF_TK",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task6.apply_model(get_model(task6.name))
    task6.set_performance_requirements(fps=fps_table[task6.name], latency=1000.0/fps_table[task6.name])
    tasks.append(task6)
    print("  [OK] T6 AF_TK/MOTR: 复杂NPU+DSP混合任务")
    
    # 任务7: 搜索任务
    task7 = NNTask(
        "T7", "PD_DNS",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task7.apply_model(get_model(task7.name))
    task7.set_performance_requirements(fps=fps_table[task7.name], latency=1000.0/fps_table[task7.name])
    tasks.append(task7)
    print("  [OK] T7 PD_DNS: 微型纯NPU任务")
    
    # 任务8：NNTone
    task8 = NNTask(
        "T8", "PD_Depth",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task8.apply_model(get_model(task8.name))
    task8.set_performance_requirements(fps=fps_table[task8.name], latency=1000.0/fps_table[task8.name])
    tasks.append(task8)
    print("  [OK] T8 PD Depth: 复杂DSP+NPU混合任务")
        
    # 任务9: NNTone
    task9 = NNTask(
        "T9", "NNTone",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task9.apply_model(get_model(task9.name))
    task9.set_performance_requirements(fps=fps_table[task9.name], latency=1000.0/fps_table[task9.name])
    tasks.append(task9)
    print("  [OK] T9 NNTone: 小型纯NPU任务")

    return tasks


def print_task_summary(tasks):
    """打印任务摘要"""
    print("\n[ANALYSIS] 任务摘要:")
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
        
        print(f"{task.task_id:<4} {task.name:<14} {task.priority.name:<10} "
              f"{task.runtime_type.value:<18} {task.fps_requirement:<6.0f} "
              f"{task.latency_requirement:<10.1f} {resource_str:<16} {deps:<10}")
    
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
    # test_bandwidth_impact()
    
    print("\n✅ 所有任务创建成功！")
