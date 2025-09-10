#!/usr/bin/env python3
"""
真实任务定义 - 2NPU+1DSP硬件配置专用版本
基于原版camera_task.py，但优化为更高性能要求以充分利用2NPU+1DSP硬件
"""

from NNScheduler.core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from NNScheduler.core.task import NNTask, create_npu_task, create_dsp_task, create_mixed_task
from NNScheduler.scenario.model_repo import get_model


def create_real_tasks_2npu1dsp():
    """创建适合2NPU+1DSP硬件配置的测试任务集"""
    
    tasks = []
    
    print("\n[INFO] 创建2NPU+1DSP测试任务:")
    # 针对2NPU+1DSP硬件配置，设置更高的主频率以充分利用硬件
    Main_Freq = 30  # 相比原版15提升一倍，充分利用双NPU优势
    fps_table = {
        "ML10T_bigmid": Main_Freq,
        "ML10T_midsmall": Main_Freq,
        "AimetlitePlus": Main_Freq * 2,  # 60 FPS - 充分利用双NPU
        "FaceEhnsLite": Main_Freq * 2,   # 60 FPS - 高性能面部识别
        "Vmask": Main_Freq * 2,          # 60 FPS - 高频视频掩码
        "Cam_Parsing": Main_Freq,        # 30 FPS - 摄像头解析
        "FaceDet": Main_Freq,            # 30 FPS - 面部检测
        "PD_Depth": Main_Freq,           # 30 FPS - 深度估计
        "AF_TK": Main_Freq,              # 30 FPS - 自动对焦跟踪
        "NNTone": Main_Freq * 2,         # 60 FPS - 神经网络色调
        "PD_DNS": Main_Freq * 2,         # 60 FPS - 降噪处理
    }
    
    # 任务1: 光流AimetlitePlus - 高性能光流计算
    task1 = NNTask(
        "T1", "AimetlitePlus",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task1.apply_model(get_model(task1.name))
    task1.set_performance_requirements(fps=fps_table[task1.name], latency=1000.0/fps_table[task1.name])
    tasks.append(task1)
    print("  [OK] T1 AimetlitePlus: 高性能光流计算 (60FPS)")
    
    # 任务2: FaceEhnsLite - 高频面部增强
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
    print("  [OK] T2 FaceEhnsLite: 高频面部增强 (60FPS)")
    
    # 任务3: Vmask - 高性能视频掩码
    task3 = NNTask(
        "T3", "Vmask",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task3.apply_model(get_model(task3.name))
    task3.set_performance_requirements(fps=fps_table[task3.name], latency=1000.0/fps_table[task3.name])
    tasks.append(task3)
    print("  [OK] T3 Vmask: 高性能视频掩码 (60FPS)")
    
    # 任务4: Cam_Parsing - 摄像头解析
    task4 = NNTask(
        "T4", "Cam_Parsing",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task4.apply_model(get_model(task4.name))
    task4.set_performance_requirements(fps=fps_table[task4.name], latency=1000.0/fps_table[task4.name])
    tasks.append(task4)
    print("  [OK] T4 Cam Parsing: 摄像头场景解析 (30FPS)")
    
    # 任务5: FaceDet - 面部检测
    task5 = NNTask(
        "T5", "FaceDet",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task5.apply_model(get_model(task5.name))
    task5.set_performance_requirements(fps=fps_table[task5.name], latency=1000.0/fps_table[task5.name])
    tasks.append(task5)
    print("  [OK] T5 FaceDet: 面部检测 (30FPS)")
    
    # 任务6: AF_TK - 自动对焦跟踪
    task6 = NNTask(
        "T6", "AF_TK",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task6.apply_model(get_model(task6.name))
    task6.set_performance_requirements(fps=fps_table[task6.name], latency=1000.0/fps_table[task6.name])
    tasks.append(task6)
    print("  [OK] T6 AF_TK: 自动对焦跟踪 (30FPS)")
    
    # 任务7: PD_DNS - 高频降噪处理
    task7 = NNTask(
        "T7", "PD_DNS",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task7.apply_model(get_model(task7.name))
    task7.set_performance_requirements(fps=fps_table[task7.name], latency=1000.0/fps_table[task7.name])
    tasks.append(task7)
    print("  [OK] T7 PD_DNS: 高频降噪处理 (60FPS)")
    
    # 任务8: PD_Depth - 深度估计
    task8 = NNTask(
        "T8", "PD_Depth",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task8.apply_model(get_model(task8.name))
    task8.set_performance_requirements(fps=fps_table[task8.name], latency=1000.0/fps_table[task8.name])
    tasks.append(task8)
    print("  [OK] T8 PD_Depth: 深度估计 (30FPS)")
        
    # 任务9: NNTone - 高频色调处理
    task9 = NNTask(
        "T9", "NNTone",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task9.apply_model(get_model(task9.name))
    task9.set_performance_requirements(fps=fps_table[task9.name], latency=1000.0/fps_table[task9.name])
    tasks.append(task9)
    print("  [OK] T9 NNTone: 高频神经网络色调处理 (60FPS)")

    return tasks


def print_task_summary_2npu1dsp(tasks):
    """打印2NPU+1DSP任务摘要"""
    print("\n[ANALYSIS] 2NPU+1DSP任务摘要:")
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
    print("\n📈 2NPU+1DSP硬件配置统计信息:")
    total_tasks = len(tasks)
    npu_only = sum(1 for t in tasks if t.uses_npu and not t.uses_dsp)
    dsp_only = sum(1 for t in tasks if t.uses_dsp and not t.uses_npu)
    mixed = sum(1 for t in tasks if t.uses_npu and t.uses_dsp)
    
    print(f"  总任务数: {total_tasks}")
    print(f"  纯NPU任务: {npu_only}")
    print(f"  纯DSP任务: {dsp_only}")
    print(f"  混合任务: {mixed}")
    
    # 性能统计
    total_fps = sum(task.fps_requirement for task in tasks)
    high_fps_tasks = sum(1 for t in tasks if t.fps_requirement >= 60)
    medium_fps_tasks = sum(1 for t in tasks if 30 <= t.fps_requirement < 60)
    low_fps_tasks = sum(1 for t in tasks if t.fps_requirement < 30)
    
    print(f"  总FPS需求: {total_fps}")
    print(f"  高频任务(≥60FPS): {high_fps_tasks}")
    print(f"  中频任务(30-59FPS): {medium_fps_tasks}")
    print(f"  低频任务(<30FPS): {low_fps_tasks}")
    
    # 优先级分布
    priority_dist = {}
    for task in tasks:
        priority_dist[task.priority.name] = priority_dist.get(task.priority.name, 0) + 1
    
    print("\n  优先级分布:")
    for priority, count in priority_dist.items():
        print(f"    {priority}: {count}")


if __name__ == "__main__":
    print("2NPU+1DSP硬件配置任务定义测试")
    print("=" * 80)
    
    # 创建任务
    tasks = create_real_tasks_2npu1dsp()
    
    # 打印摘要
    print_task_summary_2npu1dsp(tasks)
    
    print("\n✅ 2NPU+1DSP任务创建成功！")