#!/usr/bin/env python3
"""
真实任务定义 - 使用精简后的接口（无 start_time）
"""

from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.task import NNTask, create_npu_task, create_dsp_task, create_mixed_task


def create_real_tasks():
    """创建测试任务集"""
    
    tasks = []
    
    print("\n[INFO] 创建测试任务:")
    
    fps_table = {"T1": 25,
                 "T2": 10,
                 "T3": 10, 
                 "T4": 5,
                 "T5": 25,
                 "T6": 60,
                 "T7": 25,
                 "T8": 25,
                 "T9": 25
                 }
    # fps_table = {"T1": 34,
    #              "T2": 14,
    #              "T3": 14, 
    #              "T4": 7,
    #              "T5": 34,
    #              "T6": 100,
    #              "T7": 34,
    #              "T8": 34,
    #              "T9": 34
    #              }
    
    # 任务1: MOTR - 多目标跟踪（关键任务）
    task1 = create_mixed_task(
        "T1", "MOTR",
        segments=[
            (ResourceType.DSP, {20: 0.316, 40: 0.305, 120: 0.368}, "dsp_s0"),
            (ResourceType.NPU, {20: 0.430, 40: 0.303, 120: 0.326}, "npu_s1"),
            (ResourceType.NPU, {20: 12.868, 40: 7.506, 120: 4.312}, "npu_s2"),
            (ResourceType.DSP, {20: 1.734, 40: 1.226, 120: 0.994}, "dsp_s1"),
            (ResourceType.NPU, {20: 0.997, 40: 0.374, 120: 0.211}, "npu_s3"),
            (ResourceType.DSP, {20: 1.734, 40: 1.201, 120: 0.943}, "dsp_s2"),
            (ResourceType.NPU, {20: 0.602, 40: 0.373, 120: 0.209}, "npu_s4"),
            (ResourceType.DSP, {20: 1.690, 40: 1.208, 120: 0.975}, "dsp_s3"),
            (ResourceType.NPU, {20: 0.596, 40: 0.223, 120: 0.134}, "npu_s4"),
        ],
        priority=TaskPriority.CRITICAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task1.set_performance_requirements(fps=fps_table['T1'], latency=1000.0/fps_table['T1'])
    tasks.append(task1)
    print("  [OK] T1 MOTR: 9段混合任务 (4 DSP + 5 NPU)")
    
    # 任务2: YOLOv8n 大模型
    task2 = NNTask(
        "T2", "YoloV8nBig",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    # 添加NPU主段
    task2.add_segment(ResourceType.NPU, {20: 23.494, 40: 13.684, 120: 7.411}, "main")
    # 添加DSP后处理段
    task2.add_segment(ResourceType.DSP, {40: 3.423}, "postprocess")
    
    # 为主段添加切分点
    task2.add_cut_points_to_segment("main", [
        ("op6", {20: 4.699, 40: 2.737, 120: 1.482}, 0.0),   # 20%处
        ("op13", {20: 9.398, 40: 5.474, 120: 2.964}, 0.0),  # 40%处
        ("op14", {20: 14.096, 40: 8.210, 120: 4.447}, 0.0), # 60%处
        ("op19", {20: 18.795, 40: 10.947, 120: 5.929}, 0.0) # 80%处
    ])
    task2.set_performance_requirements(fps=fps_table['T2'], latency=1000.0/fps_table['T2'])
    tasks.append(task2)
    print("  [OK] T2 YoloV8nBig: 可分段NPU+DSP任务")
    
    # 任务3: YOLOv8n 小模型
    task3 = NNTask(
        "T3", "YoloV8nSmall",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task3.add_segment(ResourceType.NPU, {20: 5.689, 40: 3.454, 120: 2.088}, "main")
    task3.add_segment(ResourceType.DSP, {40: 1.957}, "postprocess")
    
    # 添加切分点
    task3.add_cut_points_to_segment("main", [
        ("op5", {20: 1.138, 40: 0.691, 120: 0.418}, 0.0),   # 20%处
        ("op15", {20: 2.276, 40: 1.382, 120: 0.835}, 0.0),  # 40%处
        ("op19", {20: 4.551, 40: 2.763, 120: 1.670}, 0.0)   # 80%处
    ])
    task3.set_performance_requirements(fps=fps_table['T3'], latency=1000.0/fps_table['T3'])
    tasks.append(task3)
    print("  [OK] T3 YoloV8nSmall: 可分段NPU+DSP任务")
    
    # 任务4: 模板匹配
    task4 = create_npu_task(
        "T4", "tk_temp",
        {20: 0.465, 40: 0.364, 120: 0.296},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task4.set_performance_requirements(fps=fps_table['T4'], latency=1000.0/fps_table['T4'])
    tasks.append(task4)
    print("  [OK] T4 tk_temp: 纯NPU任务")
    
    # 任务5: 搜索任务
    task5 = create_npu_task(
        "T5", "tk_search",
        {20: 0.960, 40: 0.755, 120: 0.558},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task5.set_performance_requirements(fps=fps_table['T5'], latency=1000.0/fps_table['T5'])
    tasks.append(task5)
    print("  [OK] T5 tk_search: 纯NPU任务")
    
    # 任务6: 重识别（高频任务）
    task6 = create_npu_task(
        "T6", "reid",
        {20: 0.891, 40: 0.778, 120: 0.631},
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task6.set_performance_requirements(fps=fps_table['T6'], latency=1000.0/fps_table['T6'])
    tasks.append(task6)
    print("  [OK] T6 reid: 高频NPU任务")
    
    # 任务7: 2D姿态估计
    task7 = create_npu_task(
        "T7", "pose2d",
        {20: 4.324, 40: 3.096, 120: 2.232},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task7.set_performance_requirements(fps=fps_table['T7'], latency=1000.0/fps_table['T7'])
    task7.add_dependency("T1")  # 依赖MOTR的检测结果
    tasks.append(task7)
    print("  [OK] T7 pose2d: NPU任务 (依赖T1)")
    
    # 任务8: motr post处理 - qim
    task8 = create_mixed_task(  
        "T8", "qim",
        segments=[
            (ResourceType.NPU, {10: 1.339, 20: 0.758, 40: 0.474, 80: 0.32, 120: 0.292}, "npu_sub"),
            (ResourceType.DSP, {10: 1.238, 20: 1.122, 40: 1.04, 80: 1, 120: 1.014}, "dsp_sub"),
        ],
        priority=TaskPriority.LOW,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task8.set_performance_requirements(fps=fps_table['T8'], latency=1000.0/fps_table['T8'])
    task8.add_dependency("T1")  # 依赖MOTR
    tasks.append(task8)
    print("  [OK] T8 qim: DSP+NPU混合任务 (依赖T1)")
    
    # 任务9： pose2d to 3d
    # task9 = create_dsp_task(
    #     "T9", "pose2d_to_3d",
    #     {40: 9.382, 120: 9.337},
    #     priority=TaskPriority.NORMAL,
    #     runtime_type=RuntimeType.ACPU_RUNTIME,
    #     segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    # )
    task9 = create_npu_task(
        "T9", "pose2d_to_3d",
        {20: 0.16, 40: 0.15, 80: 0.13, 120: 0.13},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task9.set_performance_requirements(fps=fps_table['T9'], latency=1000.0/fps_table['T9'])
    task9.add_dependency("T7")  # 依赖pose2d任务
    tasks.append(task9)
    print("  [OK] T9 pose2d-to-3d: Pure DSP task (依赖T7)")
    
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
