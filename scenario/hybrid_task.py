#!/usr/bin/env python3
"""
真实任务定义 - 使用精简后的接口（无 start_time）
"""

from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.task import NNTask, create_npu_task, create_dsp_task, create_mixed_task


def create_real_tasks():
    """创建测试任务集"""
    
    tasks = []
    
    print("\n📋 创建测试任务:")
    
    fps_table = {"Parsing": 60,
                 "ReID": 100,
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
                 "Skywater_Big": 30
                 }
    
    # 任务1: 3A Parsing
    task1 = NNTask(
        "T1", "Parsing",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    # 添加NPU主段
    task1.add_segment(ResourceType.NPU, {20: 1.63, 40: 1.16, 80: 0.93, 120: 0.90}, "main")
    # 添加DSP后处理段
    task1.add_segment(ResourceType.DSP, {20: 0.48, 40: 0.46, 80: 0.46, 120: 0.45}, "postprocess")
    task1.set_performance_requirements(fps=fps_table[task1.name], latency=1000.0/fps_table[task1.name])
    tasks.append(task1)
    print("  ✓ T1 Parsing: 3A中频NPU+DSP任务")
    
    # 任务2: 重识别（高频任务）
    task2 = create_npu_task(
        "T2", "ReID",
        {20: 1.06, 40: 0.72, 80: 0.59, 120: 0.631},
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task2.set_performance_requirements(fps=fps_table[task2.name], latency=1000.0/fps_table[task2.name])
    tasks.append(task2)
    print("  ✓ T2 ReID: 高频NPU任务")
    
    # 任务3: MOTR - 多目标跟踪（关键任务）
    task3 = create_mixed_task(
        "T3", "MOTR",
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
    task3.set_performance_requirements(fps=fps_table[task3.name], latency=1000.0/fps_table[task3.name])
    tasks.append(task3)
    print("  ✓ T3 MOTR: 9段混合任务 (4 DSP + 5 NPU)")
    
    # 任务4: motr post处理 - qim
    task4 = create_mixed_task(  
        "T4", "qim",
        segments=[
            (ResourceType.NPU, {10: 1.339, 20: 0.758, 40: 0.474, 80: 0.32, 120: 0.292}, "npu_sub"),
            (ResourceType.DSP, {10: 1.238, 20: 1.122, 40: 1.04, 80: 1, 120: 1.014}, "dsp_sub"),
        ],
        priority=TaskPriority.LOW,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task4.set_performance_requirements(fps=fps_table[task4.name], latency=1000.0/fps_table[task4.name])
    task4.add_dependency("T3")  # 依赖MOTR
    tasks.append(task4)
    print("  ✓ T4 qim: DSP+NPU混合任务 (依赖T3)")
    
    # 任务5: 2D姿态估计
    task5 = create_npu_task(
        "T5", "pose2d",
        {20: 4.324, 40: 3.096, 80: 2.28, 120: 2.04},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task5.set_performance_requirements(fps=fps_table[task5.name], latency=1000.0/fps_table[task5.name])
    task5.add_dependency("T3")  # 依赖MOTR的检测结果
    tasks.append(task5)
    print("  ✓ T5 pose2d: NPU任务 (依赖T3)")
    
    # 任务6: 模板匹配
    task6 = create_npu_task(
        "T6", "tk_template",
        {20: 0.48, 40: 0.33, 80: 0.27, 120: 0.25},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task6.set_performance_requirements(fps=fps_table[task6.name], latency=1000.0/fps_table[task6.name])
    tasks.append(task6)
    print("  ✓ T6 tk_temp: 纯NPU任务")
    
    # 任务7: 搜索任务
    task7 = create_npu_task(
        "T7", "tk_search",
        {20: 1.16, 40: 0.72, 80: 0.54, 120: 0.50},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task7.set_performance_requirements(fps=fps_table[task7.name], latency=1000.0/fps_table[task7.name])
    tasks.append(task7)
    print("  ✓ T7 tk_search: 纯NPU任务")
    
    # 任务8：灰度Mask
    task8 = create_npu_task(
        "T8", "GrayMask",
        {20: 2.42, 40: 2.00, 80: 1.82, 120: 1.80},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
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
    # 添加NPU主段
    task9.add_segment(ResourceType.NPU, {20: 20.28, 40: 12.31, 120: 7.50}, "main")
    # 添加DSP后处理段
    # task9.add_segment(ResourceType.DSP, {40: 3.423}, "postprocess")
    
    # 为主段添加切分点
    task9.add_cut_points_to_segment("main", [
        ("op6", {20: 4.699, 40: 2.737, 120: 1.482}, 0.0),   # 20%处
        ("op13", {20: 9.398, 40: 5.474, 120: 2.964}, 0.0),  # 40%处
        ("op14", {20: 14.096, 40: 8.210, 120: 4.447}, 0.0), # 60%处
        ("op19", {20: 18.795, 40: 10.947, 120: 5.929}, 0.0) # 80%处
    ])
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
    task10.add_segment(ResourceType.NPU, {20: 5.02, 40: 3.16, 120: 2.03}, "main")
    # task10.add_segment(ResourceType.DSP, {40: 1.957}, "postprocess")
    
    # 添加切分点
    task10.add_cut_points_to_segment("main", [
        ("op5", {20: 1.138, 40: 0.691, 120: 0.418}, 0.0),   # 20%处
        ("op15", {20: 2.276, 40: 1.382, 120: 0.835}, 0.0),  # 40%处
        ("op19", {20: 4.551, 40: 2.763, 120: 1.670}, 0.0)   # 80%处
    ])
    task10.set_performance_requirements(fps=fps_table[task10.name], latency=1000.0/fps_table[task10.name])
    tasks.append(task10)
    print("  ✓ T10 YoloV8nSmall: 可分段NPU任务")
    
    # 任务11: Stereo4x - 双目深度（关键任务）
    task11 = create_mixed_task(
        "T11", "Stereo4x",
        segments=[
            (ResourceType.NPU, {20: 4.347, 40: 2.730, 80: 2.002, 120: 1.867}, "npu_s0"), #scale
            (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s0"), #guess
            (ResourceType.NPU, {20: 2.900, 40: 2.016, 80: 1.642, 120: 1.608}, "npu_s1"), #scale
            (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s1"), #guess
            (ResourceType.DSP, {20: 1.16, 40: 0.655, 80: 0.441, 120: 0.404}, "dsp_s2"), #guess
            (ResourceType.NPU, {20: 1.456, 40: 1.046, 80: 0.791, 120: 0.832}, "npu_s2"), #scale
            (ResourceType.NPU, {20: 1.456, 40: 1.115, 80: 0.932, 120: 0.924}, "npu_s3"), #scale
            (ResourceType.NPU, {20: 8.780, 40: 6.761, 80: 5.712, 120: 5.699}, "npu_s4"), #scale
        ],
        priority=TaskPriority.CRITICAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
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
    task12.add_segment(ResourceType.NPU, {20: 2.31, 40: 1.49, 80: 1.14, 120: 1.02}, "main")
    task12.add_segment(ResourceType.DSP, {20: 1.23, 40: 0.71, 80: 0.45, 120: 0.41}, "postprocess")
    
    # 添加切分点
    task12.add_cut_points_to_segment("main", [
        ("op4", {20: 0.924, 40: 0.596, 80: 0.456, 120: 0.408}, 0.0),   # 40%处
        ("op14", {20: 1.201, 40: 0.775, 80: 0.593, 120: 0.530}, 0.0),  # 50%处
    ])
    task12.set_performance_requirements(fps=fps_table[task12.name], latency=70.0)
    tasks.append(task12)
    print("  ✓ T12 Skywater: 可分段NPU+DSP任务")
    
    # 任务8：灰度Mask
    task13 = create_npu_task(
        "T13", "PeakDetector",
        {20: 1.51, 40: 0.97, 80: 0.70, 120: 0.62},
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    task13.set_performance_requirements(fps=fps_table[task13.name], latency=1000.0/fps_table[task13.name])
    tasks.append(task13)
    print("  ✓ T13 PeakDetector: 纯NPU任务")
    
    # 任务14: Skywater 大模型
    task14 = NNTask(
        "T14", "Skywater_Big",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.FORCED_SEGMENTATION
    )
    task14.add_segment(ResourceType.NPU, {20: 4.19, 40: 2.49, 80: 1.70, 120: 1.67}, "main")
    task14.add_segment(ResourceType.DSP, {20: 1.52, 40: 0.90, 80: 0.58, 120: 0.58}, "postprocess")
    
    # 添加切分点
    task14.add_cut_points_to_segment("main", [
        ("op4", {20: 1.676, 40: 0.996, 80: 0.680, 120: 0.668}, 0.0),   # 40%处
        ("op14", {20: 2.179, 40: 1.295, 80: 0.884, 120: 0.868}, 0.0),  # 50%处
    ])
    task14.set_performance_requirements(fps=fps_table[task12.name], latency=1000.0/fps_table[task12.name])
    tasks.append(task14)
    print("  ✓ T14 Skywater Mono: 可分段NPU+DSP任务")
    
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
