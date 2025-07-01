from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.task import NNTask

def create_real_tasks():
    """创建测试任务"""
    
    tasks = []
    
    print("\n📋 创建测试任务:")
    seg_overhead = 0.15  # 分段开销比例
    # 任务1: cnntk_template
    task1 = NNTask("T1", "MOTR",
                   priority=TaskPriority.CRITICAL,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task1.set_dsp_npu_sequence([
        (ResourceType.NPU, {20: 0.652, 40: 0.410, 120: 0.249}, 0, "npu_s0"),
        (ResourceType.DSP, {40: 1.2}, 0.410, "dsp_s0"),
        (ResourceType.NPU, {20: 0.998, 40: 0.626, 120: 0.379}, 1.61, "npu_s1"),
        (ResourceType.NPU, {20: 16.643, 40: 9.333, 120: 5.147}, 2.236, "npu_s2"),
        (ResourceType.DSP, {40: 2.2}, 11.569, "dsp_s1"),
        (ResourceType.NPU, {20: 0.997, 40: 0.626, 120: 0.379}, 13.769, "npu_s3"),
        (ResourceType.DSP, {40: 1.5}, 15.269, "dsp_s2"),
        (ResourceType.NPU, {20: 0.484, 40: 0.285, 120: 0.153}, 16.769, "npu_s4"),
        (ResourceType.DSP, {40: 2}, 17.054, "dsp_s3"),  
        # (ResourceType.NPU, {40: 3.54}, 19.054, "npu_s5"), # fake one to match with linyu's data
    ])
    task1.set_performance_requirements(fps=25, latency=40)
    tasks.append(task1)
    print("  ✓ T1 MOTR: NOSEG")
    
    #任务2： yolov8n_big
    task2 = NNTask("T2", "YoloV8nBig",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task2.set_dsp_npu_sequence([
        # (ResourceType.NPU, {20: 23.494, 40: 13.684, 120: 7.411}, 0, "main"),
        (ResourceType.NPU, {40: 12.71}, 0, "main"),
        (ResourceType.DSP, {40: 3.423}, 12.71, "postprocess"),
    ])
    task2.add_cut_points_to_segment("main", [
        ("op6", 0.2, seg_overhead),
        ("op13", 0.4, seg_overhead),
        ("op14", 0.6, seg_overhead),
        ("op19", 0.8, seg_overhead)
    ])
    task2.set_performance_requirements(fps=10, latency=100)
    tasks.append(task2)
    print("  ✓ T2 yolov8 big: SEG")
    
    #任务3： yolov8_small
    task3 = NNTask("T3", "YoloV8nSmall",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION)
    task3.set_dsp_npu_sequence([
        # (ResourceType.NPU, {20: 5.689, 40: 3.454, 120: 2.088}, 0, "main"),
        (ResourceType.NPU, {40: 3.237}, 0, "main"),
        (ResourceType.DSP, {40: 1.957}, 3.237, "postprocess"),
    ])
    task3.add_cut_points_to_segment("main", [
        ("op5", 0.2, seg_overhead),
        ("op15", 0.4, seg_overhead),
        ("op19", 0.8, seg_overhead)
    ])
    task3.set_performance_requirements(fps=10, latency=100)
    tasks.append(task3)
    print("  ✓ T3 yolov8 small: SEG")
    
    #任务4： tk_template
    task4 = NNTask("T4", "tk_temp",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task4.set_npu_only({40: 0.364, 120: 0.296}, "main")
    task4.set_performance_requirements(fps=5, latency=200)
    tasks.append(task4)
    print("  ✓ T4 tk template: NO SEG")
    
    #任务5： tk_search
    task5 = NNTask("T5", "tk_search",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task5.set_npu_only({40: 0.755, 120: 0.558}, "main")
    task5.set_performance_requirements(fps=25, latency=40)
    tasks.append(task5)
    print("  ✓ T5 tk search: NO SEG")
    
    #任务6： re_id
    task6 = NNTask("T6", "reid",
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task6.set_npu_only({40: 0.778, 120: 0.631}, "main")
    task6.set_performance_requirements(fps=100, latency=10)
    task6.add_dependency("T1")  # re_id depends on MOTR
    tasks.append(task6)
    print("  ✓ T6 re id: NO SEG")
    
    #任务7： pose2d
    task7 = NNTask("T7", "pose2d",
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task7.set_npu_only({40: 3.096, 120: 2.232}, "main")
    task7.set_performance_requirements(fps=25, latency=40)
    task7.add_dependency("T1")  # pose2d depends on MOTR
    tasks.append(task7)
    print("  ✓ T7 pose2d: NO SEG")
    
    #任务8： qim
    task8 = NNTask("T8", "qim",
                   priority=TaskPriority.LOW,
                   runtime_type=RuntimeType.DSP_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    task8.set_dsp_npu_sequence([
        (ResourceType.DSP, {40: 0.995, 120: 4.968}, 0, "dsp_sub"),
        (ResourceType.NPU, {40: 0.656, 120: 0.89}, 0.995, "npu_sub"),
    ])
    task8.set_performance_requirements(fps=25, latency=40)
    task8.add_dependency("T1")  # qim depends on MOTR
    tasks.append(task8)
    print("  ✓ T8 qim: NO SEG")
    
    return tasks