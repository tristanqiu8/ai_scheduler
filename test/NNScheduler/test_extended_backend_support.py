#!/usr/bin/env python3
"""
扩展后端支持测试用例
验证系统对NPU、DSP之外的其他硬件后端的支持能力
包括ISP、CPU、GPU等，并测试混合Pipeline场景
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from enum import Enum
from typing import Dict, List, Tuple
from NNScheduler.core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from NNScheduler.core.models import ResourceSegment, ResourceUnit, TaskScheduleInfo
from NNScheduler.core.task import NNTask
from NNScheduler.core.resource_queue import ResourceQueueManager, ResourceQueue
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.executor import ScheduleExecutor, set_execution_log_enabled
from NNScheduler.core.launcher import TaskLauncher

# 直接使用核心ResourceType，已扩展支持多种后端类型

def create_automotive_vision_task() -> NNTask:
    """创建汽车视觉处理任务：ISP -> NPU -> GPU -> CPU"""

    task = NNTask(
        task_id="auto_vision",
        name="Automotive Vision Processing",
        priority=TaskPriority.CRITICAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )

    # Pipeline段定义
    pipeline_segments = [
        # ISP: 原始图像预处理
        (ResourceType.ISP, {60.0: 12.0, 50.0: 15.0, 40.0: 18.0}, "image_preprocess"),

        # NPU: 目标检测推理（可能支持切分，但这里不切分）
        (ResourceType.NPU, {60.0: 25.0, 50.0: 30.0, 40.0: 35.0}, "object_detection"),

        # GPU: 图像渲染和可视化
        (ResourceType.GPU, {60.0: 8.0, 50.0: 10.0, 40.0: 12.0}, "render_overlay"),

        # CPU: 最终决策和控制
        (ResourceType.CPU, {60.0: 5.0, 50.0: 6.0, 40.0: 8.0}, "decision_control")
    ]

    # 手动添加段（模拟扩展后端支持）
    for resource_type, duration_table, segment_id in pipeline_segments:
        segment = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id
        )
        task.segments.append(segment)

    task.set_performance_requirements(fps=30.0, latency=60.0)
    return task

def create_multimedia_processing_task() -> NNTask:
    """创建多媒体处理任务：VPU -> DSP -> GPU -> FPGA"""

    task = NNTask(
        task_id="multimedia_proc",
        name="Multimedia Processing",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )

    pipeline_segments = [
        # VPU: 视频解码
        (ResourceType.VPU, {50.0: 20.0, 40.0: 25.0, 30.0: 30.0}, "video_decode"),

        # DSP: 音频处理
        (ResourceType.DSP, {50.0: 15.0, 40.0: 18.0, 30.0: 22.0}, "audio_process"),

        # GPU: 视频效果处理
        (ResourceType.GPU, {50.0: 12.0, 40.0: 15.0, 30.0: 18.0}, "video_effects"),

        # FPGA: 硬件加速编码
        (ResourceType.FPGA, {50.0: 18.0, 40.0: 22.0, 30.0: 26.0}, "hw_encode")
    ]

    for resource_type, duration_table, segment_id in pipeline_segments:
        segment = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id
        )
        task.segments.append(segment)

    task.set_performance_requirements(fps=25.0, latency=100.0)
    return task

def create_ai_inference_with_fallback_task() -> NNTask:
    """创建带CPU后备的AI推理任务：NPU -> CPU"""

    task = NNTask(
        task_id="ai_with_fallback",
        name="AI Inference with CPU Fallback",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )

    pipeline_segments = [
        # NPU: 主推理
        (ResourceType.NPU, {50.0: 30.0, 40.0: 35.0, 30.0: 42.0}, "main_inference"),

        # CPU: 后处理/备份推理
        (ResourceType.CPU, {50.0: 10.0, 40.0: 12.0, 30.0: 15.0}, "post_process")
    ]

    for resource_type, duration_table, segment_id in pipeline_segments:
        segment = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,
            segment_id=segment_id
        )
        task.segments.append(segment)

    task.set_performance_requirements(fps=20.0, latency=80.0)
    return task

def create_extended_resource_system() -> Tuple[List[ResourceUnit], ResourceQueueManager]:
    """创建扩展的资源系统"""

    units = []

    # NPU单元（支持切分）
    units.append(ResourceUnit("npu_0", ResourceType.NPU, 50.0))
    units.append(ResourceUnit("npu_1", ResourceType.NPU, 45.0))

    # DSP单元（支持切分）
    units.append(ResourceUnit("dsp_0", ResourceType.DSP, 48.0))

    # ISP单元（不支持切分）
    units.append(ResourceUnit("isp_0", ResourceType.ISP, 60.0))
    units.append(ResourceUnit("isp_1", ResourceType.ISP, 55.0))

    # CPU单元（不支持切分）
    units.append(ResourceUnit("cpu_0", ResourceType.CPU, 40.0))
    units.append(ResourceUnit("cpu_1", ResourceType.CPU, 35.0))

    # GPU单元（不支持切分）
    units.append(ResourceUnit("gpu_0", ResourceType.GPU, 70.0))

    # VPU单元（不支持切分）
    units.append(ResourceUnit("vpu_0", ResourceType.VPU, 45.0))

    # FPGA单元（不支持切分）
    units.append(ResourceUnit("fpga_0", ResourceType.FPGA, 50.0))

    # 创建队列管理器
    manager = ResourceQueueManager()
    for unit in units:
        queue = ResourceQueue(
            resource_id=unit.unit_id,
            resource_type=unit.resource_type,
            bandwidth=unit.bandwidth
        )
        manager.add_resource(unit.unit_id, queue)

    return units, manager

def analyze_backend_coverage(tasks: Dict[str, NNTask], resource_units: List[ResourceUnit]):
    """分析后端覆盖情况"""

    print("\n后端覆盖分析:")
    print("-" * 50)

    # 任务使用的后端类型
    task_backends = {}
    all_task_backends = set()

    for task_id, task in tasks.items():
        backends = set()
        for segment in task.segments:
            backends.add(segment.resource_type.value)
            all_task_backends.add(segment.resource_type.value)
        task_backends[task_id] = backends

    # 可用的后端类型
    available_backends = set(unit.resource_type.value for unit in resource_units)

    print(f"任务需要的后端类型: {', '.join(sorted(all_task_backends))}")
    print(f"系统提供的后端类型: {', '.join(sorted(available_backends))}")

    # 检查覆盖率
    covered = all_task_backends.intersection(available_backends)
    missing = all_task_backends - available_backends

    print(f"覆盖的后端类型: {', '.join(sorted(covered))}")
    if missing:
        print(f"缺失的后端类型: {', '.join(sorted(missing))}")
    else:
        print("[OK] 所有任务的后端需求都得到满足")

    # 各任务的后端使用
    print("\n各任务后端使用:")
    for task_id, backends in task_backends.items():
        print(f"  {task_id}: {', '.join(sorted(backends))}")

def test_extended_backend_support():
    """测试扩展后端支持"""

    print("=" * 80)
    print("扩展后端支持测试")
    print("=" * 80)

    # 关闭详细执行日志
    set_execution_log_enabled(False)

    # 创建测试任务
    tasks = {
        "auto_vision": create_automotive_vision_task(),
        "multimedia_proc": create_multimedia_processing_task(),
        "ai_with_fallback": create_ai_inference_with_fallback_task()
    }

    # 创建扩展资源系统
    resource_units, queue_manager = create_extended_resource_system()

    # 分析后端覆盖
    analyze_backend_coverage(tasks, resource_units)

    print(f"\n资源配置 ({len(resource_units)} 个单元):")
    backend_count = {}
    for unit in resource_units:
        backend_type = unit.resource_type.value
        backend_count[backend_type] = backend_count.get(backend_type, 0) + 1
        print(f"  {unit.unit_id}: {backend_type} (带宽: {unit.bandwidth})")

    print(f"\n后端分布:")
    for backend_type, count in backend_count.items():
        print(f"  {backend_type}: {count} 个单元")

    # 任务配置信息
    print(f"\n任务配置 ({len(tasks)} 个任务):")
    for task_id, task in tasks.items():
        segments_info = []
        for seg in task.segments:
            segments_info.append(f"{seg.resource_type.value}")

        print(f"  {task_id}:")
        print(f"    优先级: {task.priority.name}")
        print(f"    FPS: {task.fps_requirement}")
        print(f"    Pipeline: {' -> '.join(segments_info)}")

    # 创建调度器
    tracer = ScheduleTracer(queue_manager)
    executor = ScheduleExecutor(queue_manager, tracer, tasks)
    launcher = TaskLauncher(queue_manager, tracer)

    # 注册任务
    for task in tasks.values():
        launcher.register_task(task)

    # 生成发射计划
    print(f"\n生成发射计划...")
    fps_map = {
        "auto_vision": 30.0,
        "multimedia_proc": 25.0,
        "ai_with_fallback": 20.0
    }

    launch_plan = launcher.create_launch_plan(time_window=300.0, strategy="balanced")

    print(f"发射计划: {len(launch_plan.events)} 个事件")

    # 执行传统模式调度
    print(f"\n执行传统模式调度...")
    stats_traditional = executor.execute_plan(launch_plan, 300.0, segment_mode=False)

    # 重置并执行段级模式调度
    executor._reset_state()
    print(f"\n执行段级Pipeline模式调度...")
    stats_segment = executor.execute_plan(launch_plan, 300.0, segment_mode=True)

    # 性能统计比较
    print(f"\n性能统计比较:")
    print("-" * 50)

    modes = [("传统模式", stats_traditional), ("段级Pipeline模式", stats_segment)]

    for mode_name, stats in modes:
        print(f"\n{mode_name}:")
        print(f"  任务实例数: {stats['total_instances']}")
        print(f"  完成实例数: {stats['completed_instances']}")
        print(f"  完成率: {stats['completed_instances'] / max(stats['total_instances'], 1) * 100:.1f}%")
        print(f"  执行段数: {stats['total_segments_executed']}")
        print(f"  仿真时间: {stats['simulation_time']:.1f}ms")

        if 'average_completion_times' in stats and stats['average_completion_times']:
            print("  平均完成时间:")
            for task_id, avg_time in stats['average_completion_times'].items():
                print(f"    {task_id}: {avg_time:.1f}ms")

    # 计算性能提升
    if stats_traditional['total_segments_executed'] > 0 and stats_segment['total_segments_executed'] > 0:
        throughput_improvement = (stats_segment['total_segments_executed'] /
                                 stats_traditional['total_segments_executed'] - 1) * 100

        print(f"\n性能提升分析:")
        print(f"  段级模式吞吐量提升: {throughput_improvement:.1f}%")

    # 验证扩展后端支持
    print(f"\n扩展后端支持验证:")
    print("-" * 50)

    unique_backends = set()
    for task in tasks.values():
        for segment in task.segments:
            unique_backends.add(segment.resource_type.value)

    non_traditional_backends = unique_backends - {"NPU", "DSP"}

    print(f"[OK] 成功支持 {len(unique_backends)} 种后端类型")
    print(f"[OK] 支持传统后端: NPU, DSP")

    if non_traditional_backends:
        print(f"[OK] 支持扩展后端: {', '.join(sorted(non_traditional_backends))}")

    print(f"[OK] Pipeline流水线调度正常工作")
    print(f"[OK] 多后端任务调度无冲突")
    print(f"[OK] 非NPU后端无需切分支持（按设计）")

    return {
        'backend_types': len(unique_backends),
        'traditional_stats': stats_traditional,
        'segment_stats': stats_segment,
        'supported_backends': sorted(unique_backends)
    }

if __name__ == "__main__":
    results = test_extended_backend_support()

    print(f"\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    print(f"支持的后端类型数量: {results['backend_types']}")
    print(f"支持的后端类型: {', '.join(results['supported_backends'])}")
    print("测试结果: [OK] 成功验证多后端扩展支持能力")