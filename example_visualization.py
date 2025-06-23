#!/usr/bin/env python3
"""
Example usage of enhanced visualization for scheduler results
"""

import os
from datetime import datetime

from enums import ResourceType
from models import ResourceUnit
from task import TaskFactory, TaskSet, NNTask
from scheduler_base import SimpleScheduler, PriorityQueueScheduler
from scheduler_advanced import GeneticScheduler
from visualization_enhanced import EnhancedSchedulerVisualizer


def create_complex_scenario():
    """Create a complex scheduling scenario with multiple resource types"""
    # Create diverse resources
    resources = {
        # DSP resources
        "DSP_0": ResourceUnit("DSP_0", "High-Speed DSP", ResourceType.DSP, 8.0),
        "DSP_1": ResourceUnit("DSP_1", "Standard DSP", ResourceType.DSP, 4.0),
        "DSP_2": ResourceUnit("DSP_2", "Low-Power DSP", ResourceType.DSP, 2.0),
        
        # NPU resources  
        "NPU_0": ResourceUnit("NPU_0", "High-Perf NPU", ResourceType.NPU, 16.0),
        "NPU_1": ResourceUnit("NPU_1", "Mid-Perf NPU", ResourceType.NPU, 8.0),
        "NPU_2": ResourceUnit("NPU_2", "Standard NPU", ResourceType.NPU, 4.0),
        "NPU_3": ResourceUnit("NPU_3", "Low-Power NPU", ResourceType.NPU, 2.0),
    }
    
    # Create diverse task set
    tasks = TaskSet()
    
    # Critical safety tasks
    safety1 = TaskFactory.create_safety_monitor(fps=30)
    safety1.name = "SafetyCore"
    tasks.add_task(safety1)
    
    safety2 = TaskFactory.create_safety_monitor(fps=15)
    safety2.name = "SafetyBackup"
    tasks.add_task(safety2)
    
    # Object detection with DSP
    detection1 = TaskFactory.create_object_detection(use_dsp=True)
    detection1.name = "ObjectDetect_Main"
    tasks.add_task(detection1)
    
    detection2 = TaskFactory.create_object_detection(use_dsp=False)
    detection2.name = "ObjectDetect_Fast"
    tasks.add_task(detection2)
    
    # Analytics tasks
    analytics1 = TaskFactory.create_analytics_task()
    analytics1.name = "Analytics_RT"
    analytics1.priority = TaskPriority.HIGH
    tasks.add_task(analytics1)
    
    analytics2 = TaskFactory.create_analytics_task()
    analytics2.name = "Analytics_Batch"
    analytics2.constraints.fps_requirement = 2.0
    tasks.add_task(analytics2)
    
    # Custom complex task with multiple segments
    complex_task = NNTask(
        name="MultiStageVision",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.DSP_RUNTIME,
        segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION
    )
    
    # Add DSP preprocessing
    complex_task.add_dsp_npu_sequence([
        (ResourceType.DSP, {2.0: 8.0, 4.0: 5.0, 8.0: 3.0}, "preprocess"),
        (ResourceType.NPU, {2.0: 40.0, 4.0: 25.0, 8.0: 15.0, 16.0: 10.0}, "inference"),
        (ResourceType.DSP, {2.0: 5.0, 4.0: 3.0, 8.0: 2.0}, "postprocess")
    ])
    
    complex_task.constraints.fps_requirement = 10.0
    complex_task.constraints.latency_requirement_ms = 100.0
    tasks.add_task(complex_task)
    
    return resources, tasks


def run_visualization_demo():
    """Run comprehensive visualization demonstration"""
    print("=" * 80)
    print("Enhanced Scheduler Visualization Demo")
    print("=" * 80)
    
    # Create scenario
    resources, tasks = create_complex_scenario()
    
    print(f"\nCreated scenario with:")
    print(f"  - {len(resources)} resources ({len([r for r in resources.values() if r.resource_type == ResourceType.DSP])} DSPs, "
          f"{len([r for r in resources.values() if r.resource_type == ResourceType.NPU])} NPUs)")
    print(f"  - {len(tasks.tasks)} tasks")
    
    # Test different schedulers
    schedulers = {
        "Simple": SimpleScheduler(resources),
        "PriorityQueue": PriorityQueueScheduler(resources),
        "Genetic": GeneticScheduler(resources, population_size=30, generations=20)
    }
    
    # Create output directory
    output_dir = f"visualization_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    for sched_name, scheduler in schedulers.items():
        print(f"\n{'=' * 60}")
        print(f"Testing {sched_name} Scheduler")
        print('=' * 60)
        
        # Generate schedule
        print("Generating schedule...")
        schedule = scheduler.schedule(tasks, time_limit_ms=500.0)
        
        if not schedule:
            print(f"WARNING: {sched_name} produced empty schedule!")
            continue
        
        # Create visualizer
        visualizer = EnhancedSchedulerVisualizer(resources, tasks)
        
        # 1. Print detailed schedule
        print("\n--- DETAILED SCHEDULE OUTPUT ---")
        visualizer.print_detailed_schedule(schedule)
        
        # 2. Generate Gantt chart
        gantt_path = os.path.join(output_dir, f"{sched_name}_gantt.png")
        print(f"\nGenerating Gantt chart: {gantt_path}")
        visualizer.visualize_schedule_gantt(
            schedule,
            title=f"{sched_name} Scheduler - Gantt Chart",
            save_path=gantt_path
        )
        
        # 3. Export Chrome tracing
        trace_path = os.path.join(output_dir, f"{sched_name}_trace.json")
        print(f"Exporting Chrome trace: {trace_path}")
        visualizer.export_chrome_tracing(schedule, trace_path)
        
        # 4. Generate timeline view
        timeline_path = os.path.join(output_dir, f"{sched_name}_timeline.png")
        print(f"Generating timeline view: {timeline_path}")
        visualizer.visualize_resource_timeline(schedule, save_path=timeline_path)
        
        # Calculate and print metrics
        metrics = scheduler.calculate_metrics(schedule)
        print(f"\n--- PERFORMANCE METRICS ---")
        print(f"  Makespan: {metrics.makespan_ms:.2f} ms")
        print(f"  Average latency: {metrics.average_latency_ms:.2f} ms")
        print(f"  Completed tasks: {metrics.completed_tasks}")
        print(f"  Deadline miss rate: {metrics.deadline_miss_rate:.2%}")
        
        # Resource utilization
        print(f"\n  Resource utilization:")
        for rtype, util in metrics.average_utilization.items():
            print(f"    {rtype.value}: {util:.1%}")
    
    print(f"\n{'=' * 80}")
    print("Visualization Demo Complete!")
    print(f"{'=' * 80}")
    print(f"\nAll outputs saved to: {output_dir}/")
    print("\nTo view Chrome traces:")
    print("1. Open Chrome browser")
    print("2. Navigate to chrome://tracing")
    print("3. Click 'Load' and select a *_trace.json file")
    print("\nThe traces show:")
    print("- Process view: Different resource types (DSP, NPU)")
    print("- Thread view: Individual resources")
    print("- Task flow: Dependencies between segments")
    print("- Timing details: Hover over events for details")


def demonstrate_segmentation_visualization():
    """Demonstrate visualization of network segmentation"""
    print("\n" + "=" * 80)
    print("Network Segmentation Visualization Demo")
    print("=" * 80)
    
    # Create simple setup
    resources = {
        "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
        "NPU_1": ResourceUnit("NPU_1", "NPU1", ResourceType.NPU, 8.0),
        "NPU_2": ResourceUnit("NPU_2", "NPU2", ResourceType.NPU, 8.0),
    }
    
    tasks = TaskSet()
    
    # Create task with segmentation
    seg_task = NNTask(
        name="SegmentedVision",
        priority=TaskPriority.HIGH,
        segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION
    )
    
    # Add segment with cut points
    from models import NetworkSegment
    segment = NetworkSegment(
        name="vision_inference",
        resource_type=ResourceType.NPU,
        duration_table={8.0: 60.0}
    )
    
    # Add cut points
    segment.add_cut_point(0.25, "layer_4", 0.12)
    segment.add_cut_point(0.50, "layer_8", 0.12)
    segment.add_cut_point(0.75, "layer_12", 0.12)
    
    seg_task.segments = [segment]
    seg_task.constraints.fps_requirement = 15.0
    seg_task.constraints.latency_requirement_ms = 80.0
    
    # Apply segmentation
    seg_task.apply_segmentation_strategy({ResourceType.NPU: 3})
    
    tasks.add_task(seg_task)
    
    # Schedule and visualize
    scheduler = SimpleScheduler(resources)
    schedule = scheduler.schedule(tasks, time_limit_ms=200.0)
    
    visualizer = EnhancedSchedulerVisualizer(resources, tasks)
    
    print("\nSegmentation applied:")
    for seg in seg_task.segments:
        print(f"  Segment '{seg.name}' split into {len(seg.sub_segments)} sub-segments")
    
    print("\nSchedule with segmentation:")
    visualizer.print_detailed_schedule(schedule)
    
    # Show in Chrome trace
    visualizer.export_chrome_tracing(schedule, "segmentation_demo_trace.json")
    print("\nSegmentation visible in Chrome trace as sub-segments")


if __name__ == "__main__":
    # Import required enums
    from enums import TaskPriority, RuntimeType, SegmentationStrategy
    
    # Run main demo
    run_visualization_demo()
    
    # Run segmentation demo
    demonstrate_segmentation_visualization()
