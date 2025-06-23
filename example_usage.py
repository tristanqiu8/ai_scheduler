#!/usr/bin/env python3
"""
Example usage of the heterogeneous DSP+NPU scheduler system
"""

import time
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from models import ResourceUnit, NetworkSegment, CutPoint
from task import NNTask, TaskSet, TaskFactory
from scheduler_base import SimpleScheduler, PriorityQueueScheduler
from scheduler_advanced import GeneticScheduler, SimulatedAnnealingScheduler, HybridScheduler


def create_resources() -> Dict[str, ResourceUnit]:
    """Create heterogeneous resources"""
    resources = {}
    
    # NPUs with different capabilities
    resources["NPU_0"] = ResourceUnit(
        id="NPU_0",
        name="High-Performance NPU",
        resource_type=ResourceType.NPU,
        bandwidth=8.0,  # TOPS
        memory_mb=2048,
        max_power_w=15.0
    )
    
    resources["NPU_1"] = ResourceUnit(
        id="NPU_1", 
        name="Mid-Range NPU",
        resource_type=ResourceType.NPU,
        bandwidth=4.0,
        memory_mb=1024,
        max_power_w=10.0
    )
    
    resources["NPU_2"] = ResourceUnit(
        id="NPU_2",
        name="Low-Power NPU",
        resource_type=ResourceType.NPU,
        bandwidth=2.0,
        memory_mb=512,
        max_power_w=5.0
    )
    
    # DSPs
    resources["DSP_0"] = ResourceUnit(
        id="DSP_0",
        name="DSP Core 0",
        resource_type=ResourceType.DSP,
        bandwidth=4.0,
        memory_mb=256,
        max_power_w=8.0
    )
    
    resources["DSP_1"] = ResourceUnit(
        id="DSP_1",
        name="DSP Core 1", 
        resource_type=ResourceType.DSP,
        bandwidth=4.0,
        memory_mb=256,
        max_power_w=8.0
    )
    
    return resources


def create_example_tasks() -> TaskSet:
    """Create a diverse set of tasks"""
    task_set = TaskSet()
    
    # Task 1: Critical safety monitoring
    safety_task = TaskFactory.create_safety_monitor(fps=30)
    task_set.add_task(safety_task)
    
    # Task 2: Object detection with DSP preprocessing
    detection_task = TaskFactory.create_object_detection(use_dsp=True)
    task_set.add_task(detection_task)
    
    # Task 3: Custom high-priority vision task
    vision_task = NNTask(
        name="VisionProcessing",
        priority=TaskPriority.HIGH,
        runtime_type=RuntimeType.ACPU_RUNTIME,
        segmentation_strategy=SegmentationStrategy.BALANCED_SEGMENTATION
    )
    
    # Add NPU segment with cut points
    vision_segment = NetworkSegment(
        name="vision_inference",
        resource_type=ResourceType.NPU,
        duration_table={2.0: 40.0, 4.0: 25.0, 8.0: 18.0},
        memory_requirement_mb=256,
        compute_flops=1000000000
    )
    
    # Add cut points for potential segmentation
    vision_segment.add_cut_point(0.25, "backbone_1", 0.11)
    vision_segment.add_cut_point(0.5, "backbone_2", 0.12)
    vision_segment.add_cut_point(0.75, "head", 0.13)
    
    vision_task.add_segment(vision_segment)
    vision_task.constraints.fps_requirement = 15.0
    vision_task.constraints.latency_requirement_ms = 60.0
    
    task_set.add_task(vision_task)
    
    # Task 4: Low priority analytics
    analytics_task = TaskFactory.create_analytics_task()
    task_set.add_task(analytics_task)
    
    # Task 5: Normal priority sensor fusion
    fusion_task = NNTask(
        name="SensorFusion",
        priority=TaskPriority.NORMAL,
        runtime_type=RuntimeType.DSP_RUNTIME,
        segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
    )
    
    fusion_task.add_dsp_npu_sequence([
        (ResourceType.DSP, {4.0: 8.0, 8.0: 5.0}, "preprocessing"),
        (ResourceType.NPU, {2.0: 20.0, 4.0: 15.0, 8.0: 10.0}, "fusion_inference"),
        (ResourceType.DSP, {4.0: 5.0, 8.0: 3.0}, "postprocessing")
    ])
    
    fusion_task.constraints.fps_requirement = 10.0
    fusion_task.constraints.latency_requirement_ms = 100.0
    
    task_set.add_task(fusion_task)
    
    # Add task dependencies
    detection_task.constraints.dependencies.add(safety_task.id)
    fusion_task.constraints.dependencies.add(vision_task.id)
    
    return task_set


def run_scheduler_comparison(resources: Dict[str, ResourceUnit], 
                           tasks: TaskSet,
                           time_limit_ms: float = 1000.0):
    """Compare different scheduling algorithms"""
    
    schedulers = {
        "Simple Priority": SimpleScheduler(resources),
        "Priority Queue": PriorityQueueScheduler(resources),
        "Genetic Algorithm": GeneticScheduler(resources, population_size=30, generations=20),
        "Simulated Annealing": SimulatedAnnealingScheduler(resources),
        "Hybrid GA+SA": HybridScheduler(resources)
    }
    
    results = {}
    
    for name, scheduler in schedulers.items():
        print(f"\nRunning {name} scheduler...")
        start_time = time.time()
        
        # Reset tasks state
        for task in tasks.tasks.values():
            task.last_scheduled_ms = 0.0
            task.completion_count = 0
            task.state = TaskState.PENDING
        
        # Run scheduling
        schedule = scheduler.schedule(tasks, time_limit_ms)
        
        # Calculate metrics
        metrics = scheduler.calculate_metrics(schedule)
        
        execution_time = time.time() - start_time
        
        results[name] = {
            'schedule': schedule,
            'metrics': metrics,
            'execution_time': execution_time
        }
        
        print(f"  Completed in {execution_time:.2f}s")
        print(f"  Makespan: {metrics.makespan_ms:.2f}ms")
        print(f"  Avg Latency: {metrics.average_latency_ms:.2f}ms")
        print(f"  Completed Tasks: {metrics.completed_tasks}")
        print(f"  Deadline Miss Rate: {metrics.deadline_miss_rate:.2%}")
    
    return results


def visualize_results(results: Dict):
    """Visualize scheduling results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Scheduler Performance Comparison', fontsize=16)
    
    # Extract data
    schedulers = list(results.keys())
    makespans = [results[s]['metrics'].makespan_ms for s in schedulers]
    latencies = [results[s]['metrics'].average_latency_ms for s in schedulers]
    miss_rates = [results[s]['metrics'].deadline_miss_rate for s in schedulers]
    exec_times = [results[s]['execution_time'] for s in schedulers]
    
    # Makespan comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(schedulers, makespans)
    ax1.set_ylabel('Makespan (ms)')
    ax1.set_title('Total Schedule Length')
    ax1.tick_params(axis='x', rotation=45)
    
    # Color best performer
    min_idx = np.argmin(makespans)
    bars1[min_idx].set_color('green')
    
    # Average latency
    ax2 = axes[0, 1]
    bars2 = ax2.bar(schedulers, latencies)
    ax2.set_ylabel('Average Latency (ms)')
    ax2.set_title('Task Latency')
    ax2.tick_params(axis='x', rotation=45)
    
    min_idx = np.argmin(latencies)
    bars2[min_idx].set_color('green')
    
    # Deadline miss rate
    ax3 = axes[1, 0]
    bars3 = ax3.bar(schedulers, miss_rates)
    ax3.set_ylabel('Deadline Miss Rate')
    ax3.set_title('Reliability')
    ax3.tick_params(axis='x', rotation=45)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    
    min_idx = np.argmin(miss_rates)
    bars3[min_idx].set_color('green')
    
    # Execution time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(schedulers, exec_times)
    ax4.set_ylabel('Execution Time (s)')
    ax4.set_title('Algorithm Performance')
    ax4.tick_params(axis='x', rotation=45)
    
    min_idx = np.argmin(exec_times)
    bars4[min_idx].set_color('green')
    
    plt.tight_layout()
    plt.show()


def visualize_schedule_gantt(schedule: List, resources: Dict[str, ResourceUnit], 
                           title: str = "Schedule Gantt Chart"):
    """Create Gantt chart for a schedule"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create resource list
    resource_list = list(resources.keys())
    y_pos = {res_id: i for i, res_id in enumerate(resource_list)}
    
    # Colors for different task priorities
    colors = {
        0: 'red',      # CRITICAL
        1: 'orange',   # HIGH
        2: 'blue',     # NORMAL
        3: 'green'     # LOW
    }
    
    # Plot tasks
    for sched in schedule:
        for seg_id, start, end, res_id in sched.segment_schedule:
            if res_id in y_pos:
                # Get task priority for color
                # This is simplified - in real implementation would look up task
                priority = 2  # Default to NORMAL
                
                ax.barh(y_pos[res_id], end - start, left=start,
                       height=0.8, color=colors.get(priority, 'gray'),
                       alpha=0.8, edgecolor='black')
                
                # Add task label
                ax.text(start + (end - start) / 2, y_pos[res_id],
                       sched.task_id[:8], ha='center', va='center',
                       fontsize=8, color='white')
    
    # Formatting
    ax.set_yticks(range(len(resource_list)))
    ax.set_yticklabels(resource_list)
    ax.set_xlabel('Time (ms)')
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='CRITICAL'),
        Patch(facecolor='orange', label='HIGH'),
        Patch(facecolor='blue', label='NORMAL'),
        Patch(facecolor='green', label='LOW')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()


def main():
    """Main demonstration"""
    print("=== Heterogeneous DSP+NPU Scheduler Demo ===\n")
    
    # Create resources and tasks
    resources = create_resources()
    tasks = create_example_tasks()
    
    print(f"Created {len(resources)} resources:")
    for res_id, resource in resources.items():
        print(f"  {res_id}: {resource.resource_type.value}, "
              f"Bandwidth={resource.bandwidth}, Memory={resource.memory_mb}MB")
    
    print(f"\nCreated {len(tasks.tasks)} tasks:")
    for task in tasks.tasks.values():
        print(f"  {task.name}: Priority={task.priority.name}, "
              f"Runtime={task.runtime_type.value}, FPS={task.constraints.fps_requirement}")
    
    # Run comparison
    results = run_scheduler_comparison(resources, tasks, time_limit_ms=500.0)
    
    # Visualize results
    print("\nGenerating performance comparison charts...")
    visualize_results(results)
    
    # Show best scheduler's Gantt chart
    best_scheduler = min(results.items(), 
                        key=lambda x: x[1]['metrics'].makespan_ms)
    print(f"\nBest performing scheduler: {best_scheduler[0]}")
    
    visualize_schedule_gantt(
        best_scheduler[1]['schedule'], 
        resources,
        f"Schedule - {best_scheduler[0]}"
    )
    
    # Print detailed statistics
    print("\n=== Detailed Statistics ===")
    stats = tasks.get_statistics()
    print(f"Task Statistics:")
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  By priority: {stats['by_priority']}")
    print(f"  By runtime: {stats['by_runtime']}")
    
    # Save results
    print("\nSaving task set to file...")
    tasks.save_to_file("example_tasks.json")
    print("Saved to example_tasks.json")


if __name__ == "__main__":
    main()
