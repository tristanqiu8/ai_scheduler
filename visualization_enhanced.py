#!/usr/bin/env python3
"""
Enhanced visualization for heterogeneous scheduler with Chrome tracing support
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import seaborn as sns
from collections import defaultdict

from enums import ResourceType, TaskPriority, RuntimeType
from models import ResourceUnit, TaskScheduleInfo
from task import TaskSet, NNTask


class EnhancedSchedulerVisualizer:
    """Enhanced visualizer with hardware pipeline and Chrome tracing support"""
    
    def __init__(self, resources: Dict[str, ResourceUnit], tasks: TaskSet):
        self.resources = resources
        self.tasks = tasks
        
        # Color schemes
        self.priority_colors = {
            TaskPriority.CRITICAL: '#FF4444',
            TaskPriority.HIGH: '#FF8844',
            TaskPriority.NORMAL: '#4488FF',
            TaskPriority.LOW: '#44FF88'
        }
        
        self.resource_colors = {
            ResourceType.NPU: '#3498db',
            ResourceType.DSP: '#e74c3c',
            ResourceType.CPU: '#2ecc71'
        }
        
        # Create resource ordering for consistent display
        self.resource_order = []
        for rtype in [ResourceType.DSP, ResourceType.NPU, ResourceType.CPU]:
            type_resources = sorted([r for r in resources.values() 
                                   if r.resource_type == rtype],
                                  key=lambda x: x.id)
            self.resource_order.extend(type_resources)
    
    def visualize_schedule_gantt(self, schedule: List[TaskScheduleInfo], 
                               title: str = "Schedule Gantt Chart",
                               save_path: Optional[str] = None):
        """Create enhanced Gantt chart with all hardware pipelines"""
        fig, ax = plt.subplots(figsize=(16, 10))
        
        # Create y-axis positions
        y_positions = {res.id: i for i, res in enumerate(self.resource_order)}
        y_labels = []
        
        # Add resource type separators
        current_type = None
        separator_positions = []
        
        for i, res in enumerate(self.resource_order):
            if res.resource_type != current_type:
                current_type = res.resource_type
                if i > 0:
                    separator_positions.append(i - 0.5)
            y_labels.append(f"{res.id}\n({res.bandwidth:.1f})")
        
        # Plot scheduled tasks
        for sched in schedule:
            task = self.tasks.get_task(sched.task_id)
            if not task:
                continue
            
            # Plot each segment
            for seg_id, start, end, res_id in sched.segment_schedule:
                if res_id not in y_positions:
                    continue
                
                y_pos = y_positions[res_id]
                duration = end - start
                
                # Main rectangle
                rect = patches.Rectangle(
                    (start, y_pos - 0.4), duration, 0.8,
                    facecolor=self.priority_colors[task.priority],
                    edgecolor='black',
                    linewidth=1,
                    alpha=0.8
                )
                ax.add_patch(rect)
                
                # Task label
                label = f"{task.name[:8]}\n{duration:.1f}ms"
                ax.text(start + duration/2, y_pos, label,
                       ha='center', va='center',
                       fontsize=8, weight='bold',
                       color='white')
                
                # Runtime type indicator
                if task.runtime_type == RuntimeType.DSP_RUNTIME:
                    ax.text(start + 1, y_pos + 0.3, "B",
                           fontsize=6, color='yellow', weight='bold')
                else:
                    ax.text(start + 1, y_pos + 0.3, "P",
                           fontsize=6, color='lightgreen', weight='bold')
            
            # Plot sub-segments if available
            if sched.sub_segment_schedule:
                for sub_id, start, end, res_id in sched.sub_segment_schedule:
                    if res_id in y_positions:
                        y_pos = y_positions[res_id]
                        # Thin line for sub-segment
                        ax.plot([start, end], [y_pos - 0.45, y_pos - 0.45],
                               'k--', linewidth=0.5, alpha=0.5)
        
        # Add resource type separators
        for sep_pos in separator_positions:
            ax.axhline(y=sep_pos, color='gray', linestyle='--', alpha=0.5)
        
        # Formatting
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Resources', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Set x-axis limits
        if schedule:
            max_time = max(s.end_time_ms for s in schedule)
            ax.set_xlim(0, max_time * 1.1)
        
        # Add legend
        legend_elements = []
        for priority in TaskPriority:
            legend_elements.append(
                patches.Patch(facecolor=self.priority_colors[priority],
                            label=priority.name, alpha=0.8)
            )
        ax.legend(handles=legend_elements, loc='upper right',
                 title='Task Priority', framealpha=0.9)
        
        # Add resource type labels
        current_type = None
        for i, res in enumerate(self.resource_order):
            if res.resource_type != current_type:
                current_type = res.resource_type
                ax.text(-ax.get_xlim()[1] * 0.02, i, current_type.value,
                       ha='right', va='center', weight='bold',
                       fontsize=10, color=self.resource_colors[current_type])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gantt chart saved to {save_path}")
        
        plt.show()
    
    def export_chrome_tracing(self, schedule: List[TaskScheduleInfo], 
                            output_file: str = "schedule_trace.json"):
        """Export schedule to Chrome tracing format (chrome://tracing)"""
        events = []
        
        # Process ID mapping for different resource types
        pid_map = {
            ResourceType.DSP: 1,
            ResourceType.NPU: 2,
            ResourceType.CPU: 3
        }
        
        # Thread ID mapping for individual resources
        tid_map = {}
        tid_counter = 1
        for res in self.resource_order:
            tid_map[res.id] = tid_counter
            tid_counter += 1
        
        # Convert schedule to trace events
        for sched in schedule:
            task = self.tasks.get_task(sched.task_id)
            if not task:
                continue
            
            # Add task-level event
            task_event = {
                "name": f"{task.name} ({task.priority.name})",
                "cat": "task",
                "ph": "X",  # Complete event
                "ts": int(sched.start_time_ms * 1000),  # Convert to microseconds
                "dur": int((sched.end_time_ms - sched.start_time_ms) * 1000),
                "pid": 0,  # Overview process
                "tid": 0,  # Overview thread
                "args": {
                    "task_id": sched.task_id,
                    "priority": task.priority.name,
                    "runtime": task.runtime_type.value,
                    "latency_ms": sched.get_latency(),
                    "state": sched.state.value
                }
            }
            events.append(task_event)
            
            # Add segment-level events
            for seg_idx, (seg_id, start, end, res_id) in enumerate(sched.segment_schedule):
                if res_id not in tid_map:
                    continue
                
                resource = self.resources.get(res_id)
                if not resource:
                    continue
                
                segment_event = {
                    "name": f"{task.name}:{seg_id}",
                    "cat": "segment",
                    "ph": "X",
                    "ts": int(start * 1000),
                    "dur": int((end - start) * 1000),
                    "pid": pid_map.get(resource.resource_type, 0),
                    "tid": tid_map[res_id],
                    "args": {
                        "task": task.name,
                        "segment": seg_id,
                        "resource": res_id,
                        "bandwidth": resource.bandwidth,
                        "duration_ms": end - start
                    }
                }
                events.append(segment_event)
                
                # Add flow events to show dependencies
                if seg_idx > 0:
                    flow_event = {
                        "name": "segment_flow",
                        "cat": "flow",
                        "ph": "s",  # Flow start
                        "ts": int(sched.segment_schedule[seg_idx-1][2] * 1000),
                        "pid": pid_map.get(resource.resource_type, 0),
                        "tid": tid_map[sched.segment_schedule[seg_idx-1][3]],
                        "id": f"{sched.task_id}_{seg_idx}"
                    }
                    events.append(flow_event)
                    
                    flow_end = {
                        "name": "segment_flow",
                        "cat": "flow",
                        "ph": "f",  # Flow end
                        "ts": int(start * 1000),
                        "pid": pid_map.get(resource.resource_type, 0),
                        "tid": tid_map[res_id],
                        "id": f"{sched.task_id}_{seg_idx}"
                    }
                    events.append(flow_end)
            
            # Add sub-segment events if available
            if sched.sub_segment_schedule:
                for sub_id, start, end, res_id in sched.sub_segment_schedule:
                    if res_id not in tid_map:
                        continue
                    
                    resource = self.resources.get(res_id)
                    if not resource:
                        continue
                    
                    sub_event = {
                        "name": f"{task.name}:sub:{sub_id}",
                        "cat": "subsegment",
                        "ph": "X",
                        "ts": int(start * 1000),
                        "dur": int((end - start) * 1000),
                        "pid": pid_map.get(resource.resource_type, 0),
                        "tid": tid_map[res_id],
                        "args": {
                            "type": "subsegment",
                            "parent_task": task.name
                        }
                    }
                    events.append(sub_event)
        
        # Add metadata
        metadata_events = []
        
        # Process names
        for rtype, pid in pid_map.items():
            metadata_events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "args": {"name": f"{rtype.value} Resources"}
            })
        
        # Thread names (individual resources)
        for res_id, tid in tid_map.items():
            resource = self.resources.get(res_id)
            if resource:
                metadata_events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid_map.get(resource.resource_type, 0),
                    "tid": tid,
                    "args": {"name": f"{res_id} (BW:{resource.bandwidth})"}
                })
        
        # Combine all events
        trace_data = {
            "traceEvents": metadata_events + events,
            "displayTimeUnit": "ms",
            "metadata": {
                "generated_by": "Enhanced Scheduler Visualizer",
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Write to file
        with open(output_file, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"Chrome tracing file exported to: {output_file}")
        print(f"Open chrome://tracing in Chrome and load this file to visualize")
    
    def print_detailed_schedule(self, schedule: List[TaskScheduleInfo]):
        """Print complete schedule results in a formatted way"""
        print("\n" + "=" * 100)
        print("DETAILED SCHEDULE RESULTS")
        print("=" * 100)
        
        if not schedule:
            print("No tasks scheduled!")
            return
        
        # Summary statistics
        total_tasks = len(set(s.task_id for s in schedule))
        total_executions = len(schedule)
        makespan = max(s.end_time_ms for s in schedule)
        
        print(f"\nSUMMARY:")
        print(f"  Total unique tasks: {total_tasks}")
        print(f"  Total executions: {total_executions}")
        print(f"  Makespan: {makespan:.2f} ms")
        
        # Resource utilization
        print(f"\nRESOURCE UTILIZATION:")
        resource_usage = defaultdict(float)
        for sched in schedule:
            for seg_id, start, end, res_id in sched.segment_schedule:
                resource_usage[res_id] += end - start
        
        for res_id in sorted(resource_usage.keys()):
            resource = self.resources.get(res_id)
            if resource:
                utilization = (resource_usage[res_id] / makespan) * 100
                print(f"  {res_id} ({resource.resource_type.value}, BW:{resource.bandwidth}): "
                      f"{utilization:.1f}% ({resource_usage[res_id]:.1f} ms)")
        
        # Detailed schedule
        print(f"\nDETAILED SCHEDULE:")
        print("-" * 100)
        print(f"{'Time':>8} | {'Task':^20} | {'Priority':^8} | {'Runtime':^8} | "
              f"{'Resource':^12} | {'Duration':>8} | {'State':^10}")
        print("-" * 100)
        
        # Sort by start time
        sorted_schedule = sorted(schedule, key=lambda s: s.start_time_ms)
        
        for sched in sorted_schedule:
            task = self.tasks.get_task(sched.task_id)
            if not task:
                continue
            
            # Main task info
            print(f"{sched.start_time_ms:8.1f} | {task.name:^20} | "
                  f"{task.priority.name:^8} | {task.runtime_type.value:^8} | "
                  f"{'TASK':^12} | {sched.get_latency():8.1f} | "
                  f"{sched.state.value:^10}")
            
            # Segment details
            for seg_id, start, end, res_id in sched.segment_schedule:
                duration = end - start
                print(f"{start:8.1f} | {f'  └─ {seg_id[:15]}':20} | "
                      f"{'-':^8} | {'-':^8} | "
                      f"{res_id:^12} | {duration:8.1f} | "
                      f"{'SEGMENT':^10}")
            
            # Sub-segment details if available
            if sched.sub_segment_schedule:
                for sub_id, start, end, res_id in sched.sub_segment_schedule:
                    duration = end - start
                    print(f"{start:8.1f} | {f'    └─ {sub_id[:13]}':20} | "
                          f"{'-':^8} | {'-':^8} | "
                          f"{res_id:^12} | {duration:8.1f} | "
                          f"{'SUBSEG':^10}")
        
        print("-" * 100)
        
        # Task completion summary
        print(f"\nTASK COMPLETION SUMMARY:")
        task_stats = defaultdict(lambda: {'count': 0, 'total_latency': 0, 'deadlines_met': 0})
        
        for sched in schedule:
            task = self.tasks.get_task(sched.task_id)
            if task:
                stats = task_stats[sched.task_id]
                stats['count'] += 1
                stats['total_latency'] += sched.get_latency()
                if sched.get_latency() <= task.constraints.latency_requirement_ms:
                    stats['deadlines_met'] += 1
        
        for task_id, stats in task_stats.items():
            task = self.tasks.get_task(task_id)
            if task:
                avg_latency = stats['total_latency'] / stats['count']
                deadline_rate = (stats['deadlines_met'] / stats['count']) * 100
                print(f"  {task.name}: {stats['count']} executions, "
                      f"avg latency: {avg_latency:.1f} ms, "
                      f"deadline met: {deadline_rate:.0f}%")
    
    def visualize_resource_timeline(self, schedule: List[TaskScheduleInfo],
                                  save_path: Optional[str] = None):
        """Create a timeline view for each resource type"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        
        resource_types = [ResourceType.DSP, ResourceType.NPU, ResourceType.CPU]
        
        for idx, (ax, rtype) in enumerate(zip(axes, resource_types)):
            # Get resources of this type
            type_resources = [r for r in self.resource_order if r.resource_type == rtype]
            
            if not type_resources:
                ax.text(0.5, 0.5, f"No {rtype.value} resources", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_ylabel(rtype.value)
                continue
            
            # Create timeline for each resource
            for i, resource in enumerate(type_resources):
                y_base = i
                
                # Find all segments on this resource
                for sched in schedule:
                    task = self.tasks.get_task(sched.task_id)
                    if not task:
                        continue
                    
                    for seg_id, start, end, res_id in sched.segment_schedule:
                        if res_id == resource.id:
                            duration = end - start
                            
                            # Draw rectangle
                            rect = patches.Rectangle(
                                (start, y_base - 0.4), duration, 0.8,
                                facecolor=self.priority_colors[task.priority],
                                edgecolor='black',
                                alpha=0.8
                            )
                            ax.add_patch(rect)
                            
                            # Add task name
                            if duration > 5:  # Only add text if segment is wide enough
                                ax.text(start + duration/2, y_base,
                                       task.name[:10], ha='center', va='center',
                                       fontsize=8, color='white')
            
            # Formatting
            ax.set_ylim(-0.5, len(type_resources) - 0.5)
            ax.set_yticks(range(len(type_resources)))
            ax.set_yticklabels([r.id for r in type_resources])
            ax.set_ylabel(f"{rtype.value} Resources", fontsize=12)
            ax.grid(True, axis='x', alpha=0.3)
            ax.set_title(f"{rtype.value} Pipeline Timeline", fontsize=12)
        
        # Common x-axis
        if schedule:
            max_time = max(s.end_time_ms for s in schedule)
            axes[-1].set_xlim(0, max_time * 1.1)
        axes[-1].set_xlabel('Time (ms)', fontsize=12)
        
        plt.suptitle('Resource Pipeline Timeline View', fontsize=14, weight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def demonstrate_visualization():
    """Demonstrate the enhanced visualization capabilities"""
    from task import TaskFactory
    from scheduler_base import SimpleScheduler
    
    # Create resources
    resources = {
        "DSP_0": ResourceUnit("DSP_0", "DSP0", ResourceType.DSP, 4.0),
        "DSP_1": ResourceUnit("DSP_1", "DSP1", ResourceType.DSP, 4.0),
        "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
        "NPU_1": ResourceUnit("NPU_1", "NPU1", ResourceType.NPU, 4.0),
        "NPU_2": ResourceUnit("NPU_2", "NPU2", ResourceType.NPU, 2.0),
    }
    
    # Create tasks
    tasks = TaskSet()
    tasks.add_task(TaskFactory.create_safety_monitor())
    tasks.add_task(TaskFactory.create_object_detection(use_dsp=True))
    tasks.add_task(TaskFactory.create_analytics_task())
    
    # Generate schedule
    scheduler = SimpleScheduler(resources)
    schedule = scheduler.schedule(tasks, time_limit_ms=300.0)
    
    # Create visualizer
    visualizer = EnhancedSchedulerVisualizer(resources, tasks)
    
    # 1. Print detailed schedule
    visualizer.print_detailed_schedule(schedule)
    
    # 2. Create Gantt chart
    visualizer.visualize_schedule_gantt(schedule, 
                                      title="Heterogeneous DSP+NPU Schedule",
                                      save_path="schedule_gantt.png")
    
    # 3. Export Chrome tracing
    visualizer.export_chrome_tracing(schedule, "schedule_trace.json")
    
    # 4. Create timeline view
    visualizer.visualize_resource_timeline(schedule, 
                                         save_path="resource_timeline.png")


if __name__ == "__main__":
    demonstrate_visualization()
