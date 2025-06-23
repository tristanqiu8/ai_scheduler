#!/usr/bin/env python3
"""
Utility functions and helpers for the scheduler system
"""

import json
import yaml
import pickle
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from enums import ResourceType, TaskPriority, RuntimeType, SchedulerConfig
from models import ResourceUnit, TaskScheduleInfo, SchedulingMetrics, SystemState
from task import NNTask, TaskSet


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScheduleValidator:
    """Validate schedule correctness"""
    
    @staticmethod
    def validate_schedule(schedule: List[TaskScheduleInfo], 
                         resources: Dict[str, ResourceUnit],
                         tasks: TaskSet) -> Tuple[bool, List[str]]:
        """Comprehensive schedule validation"""
        errors = []
        warnings = []
        
        # Check 1: Resource conflicts
        resource_timeline = {}
        for res_id in resources:
            resource_timeline[res_id] = []
        
        for sched in schedule:
            for seg_id, start, end, res_id in sched.segment_schedule:
                if res_id not in resource_timeline:
                    errors.append(f"Unknown resource {res_id} in schedule")
                    continue
                
                # Check for overlaps
                for existing_start, existing_end, existing_task in resource_timeline[res_id]:
                    if not (end <= existing_start or start >= existing_end):
                        errors.append(
                            f"Resource conflict on {res_id}: "
                            f"Task {sched.task_id} ({start:.2f}-{end:.2f}) overlaps with "
                            f"Task {existing_task} ({existing_start:.2f}-{existing_end:.2f})"
                        )
                
                resource_timeline[res_id].append((start, end, sched.task_id))
        
        # Check 2: Task constraints
        task_schedules = {}
        for sched in schedule:
            task_id = sched.task_id
            if task_id not in task_schedules:
                task_schedules[task_id] = []
            task_schedules[task_id].append(sched)
        
        for task_id, schedules in task_schedules.items():
            task = tasks.get_task(task_id)
            if not task:
                errors.append(f"Unknown task {task_id} in schedule")
                continue
            
            # Check latency requirements
            for sched in schedules:
                latency = sched.get_latency()
                if latency > task.constraints.latency_requirement_ms:
                    warnings.append(
                        f"Task {task.name} exceeds latency requirement: "
                        f"{latency:.2f}ms > {task.constraints.latency_requirement_ms}ms"
                    )
            
            # Check FPS requirements
            if len(schedules) > 1:
                intervals = []
                for i in range(1, len(schedules)):
                    interval = schedules[i].start_time_ms - schedules[i-1].start_time_ms
                    intervals.append(interval)
                
                avg_interval = np.mean(intervals)
                expected_interval = task.constraints.get_period_ms()
                
                if avg_interval > expected_interval * 1.1:  # 10% tolerance
                    warnings.append(
                        f"Task {task.name} may not meet FPS requirement: "
                        f"avg interval {avg_interval:.2f}ms > expected {expected_interval:.2f}ms"
                    )
        
        # Check 3: Dependency violations
        completed_times = {}
        for sched in sorted(schedule, key=lambda s: s.end_time_ms):
            task = tasks.get_task(sched.task_id)
            if task:
                for dep_id in task.constraints.dependencies:
                    if dep_id not in completed_times:
                        errors.append(
                            f"Task {task.name} scheduled before dependency {dep_id}"
                        )
                    elif completed_times[dep_id] > sched.start_time_ms:
                        errors.append(
                            f"Task {task.name} starts before dependency {dep_id} completes"
                        )
                completed_times[sched.task_id] = sched.end_time_ms
        
        # Check 4: Runtime type constraints
        for sched in schedule:
            task = tasks.get_task(sched.task_id)
            if task and task.runtime_type == RuntimeType.DSP_RUNTIME:
                # Check resource binding
                resource_types = set()
                for seg_id, start, end, res_id in sched.segment_schedule:
                    if res_id in resources:
                        resource_types.add(resources[res_id].resource_type)
                
                if task.constraints.required_resources - resource_types:
                    errors.append(
                        f"DSP_RUNTIME task {task.name} missing required resources"
                    )
        
        is_valid = len(errors) == 0
        
        if warnings:
            logger.warning(f"Schedule validation warnings: {len(warnings)}")
            for warning in warnings[:5]:  # Show first 5 warnings
                logger.warning(f"  - {warning}")
        
        return is_valid, errors


class ScheduleAnalyzer:
    """Analyze schedule performance and characteristics"""
    
    @staticmethod
    def analyze_schedule(schedule: List[TaskScheduleInfo],
                        resources: Dict[str, ResourceUnit],
                        tasks: TaskSet) -> Dict[str, Any]:
        """Comprehensive schedule analysis"""
        analysis = {
            'summary': {},
            'resource_analysis': {},
            'task_analysis': {},
            'timing_analysis': {},
            'segmentation_analysis': {}
        }
        
        if not schedule:
            analysis['summary'] = {
                'total_tasks_scheduled': 0,
                'total_executions': 0,  # 确保这一行存在
                'makespan': 0,
                'average_latency': 0,
                'total_overhead': 0
            }
            return analysis
        
        # Summary statistics
        analysis['summary'] = {
            'total_tasks_scheduled': len(set(s.task_id for s in schedule)),
            'total_executions': len(schedule),
            'makespan': max(s.end_time_ms for s in schedule),
            'average_latency': np.mean([s.get_latency() for s in schedule]),
            'total_overhead': sum(s.total_overhead_ms for s in schedule)
        }
        
        # Resource analysis
        resource_usage = {res_id: {'busy_time': 0, 'tasks': set()} 
                         for res_id in resources}
        
        for sched in schedule:
            for seg_id, start, end, res_id in sched.segment_schedule:
                if res_id in resource_usage:
                    resource_usage[res_id]['busy_time'] += end - start
                    resource_usage[res_id]['tasks'].add(sched.task_id)
        
        makespan = analysis['summary']['makespan']
        for res_id, usage in resource_usage.items():
            resource = resources[res_id]
            analysis['resource_analysis'][res_id] = {
                'type': resource.resource_type.value,
                'bandwidth': resource.bandwidth,
                'utilization': usage['busy_time'] / makespan if makespan > 0 else 0,
                'task_count': len(usage['tasks']),
                'average_temp': resource.current_temp_c
            }
        
        # Task analysis
        task_stats = {}
        for sched in schedule:
            task_id = sched.task_id
            if task_id not in task_stats:
                task_stats[task_id] = {
                    'executions': 0,
                    'latencies': [],
                    'response_times': [],
                    'preemptions': 0
                }
            
            task_stats[task_id]['executions'] += 1
            task_stats[task_id]['latencies'].append(sched.get_latency())
            task_stats[task_id]['preemptions'] += sched.preemption_count
        
        for task_id, stats in task_stats.items():
            task = tasks.get_task(task_id)
            if task:
                analysis['task_analysis'][task_id] = {
                    'name': task.name,
                    'priority': task.priority.name,
                    'executions': stats['executions'],
                    'avg_latency': np.mean(stats['latencies']),
                    'max_latency': max(stats['latencies']),
                    'meets_requirements': all(
                        l <= task.constraints.latency_requirement_ms 
                        for l in stats['latencies']
                    ),
                    'preemptions': stats['preemptions']
                }
        
        # Timing analysis
        intervals = []
        task_intervals = {}
        
        sorted_schedule = sorted(schedule, key=lambda s: s.start_time_ms)
        for i in range(1, len(sorted_schedule)):
            if sorted_schedule[i].task_id == sorted_schedule[i-1].task_id:
                interval = sorted_schedule[i].start_time_ms - sorted_schedule[i-1].start_time_ms
                intervals.append(interval)
                
                task_id = sorted_schedule[i].task_id
                if task_id not in task_intervals:
                    task_intervals[task_id] = []
                task_intervals[task_id].append(interval)
        
        analysis['timing_analysis'] = {
            'avg_task_interval': np.mean(intervals) if intervals else 0,
            'interval_variance': np.var(intervals) if intervals else 0,
            'jitter': {}
        }
        
        for task_id, task_ints in task_intervals.items():
            if len(task_ints) > 1:
                analysis['timing_analysis']['jitter'][task_id] = np.std(task_ints)
        
        # Segmentation analysis
        total_segments = 0
        total_sub_segments = 0
        segmentation_overhead = 0
        
        for sched in schedule:
            total_segments += len(sched.segment_schedule)
            total_sub_segments += len(sched.sub_segment_schedule)
            segmentation_overhead += sched.total_overhead_ms
        
        analysis['segmentation_analysis'] = {
            'total_segments': total_segments,
            'total_sub_segments': total_sub_segments,
            'avg_segments_per_task': total_segments / len(schedule) if schedule else 0,
            'total_segmentation_overhead': segmentation_overhead,
            'overhead_percentage': (segmentation_overhead / makespan * 100) if makespan > 0 else 0
        }
        
        return analysis


class ScheduleOptimizer:
    """Post-process schedule optimization"""
    
    @staticmethod
    def compact_schedule(schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """Compact schedule to reduce makespan"""
        if not schedule:
            return schedule
        
        # Sort by start time
        sorted_schedule = sorted(schedule, key=lambda s: s.start_time_ms)
        compacted = []
        
        # Track resource availability
        resource_available_at = {}
        
        for sched in sorted_schedule:
            # Find earliest possible start time
            earliest_start = 0
            
            # Check resource availability
            for seg_id, start, end, res_id in sched.segment_schedule:
                if res_id in resource_available_at:
                    earliest_start = max(earliest_start, resource_available_at[res_id])
            
            # Check dependencies (simplified - would need full dependency info)
            
            # Shift schedule
            time_shift = earliest_start - sched.start_time_ms
            if time_shift < 0:  # Can start earlier
                new_sched = TaskScheduleInfo(
                    task_id=sched.task_id,
                    start_time_ms=earliest_start,
                    end_time_ms=sched.end_time_ms + time_shift,
                    actual_duration_ms=sched.actual_duration_ms
                )
                
                # Update segment times
                new_segments = []
                for seg_id, start, end, res_id in sched.segment_schedule:
                    new_segments.append((seg_id, start + time_shift, end + time_shift, res_id))
                    resource_available_at[res_id] = end + time_shift
                
                new_sched.segment_schedule = new_segments
                compacted.append(new_sched)
            else:
                compacted.append(sched)
                # Update resource availability
                for seg_id, start, end, res_id in sched.segment_schedule:
                    resource_available_at[res_id] = end
        
        return compacted
    
    @staticmethod
    def balance_resource_usage(schedule: List[TaskScheduleInfo],
                              resources: Dict[str, ResourceUnit]) -> List[TaskScheduleInfo]:
        """Balance load across resources"""
        # Calculate current resource usage
        resource_usage = {res_id: 0.0 for res_id in resources}
        
        for sched in schedule:
            for seg_id, start, end, res_id in sched.segment_schedule:
                if res_id in resource_usage:
                    resource_usage[res_id] += end - start
        
        # Identify overloaded and underloaded resources
        avg_usage = np.mean(list(resource_usage.values()))
        overloaded = {k: v for k, v in resource_usage.items() if v > avg_usage * 1.2}
        underloaded = {k: v for k, v in resource_usage.items() if v < avg_usage * 0.8}
        
        # TODO: Implement task migration logic
        # This would require re-evaluating task placement
        
        return schedule


class ScheduleSerializer:
    """Serialize and deserialize schedules"""
    
    @staticmethod
    def save_schedule(schedule: List[TaskScheduleInfo], 
                     filename: str,
                     format: str = 'json'):
        """Save schedule to file"""
        if format == 'json':
            data = []
            for sched in schedule:
                data.append({
                    'task_id': sched.task_id,
                    'schedule_id': sched.schedule_id,
                    'start_time': sched.start_time_ms,
                    'end_time': sched.end_time_ms,
                    'segments': sched.segment_schedule,
                    'state': sched.state.value,
                    'overhead': sched.total_overhead_ms
                })
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
                
        elif format == 'yaml':
            import yaml
            data = []
            for sched in schedule:
                data.append({
                    'task_id': sched.task_id,
                    'timing': {
                        'start': sched.start_time_ms,
                        'end': sched.end_time_ms,
                        'duration': sched.actual_duration_ms
                    },
                    'resources': sched.resource_assignments
                })
            
            with open(filename, 'w') as f:
                yaml.dump(data, f)
                
        elif format == 'pickle':
            with open(filename, 'wb') as f:
                pickle.dump(schedule, f)
                
        logger.info(f"Saved schedule to {filename} ({format} format)")
    
    @staticmethod
    def load_schedule(filename: str, format: str = 'json') -> List[TaskScheduleInfo]:
        """Load schedule from file"""
        if format == 'json':
            with open(filename, 'r') as f:
                data = json.load(f)
            
            schedule = []
            for item in data:
                sched = TaskScheduleInfo(
                    task_id=item['task_id'],
                    schedule_id=item.get('schedule_id', ''),
                    start_time_ms=item['start_time'],
                    end_time_ms=item['end_time']
                )
                sched.segment_schedule = item.get('segments', [])
                schedule.append(sched)
            
            return schedule
            
        elif format == 'pickle':
            with open(filename, 'rb') as f:
                return pickle.load(f)
        
        else:
            raise ValueError(f"Unsupported format: {format}")


class PerformanceProfiler:
    """Profile scheduler performance"""
    
    def __init__(self):
        self.timing_data = {}
        self.memory_data = {}
        
    def profile_scheduler(self, scheduler, tasks: TaskSet, 
                         time_limit: float, runs: int = 5) -> Dict[str, Any]:
        """Profile scheduler performance"""
        import time
        import tracemalloc
        
        results = {
            'algorithm': scheduler.get_algorithm_name(),
            'runs': runs,
            'timing': [],
            'memory': [],
            'metrics': []
        }
        
        for run in range(runs):
            # Reset state
            scheduler.reset()
            
            # Memory profiling
            tracemalloc.start()
            
            # Time profiling
            start_time = time.time()
            
            # Run scheduler
            schedule = scheduler.schedule(tasks, time_limit)
            
            # Stop profiling
            end_time = time.time()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate metrics
            metrics = scheduler.calculate_metrics(schedule)
            
            # Store results
            results['timing'].append(end_time - start_time)
            results['memory'].append(peak / 1024 / 1024)  # Convert to MB
            results['metrics'].append({
                'makespan': metrics.makespan_ms,
                'avg_latency': metrics.average_latency_ms,
                'completed_tasks': metrics.completed_tasks
            })
        
        # Calculate statistics
        results['stats'] = {
            'avg_time': np.mean(results['timing']),
            'std_time': np.std(results['timing']),
            'avg_memory': np.mean(results['memory']),
            'avg_makespan': np.mean([m['makespan'] for m in results['metrics']])
        }
        
        return results


class VisualizationHelper:
    """Advanced visualization utilities"""
    
    @staticmethod
    def create_resource_heatmap(schedule: List[TaskScheduleInfo],
                               resources: Dict[str, ResourceUnit],
                               time_resolution: float = 1.0):
        """Create resource utilization heatmap"""
        if not schedule:
            return
        
        # Create time bins
        max_time = max(s.end_time_ms for s in schedule)
        time_bins = np.arange(0, max_time + time_resolution, time_resolution)
        
        # Create utilization matrix
        resource_list = list(resources.keys())
        utilization_matrix = np.zeros((len(resource_list), len(time_bins) - 1))
        
        # Fill matrix
        for sched in schedule:
            for seg_id, start, end, res_id in sched.segment_schedule:
                if res_id in resource_list:
                    res_idx = resource_list.index(res_id)
                    
                    # Find time bins
                    start_bin = int(start / time_resolution)
                    end_bin = int(end / time_resolution)
                    
                    for bin_idx in range(start_bin, min(end_bin + 1, len(time_bins) - 1)):
                        utilization_matrix[res_idx, bin_idx] = 1
        
        # Create heatmap
        plt.figure(figsize=(12, 6))
        sns.heatmap(utilization_matrix, 
                   yticklabels=resource_list,
                   xticklabels=[f"{t:.0f}" for t in time_bins[::10]],
                   cmap='RdYlBu_r',
                   cbar_kws={'label': 'Utilization'})
        
        plt.xlabel('Time (ms)')
        plt.ylabel('Resources')
        plt.title('Resource Utilization Heatmap')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_task_timeline(schedule: List[TaskScheduleInfo],
                           tasks: TaskSet):
        """Create task execution timeline"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Group schedules by task
        task_schedules = {}
        for sched in schedule:
            if sched.task_id not in task_schedules:
                task_schedules[sched.task_id] = []
            task_schedules[sched.task_id].append(sched)
        
        # Plot each task
        y_position = 0
        colors = plt.cm.Set3(np.linspace(0, 1, len(task_schedules)))
        
        for (task_id, schedules), color in zip(task_schedules.items(), colors):
            task = tasks.get_task(task_id)
            if not task:
                continue
            
            # Plot executions
            for sched in schedules:
                ax.barh(y_position, sched.get_latency(),
                       left=sched.start_time_ms,
                       height=0.8, color=color,
                       alpha=0.8, edgecolor='black',
                       label=task.name if sched == schedules[0] else "")
                
                # Add segments
                segment_y = y_position - 0.3
                for seg_id, start, end, res_id in sched.segment_schedule:
                    ax.barh(segment_y, end - start,
                           left=start, height=0.3,
                           color=color, alpha=0.5)
            
            y_position += 1
        
        # Formatting
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Tasks')
        ax.set_title('Task Execution Timeline')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def create_performance_radar(metrics_dict: Dict[str, SchedulingMetrics]):
        """Create radar chart comparing scheduler performance"""
        categories = ['Makespan', 'Latency', 'Utilization', 
                     'Deadline Met', 'Energy Eff.']
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        
        # Complete the circle
        categories += categories[:1]
        angles += angles[:1]
        
        # Plot each scheduler
        for name, metrics in metrics_dict.items():
            # Normalize values (0-1 scale)
            values = [
                1.0 / (metrics.makespan_ms / 100 + 1),  # Inverse, normalized
                1.0 / (metrics.average_latency_ms / 10 + 1),
                np.mean(list(metrics.average_utilization.values())),
                1.0 - metrics.deadline_miss_rate,
                1.0 / (metrics.energy_per_task_j + 1)
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=name)
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_title('Scheduler Performance Comparison', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.show()


# Configuration loader
def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from file"""
    with open(config_file, 'r') as f:
        if config_file.endswith('.json'):
            return json.load(f)
        elif config_file.endswith('.yaml') or config_file.endswith('.yml'):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_file}")


# Experiment runner
class ExperimentRunner:
    """Run scheduling experiments"""
    
    def __init__(self, output_dir: str = "experiments"):
        self.output_dir = output_dir
        self.results = []
        
    def run_experiment(self, name: str, 
                      schedulers: Dict[str, Any],
                      task_sets: Dict[str, TaskSet],
                      resources: Dict[str, ResourceUnit],
                      time_limits: List[float],
                      runs_per_config: int = 3):
        """Run comprehensive experiment"""
        
        experiment_results = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'configurations': [],
            'results': []
        }
        
        for task_set_name, tasks in task_sets.items():
            for time_limit in time_limits:
                for scheduler_name, scheduler in schedulers.items():
                    
                    config = {
                        'task_set': task_set_name,
                        'time_limit': time_limit,
                        'scheduler': scheduler_name
                    }
                    
                    config_results = []
                    
                    for run in range(runs_per_config):
                        # Reset scheduler
                        scheduler.reset()
                        
                        # Run scheduling
                        start_time = time.time()
                        schedule = scheduler.schedule(tasks, time_limit)
                        execution_time = time.time() - start_time
                        
                        # Calculate metrics
                        metrics = scheduler.calculate_metrics(schedule)
                        
                        # Validate schedule
                        validator = ScheduleValidator()
                        is_valid, errors = validator.validate_schedule(
                            schedule, resources, tasks
                        )
                        
                        # Analyze schedule
                        analyzer = ScheduleAnalyzer()
                        analysis = analyzer.analyze_schedule(
                            schedule, resources, tasks
                        )
                        
                        run_result = {
                            'run': run,
                            'execution_time': execution_time,
                            'is_valid': is_valid,
                            'errors': errors,
                            'metrics': {
                                'makespan': metrics.makespan_ms,
                                'avg_latency': metrics.average_latency_ms,
                                'completed_tasks': metrics.completed_tasks,
                                'deadline_miss_rate': metrics.deadline_miss_rate,
                                'avg_utilization': np.mean(list(metrics.average_utilization.values()))
                            },
                            'analysis': analysis['summary']
                        }
                        
                        config_results.append(run_result)
                    
                    experiment_results['results'].append({
                        'config': config,
                        'runs': config_results
                    })
        
        # Save results
        output_file = f"{self.output_dir}/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(experiment_results, f, indent=2)
        
        logger.info(f"Experiment results saved to {output_file}")
        
        return experiment_results
    
    def generate_report(self, experiment_results: Dict[str, Any]):
        """Generate experiment report"""
        # Create summary DataFrame
        data = []
        for result in experiment_results['results']:
            config = result['config']
            
            # Average across runs
            avg_metrics = {}
            for metric in ['execution_time', 'makespan', 'avg_latency', 
                          'completed_tasks', 'deadline_miss_rate']:
                values = []
                for run in result['runs']:
                    if metric in run:
                        values.append(run[metric])
                    elif metric in run.get('metrics', {}):
                        values.append(run['metrics'][metric])
                
                avg_metrics[metric] = np.mean(values) if values else 0
            
            data.append({
                'Task Set': config['task_set'],
                'Time Limit': config['time_limit'],
                'Scheduler': config['scheduler'],
                **avg_metrics
            })
        
        df = pd.DataFrame(data)
        
        # Generate visualizations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Experiment: {experiment_results['name']}", fontsize=16)
        
        # Makespan comparison
        pivot_makespan = df.pivot(index='Task Set', columns='Scheduler', values='makespan')
        pivot_makespan.plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Average Makespan')
        axes[0, 0].set_ylabel('Time (ms)')
        
        # Latency comparison  
        pivot_latency = df.pivot(index='Task Set', columns='Scheduler', values='avg_latency')
        pivot_latency.plot(kind='bar', ax=axes[0, 1])
        axes[0, 1].set_title('Average Task Latency')
        axes[0, 1].set_ylabel('Time (ms)')
        
        # Deadline miss rate
        pivot_deadline = df.pivot(index='Task Set', columns='Scheduler', values='deadline_miss_rate')
        pivot_deadline.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Deadline Miss Rate')
        axes[1, 0].set_ylabel('Miss Rate')
        
        # Execution time
        pivot_exec = df.pivot(index='Task Set', columns='Scheduler', values='execution_time')
        pivot_exec.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Algorithm Execution Time')
        axes[1, 1].set_ylabel('Time (s)')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{experiment_results['name']}_summary.png")
        plt.show()
        
        return df
