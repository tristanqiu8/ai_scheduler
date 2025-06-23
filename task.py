#!/usr/bin/env python3
"""
Task definition and management for heterogeneous DSP+NPU scheduler
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import uuid
import json
from abc import ABC, abstractmethod

from enums import (
    ResourceType, TaskPriority, RuntimeType, SegmentationStrategy,
    TaskState, SchedulerConfig
)
from models import NetworkSegment, CutPoint, SubSegment, TaskScheduleInfo


@dataclass
class TaskConstraints:
    """Task execution constraints"""
    # Timing constraints
    fps_requirement: float = 30.0  # Frames per second
    latency_requirement_ms: float = 100.0  # Max acceptable latency
    deadline_ms: Optional[float] = None  # Absolute deadline
    
    # Resource constraints
    required_resources: Set[ResourceType] = field(default_factory=set)
    excluded_resources: Set[str] = field(default_factory=set)  # Resource IDs to avoid
    preferred_resources: Set[str] = field(default_factory=set)  # Preferred resource IDs
    
    # Memory constraints
    max_memory_mb: int = 512
    min_memory_mb: int = 32
    
    # Dependency constraints
    dependencies: Set[str] = field(default_factory=set)  # Task IDs that must complete first
    anti_dependencies: Set[str] = field(default_factory=set)  # Tasks that must not run concurrently
    
    # Segmentation constraints
    max_segments: int = 4
    max_overhead_ms: float = 10.0
    min_segment_size_ms: float = 1.0
    
    def get_period_ms(self) -> float:
        """Get period based on FPS requirement"""
        return 1000.0 / self.fps_requirement if self.fps_requirement > 0 else float('inf')
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate constraints for consistency"""
        errors = []
        
        if self.fps_requirement <= 0:
            errors.append("FPS requirement must be positive")
        
        if self.latency_requirement_ms <= 0:
            errors.append("Latency requirement must be positive")
        
        if self.get_period_ms() < self.latency_requirement_ms:
            errors.append("Period shorter than latency requirement")
        
        if self.max_memory_mb < self.min_memory_mb:
            errors.append("Max memory less than min memory")
        
        if self.max_segments < 1:
            errors.append("Max segments must be at least 1")
        
        return len(errors) == 0, errors


@dataclass
class NNTask:
    """Neural Network task definition"""
    # Identity
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    
    # Scheduling parameters
    priority: TaskPriority = TaskPriority.NORMAL
    runtime_type: RuntimeType = RuntimeType.ACPU_RUNTIME
    segmentation_strategy: SegmentationStrategy = SegmentationStrategy.ADAPTIVE_SEGMENTATION
    
    # Network structure
    segments: List[NetworkSegment] = field(default_factory=list)
    
    # Constraints
    constraints: TaskConstraints = field(default_factory=TaskConstraints)
    
    # State tracking
    state: TaskState = TaskState.PENDING
    last_scheduled_ms: float = 0.0
    completion_count: int = 0
    failure_count: int = 0
    
    # Performance history
    latency_history: List[float] = field(default_factory=list)
    deadline_miss_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Task_{self.id[:8]}"
    
    def add_segment(self, segment: NetworkSegment) -> None:
        """Add a network segment to the task"""
        self.segments.append(segment)
        
        # Update required resources
        self.constraints.required_resources.add(segment.resource_type)
        if segment.requires_dsp:
            self.constraints.required_resources.add(ResourceType.DSP)
    
    def add_dsp_npu_sequence(self, sequences: List[Tuple[ResourceType, Dict[float, float], str]]):
        """Add a sequence of DSP and NPU segments"""
        for i, (resource_type, duration_table, name) in enumerate(sequences):
            segment = NetworkSegment(
                name=name or f"{self.name}_seg_{i}",
                resource_type=resource_type,
                duration_table=duration_table,
                dsp_optimized=(resource_type == ResourceType.DSP)
            )
            self.add_segment(segment)
    
    def set_npu_only(self, duration_table: Dict[float, float], name: str = ""):
        """Configure as NPU-only task"""
        segment = NetworkSegment(
            name=name or f"{self.name}_npu",
            resource_type=ResourceType.NPU,
            duration_table=duration_table
        )
        self.segments = [segment]
        self.constraints.required_resources = {ResourceType.NPU}
    
    def add_cut_points_to_segment(self, segment_index: int, 
                                 cut_points: List[Tuple[float, str, float]]):
        """Add cut points to a specific segment"""
        if 0 <= segment_index < len(self.segments):
            segment = self.segments[segment_index]
            for position, name, overhead in cut_points:
                segment.add_cut_point(position, name, overhead)
    
    def get_total_duration(self, resource_bandwidths: Dict[str, float], 
                          with_overhead: bool = True) -> float:
        """Calculate total execution duration"""
        total_duration = 0.0
        
        for segment in self.segments:
            # Find appropriate bandwidth for segment
            bandwidth = 4.0  # Default
            for res_id, bw in resource_bandwidths.items():
                # Match resource type (simplified)
                bandwidth = bw
                break
            
            duration = segment.get_total_duration(bandwidth, with_overhead)
            total_duration += duration
        
        return total_duration
    
    def apply_segmentation_strategy(self, available_resources: Dict[ResourceType, int]) -> Dict[str, List[str]]:
        """Apply segmentation strategy to determine cut points"""
        segmentation_decisions = {}
        
        for segment in self.segments:
            if self.segmentation_strategy == SegmentationStrategy.NO_SEGMENTATION:
                segmentation_decisions[segment.id] = []
                
            elif self.segmentation_strategy == SegmentationStrategy.AGGRESSIVE_SEGMENTATION:
                # Use all available cut points
                segmentation_decisions[segment.id] = [cp.id for cp in segment.cut_points]
                
            elif self.segmentation_strategy == SegmentationStrategy.BALANCED_SEGMENTATION:
                # Use half of available cut points
                n_cuts = len(segment.cut_points) // 2
                segmentation_decisions[segment.id] = [cp.id for cp in segment.cut_points[:n_cuts]]
                
            elif self.segmentation_strategy == SegmentationStrategy.ADAPTIVE_SEGMENTATION:
                # Adapt based on priority and resources
                available = available_resources.get(segment.resource_type, 1)
                
                if self.priority == TaskPriority.CRITICAL and available > 2:
                    # Aggressive for critical tasks with resources
                    n_cuts = min(len(segment.cut_points), available - 1)
                elif self.priority == TaskPriority.HIGH and available > 1:
                    # Moderate for high priority
                    n_cuts = min(len(segment.cut_points) // 2, available - 1)
                else:
                    # Conservative otherwise
                    n_cuts = 0
                
                # Select cuts with lowest overhead
                sorted_cuts = sorted(segment.cut_points, key=lambda cp: cp.overhead_ms)
                segmentation_decisions[segment.id] = [cp.id for cp in sorted_cuts[:n_cuts]]
        
        # Apply decisions
        for segment_id, cut_ids in segmentation_decisions.items():
            segment = next((s for s in self.segments if s.id == segment_id), None)
            if segment:
                segment.apply_segmentation(cut_ids)
        
        return segmentation_decisions
    
    def check_dependencies_met(self, completed_tasks: Set[str]) -> bool:
        """Check if all dependencies are satisfied"""
        return self.constraints.dependencies.issubset(completed_tasks)
    
    def check_anti_dependencies(self, running_tasks: Set[str]) -> bool:
        """Check if anti-dependencies are satisfied"""
        return len(self.constraints.anti_dependencies.intersection(running_tasks)) == 0
    
    def update_performance_history(self, latency_ms: float):
        """Update performance tracking"""
        self.latency_history.append(latency_ms)
        
        # Keep last 100 entries
        if len(self.latency_history) > 100:
            self.latency_history = self.latency_history[-100:]
        
        # Check deadline miss
        if latency_ms > self.constraints.latency_requirement_ms:
            self.deadline_miss_count += 1
    
    def get_average_latency(self) -> float:
        """Get average historical latency"""
        if not self.latency_history:
            return 0.0
        return sum(self.latency_history) / len(self.latency_history)
    
    def get_deadline_miss_rate(self) -> float:
        """Get deadline miss rate"""
        total = self.completion_count + self.deadline_miss_count
        if total == 0:
            return 0.0
        return self.deadline_miss_count / total
    
    def estimate_memory_usage(self) -> int:
        """Estimate peak memory usage"""
        if not self.segments:
            return 0
        
        if self.runtime_type == RuntimeType.DSP_RUNTIME:
            # All segments may need to be in memory
            return sum(seg.estimate_memory_usage() for seg in self.segments)
        else:
            # Only max of individual segments
            return max(seg.estimate_memory_usage() for seg in self.segments)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'priority': self.priority.name,
            'runtime_type': self.runtime_type.value,
            'segmentation_strategy': self.segmentation_strategy.value,
            'segments': [
                {
                    'id': seg.id,
                    'name': seg.name,
                    'resource_type': seg.resource_type.value,
                    'duration_table': seg.duration_table,
                    'cut_points': [cp.to_dict() for cp in seg.cut_points],
                    'is_segmented': seg.is_segmented,
                    'active_cuts': seg.active_cuts
                }
                for seg in self.segments
            ],
            'constraints': {
                'fps': self.constraints.fps_requirement,
                'latency_ms': self.constraints.latency_requirement_ms,
                'dependencies': list(self.constraints.dependencies),
                'max_segments': self.constraints.max_segments
            },
            'state': self.state.value,
            'completion_count': self.completion_count,
            'average_latency': self.get_average_latency(),
            'deadline_miss_rate': self.get_deadline_miss_rate()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'NNTask':
        """Create task from dictionary"""
        task = cls(
            id=data['id'],
            name=data['name'],
            priority=TaskPriority[data['priority']],
            runtime_type=RuntimeType(data['runtime_type']),
            segmentation_strategy=SegmentationStrategy(data['segmentation_strategy'])
        )
        
        # Restore constraints
        task.constraints.fps_requirement = data['constraints']['fps']
        task.constraints.latency_requirement_ms = data['constraints']['latency_ms']
        task.constraints.dependencies = set(data['constraints']['dependencies'])
        
        # Restore segments
        for seg_data in data['segments']:
            segment = NetworkSegment(
                id=seg_data['id'],
                name=seg_data['name'],
                resource_type=ResourceType(seg_data['resource_type']),
                duration_table=seg_data['duration_table']
            )
            
            # Restore cut points
            for cp_data in seg_data['cut_points']:
                segment.cut_points.append(CutPoint(
                    id=cp_data['id'],
                    name=cp_data['name'],
                    position=cp_data['position'],
                    overhead_ms=cp_data['overhead_ms']
                ))
            
            task.segments.append(segment)
        
        return task


class TaskFactory:
    """Factory for creating common task types"""
    
    @staticmethod
    def create_safety_monitor(fps: float = 30) -> NNTask:
        """Create a critical safety monitoring task"""
        task = NNTask(
            name="SafetyMonitor",
            priority=TaskPriority.CRITICAL,
            runtime_type=RuntimeType.ACPU_RUNTIME,
            segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION
        )
        
        task.set_npu_only(
            duration_table={2.0: 25.0, 4.0: 15.0, 8.0: 10.0},
            name="safety_inference"
        )
        
        task.constraints.fps_requirement = fps
        task.constraints.latency_requirement_ms = 30.0
        
        # Add cut points for parallel execution
        task.segments[0].add_cut_point(0.3, "backbone_end", 0.1)
        task.segments[0].add_cut_point(0.7, "neck_end", 0.12)
        
        return task
    
    @staticmethod
    def create_object_detection(use_dsp: bool = True) -> NNTask:
        """Create an object detection task"""
        task = NNTask(
            name="ObjectDetection",
            priority=TaskPriority.HIGH,
            runtime_type=RuntimeType.DSP_RUNTIME if use_dsp else RuntimeType.ACPU_RUNTIME,
            segmentation_strategy=SegmentationStrategy.BALANCED_SEGMENTATION
        )
        
        if use_dsp:
            # DSP preprocessing + NPU inference
            task.add_dsp_npu_sequence([
                (ResourceType.DSP, {4.0: 5.0, 8.0: 3.0}, "preprocessing"),
                (ResourceType.NPU, {2.0: 30.0, 4.0: 20.0, 8.0: 15.0}, "detection"),
                (ResourceType.DSP, {4.0: 3.0, 8.0: 2.0}, "postprocessing")
            ])
        else:
            # NPU only
            task.set_npu_only(
                duration_table={2.0: 35.0, 4.0: 25.0, 8.0: 18.0},
                name="detection_full"
            )
        
        task.constraints.fps_requirement = 20.0
        task.constraints.latency_requirement_ms = 50.0
        
        return task
    
    @staticmethod
    def create_analytics_task() -> NNTask:
        """Create a low-priority analytics task"""
        task = NNTask(
            name="Analytics",
            priority=TaskPriority.LOW,
            runtime_type=RuntimeType.ACPU_RUNTIME,
            segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION
        )
        
        task.set_npu_only(
            duration_table={2.0: 50.0, 4.0: 35.0, 8.0: 25.0},
            name="analytics_inference"
        )
        
        task.constraints.fps_requirement = 5.0
        task.constraints.latency_requirement_ms = 200.0
        
        return task


class TaskSet:
    """Collection of tasks with management functionality"""
    
    def __init__(self):
        self.tasks: Dict[str, NNTask] = {}
        
    def add_task(self, task: NNTask) -> None:
        """Add a task to the set"""
        self.tasks[task.id] = task
        
    def remove_task(self, task_id: str) -> bool:
        """Remove a task from the set"""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def get_task(self, task_id: str) -> Optional[NNTask]:
        """Get task by ID"""
        return self.tasks.get(task_id)
    
    def get_ready_tasks(self, current_time_ms: float, 
                       completed_tasks: Set[str]) -> List[NNTask]:
        """Get tasks ready for scheduling"""
        ready = []
        
        for task in self.tasks.values():
            # Check if enough time has passed since last execution
            time_since_last = current_time_ms - task.last_scheduled_ms
            if time_since_last < task.constraints.get_period_ms():
                continue
            
            # Check dependencies
            if not task.check_dependencies_met(completed_tasks):
                continue
            
            # Check state
            if task.state not in [TaskState.PENDING, TaskState.READY]:
                continue
            
            ready.append(task)
        
        return ready
    
    def get_tasks_by_priority(self, priority: TaskPriority) -> List[NNTask]:
        """Get all tasks with specific priority"""
        return [task for task in self.tasks.values() if task.priority == priority]
    
    def get_statistics(self) -> Dict:
        """Get task set statistics"""
        stats = {
            'total_tasks': len(self.tasks),
            'by_priority': {},
            'by_runtime': {},
            'average_latency': 0.0,
            'deadline_miss_rate': 0.0
        }
        
        # Count by priority
        for priority in TaskPriority:
            count = len(self.get_tasks_by_priority(priority))
            stats['by_priority'][priority.name] = count
        
        # Count by runtime
        for runtime in RuntimeType:
            count = len([t for t in self.tasks.values() if t.runtime_type == runtime])
            stats['by_runtime'][runtime.value] = count
        
        # Calculate averages
        total_latency = 0.0
        total_misses = 0
        total_completions = 0
        
        for task in self.tasks.values():
            if task.completion_count > 0:
                total_latency += task.get_average_latency() * task.completion_count
                total_completions += task.completion_count
                total_misses += task.deadline_miss_count
        
        if total_completions > 0:
            stats['average_latency'] = total_latency / total_completions
            stats['deadline_miss_rate'] = total_misses / total_completions
        
        return stats
    
    def save_to_file(self, filename: str):
        """Save task set to JSON file"""
        data = {
            'tasks': [task.to_dict() for task in self.tasks.values()]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self, filename: str):
        """Load task set from JSON file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.tasks.clear()
        for task_data in data['tasks']:
            task = NNTask.from_dict(task_data)
            self.tasks[task.id] = task
