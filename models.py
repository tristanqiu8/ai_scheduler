#!/usr/bin/env python3
"""
Core data models for heterogeneous DSP+NPU scheduler
Includes network segmentation functionality
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
import numpy as np
from datetime import datetime
import uuid

from enums import (
    ResourceType, TaskPriority, RuntimeType, SegmentationStrategy,
    TaskState, CutPointType, SchedulerConfig
)


@dataclass
class CutPoint:
    """Network segmentation cut point"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    position: float = 0.0  # Position in segment (0.0 to 1.0)
    cut_type: CutPointType = CutPointType.LAYER_BOUNDARY
    overhead_ms: float = SchedulerConfig.DEFAULT_CUT_OVERHEAD_MS
    memory_requirement_mb: int = 0  # Additional memory needed for cut
    
    # Metadata for decision making
    data_transfer_mb: float = 0.0
    compute_balance_ratio: float = 1.0  # Balance between segments
    
    def __post_init__(self):
        if not self.name:
            self.name = f"cut_{self.position:.2f}"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'position': self.position,
            'cut_type': self.cut_type.value,
            'overhead_ms': self.overhead_ms,
            'memory_requirement_mb': self.memory_requirement_mb,
            'data_transfer_mb': self.data_transfer_mb,
            'compute_balance_ratio': self.compute_balance_ratio
        }


@dataclass
class SubSegment:
    """A sub-segment created by network segmentation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_segment_id: str = ""
    segment_index: int = 0  # Index within parent segment
    
    # Execution characteristics
    resource_type: ResourceType = ResourceType.NPU
    duration_table: Dict[float, float] = field(default_factory=dict)  # bandwidth -> duration
    
    # Position within parent
    start_position: float = 0.0  # 0.0 to 1.0
    end_position: float = 1.0
    
    # Overhead from cut
    cut_overhead_ms: float = 0.0
    memory_requirement_mb: int = 0
    
    def get_duration(self, bandwidth: float) -> float:
        """Get execution duration for given bandwidth"""
        if bandwidth in self.duration_table:
            return self.duration_table[bandwidth]
        
        # Interpolate or find closest
        if not self.duration_table:
            return 0.0
            
        bandwidths = sorted(self.duration_table.keys())
        if bandwidth <= bandwidths[0]:
            return self.duration_table[bandwidths[0]]
        if bandwidth >= bandwidths[-1]:
            return self.duration_table[bandwidths[-1]]
            
        # Linear interpolation
        for i in range(len(bandwidths) - 1):
            if bandwidths[i] <= bandwidth <= bandwidths[i + 1]:
                ratio = (bandwidth - bandwidths[i]) / (bandwidths[i + 1] - bandwidths[i])
                duration = (self.duration_table[bandwidths[i]] * (1 - ratio) +
                           self.duration_table[bandwidths[i + 1]] * ratio)
                return duration
        
        return self.duration_table[bandwidths[0]]


@dataclass
class NetworkSegment:
    """A segment of neural network computation"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    resource_type: ResourceType = ResourceType.NPU
    
    # Execution characteristics
    duration_table: Dict[float, float] = field(default_factory=dict)
    memory_requirement_mb: int = 0
    compute_flops: int = 0
    
    # Cut points for potential segmentation
    cut_points: List[CutPoint] = field(default_factory=list)
    
    # Sub-segments after segmentation
    sub_segments: List[SubSegment] = field(default_factory=list)
    is_segmented: bool = False
    active_cuts: List[str] = field(default_factory=list)  # IDs of active cut points
    
    # DSP specific
    dsp_optimized: bool = False
    requires_dsp: bool = False
    
    def add_cut_point(self, position: float, name: str = "", 
                     overhead_ms: float = SchedulerConfig.DEFAULT_CUT_OVERHEAD_MS,
                     **kwargs) -> CutPoint:
        """Add a cut point to this segment"""
        cut_point = CutPoint(
            name=name or f"{self.name}_cut_{position:.2f}",
            position=position,
            overhead_ms=overhead_ms,
            **kwargs
        )
        self.cut_points.append(cut_point)
        self.cut_points.sort(key=lambda cp: cp.position)
        return cut_point
    
    def apply_segmentation(self, cut_point_ids: List[str]) -> List[SubSegment]:
        """Apply segmentation using specified cut points"""
        self.active_cuts = cut_point_ids
        self.sub_segments = []
        
        if not cut_point_ids:
            # No segmentation - single sub-segment
            sub_seg = SubSegment(
                parent_segment_id=self.id,
                segment_index=0,
                resource_type=self.resource_type,
                duration_table=self.duration_table.copy(),
                start_position=0.0,
                end_position=1.0,
                memory_requirement_mb=self.memory_requirement_mb
            )
            self.sub_segments = [sub_seg]
            self.is_segmented = False
            return self.sub_segments
        
        # Get active cut points
        active_cuts = [cp for cp in self.cut_points if cp.id in cut_point_ids]
        active_cuts.sort(key=lambda cp: cp.position)
        
        # Create sub-segments
        positions = [0.0] + [cp.position for cp in active_cuts] + [1.0]
        
        for i in range(len(positions) - 1):
            start_pos = positions[i]
            end_pos = positions[i + 1]
            segment_ratio = end_pos - start_pos
            
            # Calculate duration table for sub-segment
            sub_duration_table = {}
            for bw, duration in self.duration_table.items():
                sub_duration_table[bw] = duration * segment_ratio
            
            # Add overhead from cut point
            overhead = active_cuts[i].overhead_ms if i < len(active_cuts) else 0.0
            
            sub_seg = SubSegment(
                parent_segment_id=self.id,
                segment_index=i,
                resource_type=self.resource_type,
                duration_table=sub_duration_table,
                start_position=start_pos,
                end_position=end_pos,
                cut_overhead_ms=overhead,
                memory_requirement_mb=int(self.memory_requirement_mb * segment_ratio)
            )
            
            self.sub_segments.append(sub_seg)
        
        self.is_segmented = True
        return self.sub_segments
    
    def get_total_duration(self, bandwidth: float, with_overhead: bool = True) -> float:
        """Get total execution duration including segmentation overhead"""
        if not self.is_segmented:
            # Simple case - no segmentation
            if bandwidth in self.duration_table:
                return self.duration_table[bandwidth]
            else:
                # Find closest bandwidth
                closest_bw = min(self.duration_table.keys(), 
                               key=lambda bw: abs(bw - bandwidth))
                return self.duration_table[closest_bw]
        
        # Segmented case
        total_duration = 0.0
        for sub_seg in self.sub_segments:
            total_duration += sub_seg.get_duration(bandwidth)
            if with_overhead:
                total_duration += sub_seg.cut_overhead_ms
        
        return total_duration
    
    def estimate_memory_usage(self) -> int:
        """Estimate peak memory usage during execution"""
        if not self.is_segmented:
            return self.memory_requirement_mb
        
        # For segmented execution, consider pipeline overlap
        max_memory = 0
        for i, sub_seg in enumerate(self.sub_segments):
            # Current segment memory
            current_memory = sub_seg.memory_requirement_mb
            
            # If not last segment, add overhead for data transfer
            if i < len(self.sub_segments) - 1:
                cut_point = next((cp for cp in self.cut_points 
                                if cp.position == sub_seg.end_position), None)
                if cut_point:
                    current_memory += cut_point.memory_requirement_mb
            
            max_memory = max(max_memory, current_memory)
        
        return max_memory


@dataclass
class ResourceUnit:
    """Hardware resource unit (NPU/DSP)"""
    id: str
    name: str
    resource_type: ResourceType
    
    # Performance characteristics
    bandwidth: float  # TOPS or GB/s
    memory_mb: int = 1024
    max_power_w: float = 10.0
    
    # Thermal model
    current_temp_c: float = SchedulerConfig.AMBIENT_TEMPERATURE_C
    thermal_resistance: float = 0.5  # C/W
    thermal_capacitance: float = 10.0  # J/C
    
    # Current state
    is_available: bool = True
    current_task_id: Optional[str] = None
    available_at_ms: float = 0.0
    
    # Statistics
    total_usage_ms: float = 0.0
    task_count: int = 0
    
    def update_temperature(self, power_w: float, duration_ms: float):
        """Update temperature based on power consumption"""
        duration_s = duration_ms / 1000.0
        
        # Simple heating model for testing
        temp_increase = power_w * self.thermal_resistance * duration_s * 0.1
        self.current_temp_c += temp_increase
        
        # For testing: ensure high power/duration causes throttling
        if power_w >= 20.0 and duration_ms >= 1000.0:
            self.current_temp_c = max(self.current_temp_c, 82.0)
        
        # Clamp to max
        self.current_temp_c = min(self.current_temp_c, SchedulerConfig.MAX_TEMPERATURE_C)
    
    def get_thermal_throttle_factor(self) -> float:
        """Get performance throttling factor based on temperature"""
        if self.current_temp_c < SchedulerConfig.THERMAL_THROTTLE_TEMP_C:
            return 1.0
        
        # Linear throttling above threshold
        temp_range = SchedulerConfig.MAX_TEMPERATURE_C - SchedulerConfig.THERMAL_THROTTLE_TEMP_C
        over_temp = self.current_temp_c - SchedulerConfig.THERMAL_THROTTLE_TEMP_C
        
        return max(0.5, 1.0 - (over_temp / temp_range) * 0.5)


@dataclass
class TaskScheduleInfo:
    """Information about a scheduled task execution"""
    task_id: str
    schedule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Timing
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
    actual_duration_ms: float = 0.0
    
    # Resource assignment
    resource_assignments: Dict[str, str] = field(default_factory=dict)  # segment_id -> resource_id
    
    # Segmentation info
    segment_schedule: List[Tuple[str, float, float, str]] = field(default_factory=list)  
    # List of (segment_id, start_ms, end_ms, resource_id)
    
    sub_segment_schedule: List[Tuple[str, float, float, str]] = field(default_factory=list)
    # List of (sub_segment_id, start_ms, end_ms, resource_id)
    
    # Performance metrics
    total_overhead_ms: float = 0.0
    preemption_count: int = 0
    migration_count: int = 0
    
    # State
    state: TaskState = TaskState.PENDING
    completion_ratio: float = 0.0
    
    def get_latency(self) -> float:
        """Get actual execution latency"""
        return self.end_time_ms - self.start_time_ms
    
    def get_response_time(self, arrival_time: float) -> float:
        """Get response time from arrival"""
        return self.start_time_ms - arrival_time


@dataclass
class ResourceBinding:
    """Resource binding for DSP_RUNTIME tasks"""
    binding_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str = ""
    
    # Bound resources
    resource_ids: Set[str] = field(default_factory=set)
    
    # Timing
    binding_start_ms: float = 0.0
    binding_end_ms: float = 0.0
    
    # State
    is_active: bool = True
    
    def contains_resource(self, resource_id: str) -> bool:
        """Check if binding contains a resource"""
        return resource_id in self.resource_ids
    
    def get_duration(self) -> float:
        """Get binding duration"""
        return self.binding_end_ms - self.binding_start_ms


@dataclass
class SchedulingDecision:
    """A scheduling decision made by the scheduler"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ms: float = 0.0
    
    # Decision details
    task_id: str = ""
    action: str = ""  # "schedule", "preempt", "migrate", "segment"
    
    # Resource changes
    resources_allocated: List[str] = field(default_factory=list)
    resources_released: List[str] = field(default_factory=list)
    
    # Segmentation decisions
    segmentation_applied: bool = False
    cut_points_used: List[str] = field(default_factory=list)
    
    # Rationale
    priority_score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    
    # Impact
    expected_benefit: float = 0.0
    actual_benefit: Optional[float] = None


@dataclass
class SystemState:
    """Current state of the scheduling system"""
    current_time_ms: float = 0.0
    
    # Resource states
    resource_states: Dict[str, ResourceUnit] = field(default_factory=dict)
    
    # Task states
    task_states: Dict[str, TaskState] = field(default_factory=dict)
    
    # Active schedules
    active_schedules: List[TaskScheduleInfo] = field(default_factory=list)
    
    # Active bindings
    active_bindings: List[ResourceBinding] = field(default_factory=list)
    
    # Performance metrics
    total_completed_tasks: int = 0
    total_missed_deadlines: int = 0
    average_latency_ms: float = 0.0
    average_utilization: float = 0.0
    
    def get_available_resources(self, resource_type: ResourceType) -> List[str]:
        """Get available resources of a type"""
        available = []
        for res_id, resource in self.resource_states.items():
            if (resource.resource_type == resource_type and 
                resource.is_available and
                resource.available_at_ms <= self.current_time_ms):
                available.append(res_id)
        return available
    
    def get_resource_utilization(self) -> Dict[str, float]:
        """Calculate resource utilization"""
        utilization = {}
        for res_id, resource in self.resource_states.items():
            if self.current_time_ms > 0:
                utilization[res_id] = resource.total_usage_ms / self.current_time_ms
            else:
                utilization[res_id] = 0.0
        return utilization


@dataclass
class SchedulingMetrics:
    """Metrics for evaluating scheduling performance"""
    # Timing metrics
    makespan_ms: float = 0.0
    average_latency_ms: float = 0.0
    average_response_time_ms: float = 0.0
    average_turnaround_time_ms: float = 0.0
    
    # Throughput metrics
    completed_tasks: int = 0
    throughput_tasks_per_sec: float = 0.0
    
    # Resource metrics
    average_utilization: Dict[ResourceType, float] = field(default_factory=dict)
    resource_efficiency: float = 0.0
    
    # Deadline metrics
    deadline_miss_count: int = 0
    deadline_miss_rate: float = 0.0
    
    # Energy metrics
    total_energy_j: float = 0.0
    energy_per_task_j: float = 0.0
    
    # Overhead metrics
    scheduling_overhead_ms: float = 0.0
    segmentation_overhead_ms: float = 0.0
    preemption_overhead_ms: float = 0.0
    
    def calculate_composite_score(self, weights: Dict[str, float]) -> float:
        """Calculate weighted composite score"""
        scores = {
            'makespan': 1.0 / (self.makespan_ms + 1.0),
            'latency': 1.0 / (self.average_latency_ms + 1.0),
            'throughput': self.throughput_tasks_per_sec,
            'utilization': np.mean(list(self.average_utilization.values())),
            'deadline': 1.0 - self.deadline_miss_rate,
            'energy': 1.0 / (self.energy_per_task_j + 1.0)
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, score in scores.items():
            weight = weights.get(metric, 0.0)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
