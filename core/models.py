from typing import Dict, Optional, Set, List, Tuple
from dataclasses import dataclass, field
from core.enums import ResourceType, RuntimeType, CutPointStatus

@dataclass
class CutPoint:
    """Network cut point definition with duration info"""
    op_id: str  # Operation ID (e.g., "op1", "op10", "op23")
    # Duration table for the segment BEFORE this cut point
    before_duration_table: Dict[float, float]  # {BW: duration} for segment before cut
    overhead_ms: float = 0.15  # Overhead introduced by cutting at this point
    status: CutPointStatus = CutPointStatus.AUTO

@dataclass
class SubSegment:
    """Sub-segment created by cutting at cut points"""
    sub_id: str  # Sub-segment identifier
    resource_type: ResourceType
    duration_table: Dict[float, float]  # {BW: duration} lookup table, same format as ResourceSegment
    cut_overhead: float = 0.0  # Overhead from cutting operations
    original_segment_id: str = ""  # Reference to original segment
    
    def get_duration(self, bw: float) -> float:
        """Get execution time based on bandwidth, including cut overhead"""
        base_duration = self.duration_table.get(bw, 0.0)
        if bw not in self.duration_table and self.duration_table:
            # If exact bandwidth not found, use closest
            closest_bw = min(self.duration_table.keys(), key=lambda x: abs(x - bw))
            base_duration = self.duration_table[closest_bw]
        return base_duration + self.cut_overhead

@dataclass
class ResourceSegment:
    """Enhanced resource usage segment with cutting support"""
    resource_type: ResourceType
    duration_table: Dict[float, float]  # {BW: duration} lookup table
    start_time: float  # Start time relative to task beginning
    segment_id: str = ""  # Unique identifier for this segment
    
    # Network cutting related attributes
    cut_points: List[CutPoint] = field(default_factory=list)
    sub_segments: List[SubSegment] = field(default_factory=list)
    is_segmented: bool = False
    segmentation_overhead: float = 0.0  # Total overhead from all cuts
    
    def get_duration(self, bw: float) -> float:
        """Get execution time based on bandwidth"""
        if bw in self.duration_table:
            return self.duration_table[bw]
        # If exact bandwidth not found, use closest
        closest_bw = min(self.duration_table.keys(), key=lambda x: abs(x - bw))
        return self.duration_table[closest_bw]
    
    def add_cut_point(self, op_id: str, before_duration_table: Dict[float, float], 
                      overhead_ms: float = 0.15):
        """Add a cut point to this segment with duration info
        
        Args:
            op_id: Operation ID where to cut
            before_duration_table: Duration table for the segment BEFORE this cut point
            overhead_ms: Overhead introduced by cutting at this point
        """
        cut_point = CutPoint(
            op_id=op_id, 
            before_duration_table=before_duration_table,
            overhead_ms=overhead_ms
        )
        self.cut_points.append(cut_point)
    
    def apply_segmentation(self, enabled_cuts: List[str]) -> List[SubSegment]:
        """Apply segmentation using specified cut points"""
        if not enabled_cuts or not self.cut_points:
            # No segmentation - return single sub-segment
            sub_seg = SubSegment(
                sub_id=f"{self.segment_id}_0",
                resource_type=self.resource_type,
                duration_table=self.duration_table.copy(),
                original_segment_id=self.segment_id
            )
            self.sub_segments = [sub_seg]
            self.is_segmented = False
            return self.sub_segments
        
        # Filter and sort enabled cut points
        enabled_cut_points = [cp for cp in self.cut_points if cp.op_id in enabled_cuts]
        
        if not enabled_cut_points:
            return self.apply_segmentation([])  # No valid cuts
        
        # Create sub-segments based on cut points
        self.sub_segments = []
        total_overhead = 0.0
        
        # Track cumulative duration to calculate remaining segment
        cumulative_duration = {bw: 0.0 for bw in self.duration_table.keys()}
        
        # Create sub-segments for each cut point
        for i, cut_point in enumerate(enabled_cut_points):
            # The duration table in cut_point defines this sub-segment's duration
            sub_seg = SubSegment(
                sub_id=f"{self.segment_id}_{i}",
                resource_type=self.resource_type,
                duration_table=cut_point.before_duration_table.copy(),
                cut_overhead=cut_point.overhead_ms,
                original_segment_id=self.segment_id
            )
            
            self.sub_segments.append(sub_seg)
            total_overhead += cut_point.overhead_ms
            
            # Update cumulative duration
            for bw in cumulative_duration:
                if bw in cut_point.before_duration_table:
                    cumulative_duration[bw] += cut_point.before_duration_table[bw]
        
        # Create final sub-segment with remaining duration
        final_duration_table = {}
        for bw, total_duration in self.duration_table.items():
            remaining_duration = total_duration - cumulative_duration.get(bw, 0.0)
            if remaining_duration > 0:
                final_duration_table[bw] = remaining_duration
        
        if final_duration_table:
            final_sub_seg = SubSegment(
                sub_id=f"{self.segment_id}_{len(enabled_cut_points)}",
                resource_type=self.resource_type,
                duration_table=final_duration_table,
                cut_overhead=0.0,  # No overhead for final segment
                original_segment_id=self.segment_id
            )
            self.sub_segments.append(final_sub_seg)
        
        self.is_segmented = True
        self.segmentation_overhead = total_overhead
        return self.sub_segments
    
    def get_total_duration_with_cuts(self, bw: float, enabled_cuts: List[str]) -> float:
        """Get total duration including segmentation overhead"""
        base_duration = self.get_duration(bw)
        if not enabled_cuts:
            return base_duration
        
        # Calculate overhead from enabled cuts
        total_overhead = sum(cp.overhead_ms for cp in self.cut_points 
                           if cp.op_id in enabled_cuts)
        return base_duration + total_overhead
    
    def get_available_cuts(self) -> List[str]:
        """Get list of available cut point IDs"""
        return [cp.op_id for cp in self.cut_points]

@dataclass
class ResourceUnit:
    """Resource unit (NPU or DSP instance)"""
    unit_id: str
    resource_type: ResourceType
    bandwidth: float  # Bandwidth capability of this unit
    
    def __hash__(self):
        return hash(self.unit_id)

@dataclass
class TaskScheduleInfo:
    """Enhanced task scheduling information with segmentation details"""
    task_id: str
    start_time: float
    end_time: float
    assigned_resources: Dict[ResourceType, str]  # {resource type: assigned resource ID}
    actual_latency: float  # Actual latency from scheduling to completion
    runtime_type: RuntimeType  # Runtime configuration used
    
    # Segmentation related information
    used_cuts: Dict[str, List[str]] = field(default_factory=dict)  # {segment_id: [cut_point_ids]}
    segmentation_overhead: float = 0.0  # Total overhead from all segmentations
    sub_segment_schedule: List[Tuple[str, float, float]] = field(default_factory=list)  # [(sub_seg_id, start, end)]

@dataclass
class ResourceBinding:
    """Resource binding information for DSP_Runtime tasks"""
    task_id: str
    bound_resources: Set[str]  # Set of resource IDs that are bound together
    binding_start: float
    binding_end: float

@dataclass
class SegmentationDecision:
    """Decision record for segment cutting"""
    segment_id: str
    task_id: str
    available_cuts: List[str]
    selected_cuts: List[str]
    decision_reason: str  # Why this segmentation was chosen
    estimated_benefit: float  # Expected improvement in scheduling efficiency
    actual_overhead: float  # Actual overhead introduced