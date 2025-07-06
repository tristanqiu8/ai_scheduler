#!/usr/bin/env python3
"""
Enhanced neural network task class with custom segmentation as default
"""

from typing import List, Dict, Set, Optional, Tuple
from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.models import ResourceSegment, TaskScheduleInfo, SubSegment, SegmentationDecision

class NNTask:
    """Enhanced neural network task class with custom segmentation as default"""
    
    def __init__(self, task_id: str, name: str = "", priority: TaskPriority = TaskPriority.NORMAL, 
                 runtime_type: RuntimeType = RuntimeType.ACPU_RUNTIME,
                 segmentation_strategy: SegmentationStrategy = SegmentationStrategy.CUSTOM_SEGMENTATION):
        self.task_id = task_id
        self.name = name or f"Task_{task_id}"
        self.priority = priority
        self.runtime_type = runtime_type
        self.segmentation_strategy = segmentation_strategy
        
        self.segments: List[ResourceSegment] = []
        self.dependencies: Set[str] = set()
        self.fps_requirement: float = 30.0
        self.latency_requirement: float = 100.0
        
        # Scheduling related info
        self.schedule_info: Optional[TaskScheduleInfo] = None
        self.last_execution_time: float = -float('inf')
        self.ready_time: float = 0
        
        # Network segmentation related
        self.segmentation_decisions: List[SegmentationDecision] = []
        self.current_segmentation: Dict[str, List[str]] = {}
        self.total_segmentation_overhead: float = 0.0
        
        # Preset cut points for CUSTOM_SEGMENTATION
        self.preset_cut_configurations: Dict[str, List[List[str]]] = {}
        self.selected_cut_config_index: Dict[str, int] = {}
    
    def set_preset_cut_configurations(self, segment_id: str, configurations: List[List[str]]):
        """Set preset cut point configurations for a segment
        Args:
            segment_id: ID of the segment
            configurations: List of cut point configurations, each is a list of cut point IDs
        Example:
            task.set_preset_cut_configurations("npu_seg", [
                [],                    # Config 0: No cuts
                ["op1"],              # Config 1: Cut at op1 only
                ["op1", "op10"],      # Config 2: Cut at op1 and op10
                ["op1", "op10", "op23"]  # Config 3: Cut at all points
            ])
        """
        self.preset_cut_configurations[segment_id] = configurations
        # Default to no cuts (config 0)
        self.selected_cut_config_index[segment_id] = 0
    
    def select_cut_configuration(self, segment_id: str, config_index: int):
        """Select a specific cut configuration for a segment
        Args:
            segment_id: ID of the segment
            config_index: Index of the configuration to use
        """
        if segment_id in self.preset_cut_configurations:
            max_index = len(self.preset_cut_configurations[segment_id]) - 1
            if 0 <= config_index <= max_index:
                self.selected_cut_config_index[segment_id] = config_index
                # Update current segmentation
                self.current_segmentation[segment_id] = self.preset_cut_configurations[segment_id][config_index]
            else:
                raise ValueError(f"Config index {config_index} out of range [0, {max_index}]")
        else:
            raise ValueError(f"No preset configurations for segment {segment_id}")
    
    def get_optimal_segmentation(self, available_resources: Dict[ResourceType, List[float]], 
                               current_time: float = 0.0) -> Dict[str, List[str]]:
        """Determine optimal segmentation based on strategy and task properties"""
        if self.segmentation_strategy == SegmentationStrategy.NO_SEGMENTATION:
            return {seg.segment_id: [] for seg in self.segments}
        
        elif self.segmentation_strategy == SegmentationStrategy.FORCED_SEGMENTATION:
            return {seg.segment_id: seg.get_available_cuts() for seg in self.segments}
        
        elif self.segmentation_strategy == SegmentationStrategy.CUSTOM_SEGMENTATION:
            # Use preset configurations based on selected index
            result = {}
            for segment in self.segments:
                if segment.segment_id in self.preset_cut_configurations:
                    config_idx = self.selected_cut_config_index.get(segment.segment_id, 0)
                    result[segment.segment_id] = self.preset_cut_configurations[segment.segment_id][config_idx]
                else:
                    # If no preset config, use current segmentation or empty
                    result[segment.segment_id] = self.current_segmentation.get(segment.segment_id, [])
            return result
        
        else:  # ADAPTIVE_SEGMENTATION
            # Modified adaptive logic to consider priority and runtime_type more heavily
            return self._calculate_priority_aware_segmentation(available_resources, current_time)
    
    def _calculate_priority_aware_segmentation(self, available_resources: Dict[ResourceType, List[float]], 
                                              current_time: float) -> Dict[str, List[str]]:
        """Calculate segmentation with strong consideration for priority and runtime_type"""
        decisions = {}
        
        for segment in self.segments:
            segment_decisions = []
            available_cuts = segment.get_available_cuts()
            
            if not available_cuts:
                decisions[segment.segment_id] = []
                continue
            
            # Get available bandwidths for this resource type
            available_bw = available_resources.get(segment.resource_type, [])
            if not available_bw:
                decisions[segment.segment_id] = []
                continue
            
            # Priority-based cut selection
            max_cuts = len(available_cuts)
            
            # Determine number of cuts based on priority and runtime type
            if self.priority == TaskPriority.CRITICAL:
                # Critical tasks: aggressive segmentation for low latency
                if self.runtime_type == RuntimeType.DSP_RUNTIME:
                    # DSP_Runtime critical tasks need careful segmentation due to binding
                    num_cuts = min(2, max_cuts)  # Limited cuts to reduce binding complexity
                else:
                    # ACPU_Runtime critical tasks can use more cuts
                    num_cuts = min(max_cuts, 3)  # Up to 3 cuts for parallelism
            
            elif self.priority == TaskPriority.HIGH:
                # High priority: moderate segmentation
                if self.runtime_type == RuntimeType.DSP_RUNTIME:
                    num_cuts = min(1, max_cuts)  # Single cut for DSP_Runtime
                else:
                    num_cuts = min(2, max_cuts)  # Up to 2 cuts for ACPU_Runtime
            
            elif self.priority == TaskPriority.NORMAL:
                # Normal priority: conservative segmentation
                if len(available_bw) > 2:  # Only segment if multiple resources available
                    num_cuts = min(1, max_cuts)
                else:
                    num_cuts = 0
            
            else:  # LOW priority
                # Low priority: minimal segmentation
                num_cuts = 0 if self.runtime_type == RuntimeType.DSP_RUNTIME else min(1, max_cuts)
            
            # Consider overhead limits
            max_acceptable_overhead = self.latency_requirement * 0.15  # 15% of latency requirement
            
            # Select cuts that don't exceed overhead threshold
            selected_cuts = []
            total_overhead = 0.0
            
            for i, cut_point in enumerate(segment.cut_points):
                if len(selected_cuts) >= num_cuts:
                    break
                if total_overhead + cut_point.overhead_ms <= max_acceptable_overhead:
                    selected_cuts.append(cut_point.op_id)
                    total_overhead += cut_point.overhead_ms
            
            decisions[segment.segment_id] = selected_cuts
        
        return decisions
    
    def get_priority_score(self) -> float:
        """Get numerical priority score for scheduling decisions"""
        # Lower value = higher priority
        base_score = self.priority.value
        
        # Adjust score based on runtime type
        if self.runtime_type == RuntimeType.DSP_RUNTIME:
            # DSP_Runtime tasks get slight priority boost due to resource binding needs
            base_score -= 0.5
        
        return max(0, base_score)
    
    def get_segmentation_complexity(self) -> float:
        """Calculate segmentation complexity score"""
        if not self.is_segmented:
            return 0.0
        
        complexity = 0.0
        for segment in self.segments:
            if segment.is_segmented:
                # More sub-segments = higher complexity
                complexity += len(segment.sub_segments) * 0.5
                # Add overhead penalty
                complexity += segment.segmentation_overhead * 0.1
        
        # Runtime type affects complexity
        if self.runtime_type == RuntimeType.DSP_RUNTIME:
            complexity *= 1.5  # DSP_Runtime binding adds complexity
        
        return complexity
    
    def set_npu_only(self, duration_table: Dict[float, float], segment_id: str = "npu_seg"):
        """Set as NPU-only task with optional cutting support"""
        segment = ResourceSegment(ResourceType.NPU, duration_table, 0, segment_id)
        self.segments = [segment]
        
    def set_dsp_npu_sequence(self, segments: List[Tuple[ResourceType, Dict[float, float], float, str]]):
        """Set as DSP+NPU mixed execution task with segment IDs"""
        self.segments = []
        for i, (resource_type, duration_table, start_time, segment_id) in enumerate(segments):
            if not segment_id:
                segment_id = f"{resource_type.value.lower()}_seg_{i}"
            segment = ResourceSegment(resource_type, duration_table, start_time, segment_id)
            self.segments.append(segment)
    
    def add_cut_points_to_segment(self, segment_id: str, 
                                 cut_points: List[Tuple[str, Dict[float, float], float]]):
        """Add cut points to a specific segment
        Args:
            segment_id: ID of the segment to add cut points to
            cut_points: List of (op_id, before_duration_table, overhead_ms)
        """
        segment = self.get_segment_by_id(segment_id)
        if segment:
            for op_id, before_duration_table, overhead_ms in cut_points:
                segment.add_cut_point(op_id, before_duration_table, overhead_ms)
        else:
            raise ValueError(f"Segment {segment_id} not found in task {self.task_id}")
    
    def get_segment_by_id(self, segment_id: str) -> Optional[ResourceSegment]:
        """Get segment by its ID"""
        return next((seg for seg in self.segments if seg.segment_id == segment_id), None)
    
    def get_all_available_cuts(self) -> Dict[str, List[str]]:
        """Get all available cut points for all segments"""
        available_cuts = {}
        for segment in self.segments:
            available_cuts[segment.segment_id] = segment.get_available_cuts()
        return available_cuts
    
    def apply_segmentation_decision(self, decisions: Dict[str, List[str]]) -> float:
        """Apply segmentation decisions and return total overhead"""
        total_overhead = 0.0
        self.current_segmentation = decisions.copy()
        
        for segment in self.segments:
            enabled_cuts = decisions.get(segment.segment_id, [])
            segment.apply_segmentation(enabled_cuts)
            total_overhead += segment.segmentation_overhead
        
        self.total_segmentation_overhead = total_overhead
        return total_overhead
    
    def get_total_duration_with_segmentation(self, resource_bw_map: Dict[ResourceType, float], 
                                           segmentation_decisions: Optional[Dict[str, List[str]]] = None) -> float:
        """Get total execution time including segmentation overhead"""
        if segmentation_decisions is None:
            segmentation_decisions = self.current_segmentation
        
        max_end_time = 0.0
        current_time = 0.0
        
        for segment in self.segments:
            # 处理段的开始时间（保持向后兼容）
            if segment.start_time > current_time:
                current_time = segment.start_time
            
            enabled_cuts = segmentation_decisions.get(segment.segment_id, [])
            bw = resource_bw_map.get(segment.resource_type, 1.0)
            
            if segment.is_segmented and segment.sub_segments:
                # 分段的情况
                for sub_seg in segment.sub_segments:
                    duration = sub_seg.get_duration(bw)
                    current_time += duration
            else:
                # 未分段的情况
                duration = segment.get_total_duration_with_cuts(bw, enabled_cuts)
                current_time += duration
            
            max_end_time = max(max_end_time, current_time)
        
        return max_end_time
    
    def get_sub_segments_for_scheduling(self) -> List[SubSegment]:
        """Get all sub-segments created by current segmentation for scheduling"""
        all_sub_segments = []
        for segment in self.segments:
            if segment.is_segmented:
                all_sub_segments.extend(segment.sub_segments)
            else:
                # Create a single sub-segment for non-segmented segments
                sub_seg = SubSegment(
                    sub_id=f"{segment.segment_id}_0",
                    resource_type=segment.resource_type,
                    duration_table=segment.duration_table.copy(),
                    original_segment_id=segment.segment_id
                )
                all_sub_segments.append(sub_seg)
        
        return all_sub_segments
    
    def add_dependency(self, task_id: str):
        self.dependencies.add(task_id)
    
    def add_dependencies(self, task_ids: List[str]):
        self.dependencies.update(task_ids)
    
    def set_performance_requirements(self, fps: float, latency: float):
        self.fps_requirement = fps
        self.latency_requirement = latency
    
    def set_runtime_type(self, runtime_type: RuntimeType):
        """Set runtime configuration type"""
        self.runtime_type = runtime_type
    
    def set_segmentation_strategy(self, strategy: SegmentationStrategy):
        """Set network segmentation strategy"""
        self.segmentation_strategy = strategy
    
    def get_total_duration(self, resource_bw_map: Dict[ResourceType, float]) -> float:
        """Get total execution time based on assigned resource bandwidths (legacy method)"""
        return self.get_total_duration_with_segmentation(resource_bw_map)
    
    def requires_resource_binding(self) -> bool:
        """Check if task requires resource binding (DSP_Runtime with multiple resource types)"""
        return (self.runtime_type == RuntimeType.DSP_RUNTIME and 
                len(set(seg.resource_type for seg in self.segments)) > 1)
    
    def get_segmentation_summary(self) -> str:
        """Get a summary of current segmentation decisions"""
        if not self.current_segmentation:
            return "No segmentation applied"
        
        summary_parts = []
        for segment_id, cuts in self.current_segmentation.items():
            if cuts:
                summary_parts.append(f"{segment_id}: {len(cuts)} cuts ({', '.join(cuts)})")
            else:
                summary_parts.append(f"{segment_id}: no cuts")
        
        overhead_info = f"Total overhead: {self.total_segmentation_overhead:.2f}ms"
        return f"{'; '.join(summary_parts)}. {overhead_info}"
    
    @property
    def min_interval_ms(self) -> float:
        """Minimum scheduling interval based on FPS requirement"""
        return 1000.0 / self.fps_requirement if self.fps_requirement > 0 else float('inf')
    
    @property
    def uses_dsp(self) -> bool:
        return any(seg.resource_type == ResourceType.DSP for seg in self.segments)
    
    @property
    def uses_npu(self) -> bool:
        return any(seg.resource_type == ResourceType.NPU for seg in self.segments)
    
    @property
    def is_segmented(self) -> bool:
        """Check if any segment is currently segmented"""
        return any(seg.is_segmented for seg in self.segments)
    
    def __repr__(self):
        sched_str = f", scheduled@{self.schedule_info.start_time:.1f}ms" if self.schedule_info else ""
        seg_str = f", segmented" if self.is_segmented else ""
        return (f"Task{self.task_id}({self.name}, {self.runtime_type.value}, "
                f"priority={self.priority.name}, fps={self.fps_requirement}"
                f"{sched_str}{seg_str})")
