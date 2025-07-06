from typing import List, Dict, Set, Optional, Dict, Set, Tuple
from collections import defaultdict
from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.models import ResourceUnit, TaskScheduleInfo, ResourceBinding, SegmentationDecision, SubSegment
from core.task import NNTask
from core.priority_queue import ResourcePriorityQueues
import numpy as np

class MultiResourceScheduler:
    """Enhanced multi-resource scheduler with network segmentation support"""
    
    def __init__(self, enable_segmentation: bool = True, max_segmentation_overhead_ratio: float = 0.15):
        self.tasks: Dict[str, NNTask] = {}
        self.resources: Dict[ResourceType, List[ResourceUnit]] = {
            ResourceType.NPU: [],
            ResourceType.DSP: []
        }
        # Priority queues for each resource
        self.resource_queues: Dict[str, ResourcePriorityQueues] = {}
        self.schedule_history: List[TaskScheduleInfo] = []
        # Track resource bindings for DSP_Runtime tasks
        self.active_bindings: List[ResourceBinding] = []
        
        # Network segmentation configuration
        self.enable_segmentation = enable_segmentation
        self.max_segmentation_overhead_ratio = max_segmentation_overhead_ratio  # Max overhead as ratio of task latency
        self.segmentation_decisions_history: List[SegmentationDecision] = []
        
        # Statistics
        self.segmentation_stats = {
            'total_decisions': 0,
            'segmented_tasks': 0,
            'total_overhead': 0.0,
            'average_benefit': 0.0
        }
        
    def add_npu(self, npu_id: str, bandwidth: float):
        """Add NPU resource"""
        npu = ResourceUnit(npu_id, ResourceType.NPU, bandwidth)
        self.resources[ResourceType.NPU].append(npu)
        self.resource_queues[npu_id] = ResourcePriorityQueues(npu_id)
        
    def add_dsp(self, dsp_id: str, bandwidth: float):
        """Add DSP resource"""
        dsp = ResourceUnit(dsp_id, ResourceType.DSP, bandwidth)
        self.resources[ResourceType.DSP].append(dsp)
        self.resource_queues[dsp_id] = ResourcePriorityQueues(dsp_id)
    
    def add_task(self, task: NNTask):
        """Add task"""
        self.tasks[task.task_id] = task
    
    def get_available_resources_info(self, current_time: float) -> Dict[ResourceType, List[float]]:
        """Get information about available resources and their bandwidths"""
        available_resources = {ResourceType.NPU: [], ResourceType.DSP: []}
        
        for resource_type, resources in self.resources.items():
            for resource in resources:
                queue = self.resource_queues[resource.unit_id]
                if queue.available_time <= current_time and not queue.is_bound_to_other_task("", current_time):
                    available_resources[resource_type].append(resource.bandwidth)
        
        return available_resources
    
    def make_segmentation_decision(self, task: NNTask, current_time: float) -> Dict[str, List[str]]:
        """Make segmentation decision for a task based on current state"""
        if not self.enable_segmentation:
            return {seg.segment_id: [] for seg in task.segments}
        
        # Get available resources
        available_resources = self.get_available_resources_info(current_time)
        
        # Calculate optimal segmentation based on task strategy
        optimal_segmentation = task.get_optimal_segmentation(available_resources, current_time)
        
        # Validate segmentation doesn't exceed overhead limits
        validated_segmentation = self._validate_segmentation_overhead(task, optimal_segmentation)
        
        # Record decision
        for segment_id, cuts in validated_segmentation.items():
            if cuts or segment_id in optimal_segmentation:
                decision = SegmentationDecision(
                    segment_id=segment_id,
                    task_id=task.task_id,
                    available_cuts=task.get_segment_by_id(segment_id).get_available_cuts() if task.get_segment_by_id(segment_id) else [],
                    selected_cuts=cuts,
                    decision_reason=f"Adaptive decision based on {task.segmentation_strategy.value}",
                    estimated_benefit=self._estimate_segmentation_benefit(task, segment_id, cuts),
                    actual_overhead=sum(cp.overhead_ms for cp in task.get_segment_by_id(segment_id).cut_points if cp.op_id in cuts) if task.get_segment_by_id(segment_id) else 0.0
                )
                self.segmentation_decisions_history.append(decision)
        
        return validated_segmentation
    
    def _validate_segmentation_overhead(self, task: NNTask, segmentation: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Validate that segmentation overhead doesn't exceed acceptable limits"""
        max_overhead = task.latency_requirement * self.max_segmentation_overhead_ratio
        validated_segmentation = {}
        
        for segment_id, cuts in segmentation.items():
            segment = task.get_segment_by_id(segment_id)
            if not segment:
                validated_segmentation[segment_id] = []
                continue
            
            # Calculate total overhead for these cuts
            total_overhead = sum(cp.overhead_ms for cp in segment.cut_points if cp.op_id in cuts)
            
            if total_overhead <= max_overhead:
                validated_segmentation[segment_id] = cuts
            else:
                # Reduce cuts to fit within overhead limit
                validated_cuts = []
                current_overhead = 0.0
                for cut_id in cuts:
                    cut_point = next((cp for cp in segment.cut_points if cp.op_id == cut_id), None)
                    if cut_point and current_overhead + cut_point.overhead_ms <= max_overhead:
                        validated_cuts.append(cut_id)
                        current_overhead += cut_point.overhead_ms
                    else:
                        break
                validated_segmentation[segment_id] = validated_cuts
        
        return validated_segmentation
    
    def _estimate_segmentation_benefit(self, task: NNTask, segment_id: str, cuts: List[str]) -> float:
        """Estimate the benefit of segmentation for scheduling efficiency"""
        if not cuts:
            return 0.0
        
        segment = task.get_segment_by_id(segment_id)
        if not segment:
            return 0.0
        
        # Simple heuristic: benefit is proportional to number of cuts and resource availability
        available_resources = len(self.resources[segment.resource_type])
        num_cuts = len(cuts)
        
        # More cuts = more parallelism opportunities, but diminishing returns
        base_benefit = num_cuts * 0.1 * available_resources
        
        # Adjust based on task priority (higher priority gets more benefit estimation)
        priority_multiplier = 1.0 + (3 - task.priority.value) * 0.2
        
        return base_benefit * priority_multiplier
    
    def cleanup_expired_bindings(self, current_time: float):
        """Clean up expired resource bindings"""
        # Remove expired bindings
        self.active_bindings = [binding for binding in self.active_bindings 
                               if binding.binding_end > current_time]
        
        # Release resource bindings
        for queue in self.resource_queues.values():
            if queue.bound_until <= current_time:
                queue.release_binding()
    
    def find_available_resources_for_task_with_segmentation(self, task: NNTask, current_time: float) -> Optional[Dict[ResourceType, str]]:
        """Find available resources for a task with segmentation consideration"""
        # First, make segmentation decision
        segmentation_decision = self.make_segmentation_decision(task, current_time)
        
        # Apply segmentation to task
        task.apply_segmentation_decision(segmentation_decision)
        
        # Now find resources based on runtime type
        if task.runtime_type == RuntimeType.DSP_RUNTIME and task.requires_resource_binding():
            return self.find_bound_resources_with_segmentation(task, current_time)
        else:
            return self.find_pipelined_resources_with_segmentation(task, current_time)
    
    def find_bound_resources_with_segmentation(self, task: NNTask, current_time: float) -> Optional[Dict[ResourceType, str]]:
        """Find resources that can be bound together for DSP_Runtime tasks with segmentation"""
        sub_segments = task.get_sub_segments_for_scheduling()
        required_resource_types = set(sub_seg.resource_type for sub_seg in sub_segments)
        
        # Check all combinations of resources to find a set that can be bound together
        for resource_combo in self.get_resource_combinations(required_resource_types):
            can_bind_all = True
            binding_start = current_time
            binding_end = current_time
            
            # Check if all resources in this combination are available
            for resource_type, resource_id in resource_combo.items():
                queue = self.resource_queues[resource_id]
                
                # Check if resource is available and not bound to another task
                if (queue.is_bound_to_other_task(task.task_id, current_time) or
                    queue.has_higher_priority_tasks(task.priority, current_time, task.task_id)):
                    can_bind_all = False
                    break
                
                # Calculate binding times
                binding_start = max(binding_start, queue.available_time)
            
            if can_bind_all:
                # Calculate total binding duration with segmentation
                for sub_seg in sub_segments:
                    if sub_seg.resource_type in resource_combo:
                        resource_id = resource_combo[sub_seg.resource_type]
                        resource = next(r for r in self.resources[sub_seg.resource_type] 
                                      if r.unit_id == resource_id)
                        seg_end = binding_start + sub_seg.start_time + sub_seg.get_duration(resource.bandwidth)
                        binding_end = max(binding_end, seg_end)
                
                # Bind all resources
                bound_resource_ids = set(resource_combo.values())
                for resource_id in bound_resource_ids:
                    self.resource_queues[resource_id].bind_resource(task.task_id, binding_end)
                
                # Record binding
                binding = ResourceBinding(
                    task_id=task.task_id,
                    bound_resources=bound_resource_ids,
                    binding_start=binding_start,
                    binding_end=binding_end
                )
                self.active_bindings.append(binding)
                
                return resource_combo
        
        return None
    
    def find_pipelined_resources_with_segmentation(self, task: NNTask, current_time: float) -> Optional[Dict[ResourceType, str]]:
        """Find resources for ACPU_Runtime tasks with segmentation support"""
        sub_segments = task.get_sub_segments_for_scheduling()
        assigned_resources = {}
        earliest_start = current_time
        
        # Check resource availability for all sub-segments
        for sub_seg in sub_segments:
            best_resource = None
            best_start_time = float('inf')
            
            # Find best resource for this sub-segment
            for resource in self.resources[sub_seg.resource_type]:
                queue = self.resource_queues[resource.unit_id]
                
                # Check if higher priority tasks are waiting or resource is bound
                if (queue.has_higher_priority_tasks(task.priority, current_time, task.task_id) or
                    queue.is_bound_to_other_task(task.task_id, current_time)):
                    continue
                
                # Calculate when this resource could start this sub-segment
                resource_start = max(queue.available_time, earliest_start + sub_seg.start_time)
                
                if resource_start < best_start_time:
                    best_start_time = resource_start
                    best_resource = resource
            
            if best_resource:
                assigned_resources[sub_seg.resource_type] = best_resource.unit_id
                earliest_start = max(earliest_start, best_start_time - sub_seg.start_time)
            else:
                return None
        
        return assigned_resources
    
    def get_resource_combinations(self, required_types: Set[ResourceType]) -> List[Dict[ResourceType, str]]:
        """Get all possible combinations of resources for the required types"""
        combinations = []
        
        def generate_combinations(types_list, current_combo, index):
            if index == len(types_list):
                combinations.append(current_combo.copy())
                return
            
            resource_type = types_list[index]
            for resource in self.resources[resource_type]:
                current_combo[resource_type] = resource.unit_id
                generate_combinations(types_list, current_combo, index + 1)
                del current_combo[resource_type]
        
        if required_types:
            generate_combinations(list(required_types), {}, 0)
        
        return combinations
    
    def priority_aware_schedule_with_segmentation(self, time_window: float = 1000.0) -> List[TaskScheduleInfo]:
        """Enhanced priority-aware scheduling algorithm with network segmentation support"""
        # Reset scheduling state
        for queue in self.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            for p in TaskPriority:
                queue.queues[p].clear()
        
        for task in self.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
            task.current_segmentation = {}
            task.total_segmentation_overhead = 0.0
        
        self.schedule_history.clear()
        self.active_bindings.clear()
        self.segmentation_decisions_history.clear()
        
        # Reset segmentation statistics
        self.segmentation_stats = {
            'total_decisions': 0,
            'segmented_tasks': 0,
            'total_overhead': 0.0,
            'average_benefit': 0.0
        }
        
        # Track task execution counts for FPS calculation
        task_execution_count = defaultdict(int)
        current_time = 0.0
        
        while current_time < time_window:
            # Clean up expired bindings
            self.cleanup_expired_bindings(current_time)
            
            # Phase 1: Check which tasks are ready (dependencies satisfied)
            for task in self.tasks.values():
                # Skip if task was recently executed
                if task.last_execution_time + task.min_interval_ms > current_time:
                    continue
                
                # Check dependencies
                deps_satisfied = True
                max_dep_end_time = 0.0
                
                for dep_id in task.dependencies:
                    dep_task = self.tasks.get(dep_id)
                    if dep_task:
                        if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                            deps_satisfied = False
                            break
                        if dep_task.schedule_info:
                            max_dep_end_time = max(max_dep_end_time, dep_task.schedule_info.end_time)
                
                if deps_satisfied:
                    task.ready_time = max(current_time, max_dep_end_time)
            
            # Phase 2: Schedule ready tasks with segmentation
            scheduled_any = False
            
            for task in self.tasks.values():
                if (task.ready_time <= current_time and 
                    task.last_execution_time + task.min_interval_ms <= current_time):
                    
                    # Try to find available resources with segmentation consideration
                    assigned_resources = self.find_available_resources_for_task_with_segmentation(task, current_time)
                    
                    if assigned_resources:
                        # Get sub-segments after segmentation
                        sub_segments = task.get_sub_segments_for_scheduling()
                        
                        # Execute the scheduling
                        actual_start = current_time
                        actual_end = actual_start
                        sub_segment_schedule = []
                        
                        # Update resource availability times based on sub-segments
                        for sub_seg in sub_segments:
                            resource_id = assigned_resources[sub_seg.resource_type]
                            resource = next(r for r in self.resources[sub_seg.resource_type] 
                                          if r.unit_id == resource_id)
                            
                            sub_seg_start = actual_start + sub_seg.start_time
                            sub_seg_duration = sub_seg.get_duration(resource.bandwidth)
                            sub_seg_end = sub_seg_start + sub_seg_duration
                            
                            # Record sub-segment schedule
                            sub_segment_schedule.append((sub_seg.sub_id, sub_seg_start, sub_seg_end))
                            
                            # Update resource availability (but don't override binding for DSP_Runtime)
                            if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                                self.resource_queues[resource_id].available_time = sub_seg_end
                            
                            actual_end = max(actual_end, sub_seg_end)
                        
                        # Create enhanced schedule info with segmentation details
                        schedule_info = TaskScheduleInfo(
                            task_id=task.task_id,
                            start_time=actual_start,
                            end_time=actual_end,
                            assigned_resources=assigned_resources,
                            actual_latency=actual_end - current_time,
                            runtime_type=task.runtime_type,
                            used_cuts=task.current_segmentation.copy(),
                            segmentation_overhead=task.total_segmentation_overhead,
                            sub_segment_schedule=sub_segment_schedule
                        )
                        
                        task.schedule_info = schedule_info
                        task.last_execution_time = actual_start
                        self.schedule_history.append(schedule_info)
                        task_execution_count[task.task_id] += 1
                        scheduled_any = True
                        
                        # Update statistics
                        if task.is_segmented:
                            self.segmentation_stats['segmented_tasks'] += 1
                            self.segmentation_stats['total_overhead'] += task.total_segmentation_overhead
            
            # Phase 3: Advance time
            if not scheduled_any:
                # Find next event time
                next_time = current_time + 1.0
                
                # Check when resources become available or bindings expire
                for queue in self.resource_queues.values():
                    if queue.available_time > current_time:
                        next_time = min(next_time, queue.available_time)
                    if queue.bound_until > current_time:
                        next_time = min(next_time, queue.bound_until)
                
                # Check when tasks can be scheduled again
                for task in self.tasks.values():
                    next_schedule_time = task.last_execution_time + task.min_interval_ms
                    if next_schedule_time > current_time:
                        next_time = min(next_time, next_schedule_time)
                
                current_time = min(next_time, time_window)
            else:
                # Small time advance to check for new opportunities
                current_time += 0.1
        
        # Update final statistics
        self.segmentation_stats['total_decisions'] = len(self.segmentation_decisions_history)
        if self.segmentation_decisions_history:
            self.segmentation_stats['average_benefit'] = sum(d.estimated_benefit for d in self.segmentation_decisions_history) / len(self.segmentation_decisions_history)
        
        return self.schedule_history
    
    def get_resource_utilization(self, time_window: float) -> Dict[str, float]:
        """Calculate resource utilization with segmentation consideration"""
        utilization = {}
        
        for resource_type, resources in self.resources.items():
            for resource in resources:
                busy_time = 0.0
                for schedule in self.schedule_history:
                    if resource.unit_id in schedule.assigned_resources.values():
                        # Consider sub-segment scheduling for more accurate utilization
                        if schedule.sub_segment_schedule:
                            for sub_seg_id, start, end in schedule.sub_segment_schedule:
                                # Check if this sub-segment used this resource
                                task = self.tasks[schedule.task_id]
                                for sub_seg in task.get_sub_segments_for_scheduling():
                                    if (sub_seg.sub_id == sub_seg_id and 
                                        sub_seg.resource_type == resource.resource_type):
                                        busy_time += (end - start)
                                        break
                        else:
                            # Fallback to total task duration
                            busy_time += (schedule.end_time - schedule.start_time)
                
                utilization[resource.unit_id] = min(busy_time / time_window * 100, 100)
        
        return utilization
    
    def print_segmentation_summary(self):
        """Print summary of segmentation decisions and their impact"""
        print("\n=== Network Segmentation Summary ===")
        print(f"Segmentation enabled: {self.enable_segmentation}")
        print(f"Max overhead ratio: {self.max_segmentation_overhead_ratio:.1%}")
        print(f"Total decisions made: {self.segmentation_stats['total_decisions']}")
        print(f"Tasks with segmentation: {self.segmentation_stats['segmented_tasks']}")
        print(f"Total segmentation overhead: {self.segmentation_stats['total_overhead']:.2f}ms")
        print(f"Average estimated benefit: {self.segmentation_stats['average_benefit']:.2f}")
        
        if self.segmentation_decisions_history:
            print("\nSegmentation Decisions by Task:")
            for task_id, task in self.tasks.items():
                if task.is_segmented:
                    print(f"  {task_id} ({task.name}):")
                    print(f"    Strategy: {task.segmentation_strategy.value}")
                    print(f"    {task.get_segmentation_summary()}")
        
        print("\nSegmentation Impact by Priority:")
        priority_stats = defaultdict(lambda: {'count': 0, 'overhead': 0.0})
        for task in self.tasks.values():
            if task.is_segmented:
                priority_stats[task.priority.name]['count'] += 1
                priority_stats[task.priority.name]['overhead'] += task.total_segmentation_overhead
        
        for priority, stats in priority_stats.items():
            if stats['count'] > 0:
                avg_overhead = stats['overhead'] / stats['count']
                print(f"  {priority}: {stats['count']} tasks, avg overhead: {avg_overhead:.2f}ms")
    
    def print_schedule_summary(self):
        """Enhanced print scheduling summary with segmentation information"""
        print("=== Enhanced Scheduling Summary ===")
        print(f"NPU Resources: {len(self.resources[ResourceType.NPU])}")
        print(f"DSP Resources: {len(self.resources[ResourceType.DSP])}")
        print(f"Total Tasks: {len(self.tasks)}")
        print(f"Total Scheduled Events: {len(self.schedule_history)}")
        
        # Count by priority and runtime type
        priority_counts = defaultdict(int)
        runtime_counts = defaultdict(int)
        segmentation_counts = defaultdict(int)
        
        for task in self.tasks.values():
            priority_counts[task.priority.name] += 1
            runtime_counts[task.runtime_type.value] += 1
            segmentation_counts[task.segmentation_strategy.value] += 1
        
        print("\nTasks by Priority:")
        for priority in TaskPriority:
            count = priority_counts[priority.name]
            print(f"  {priority.name}: {count} tasks")
        
        print("\nTasks by Runtime Type:")
        for runtime_type in RuntimeType:
            count = runtime_counts[runtime_type.value]
            print(f"  {runtime_type.value}: {count} tasks")
        
        print("\nTasks by Segmentation Strategy:")
        for strategy in SegmentationStrategy:
            count = segmentation_counts[strategy.value]
            if count > 0:
                print(f"  {strategy.value}: {count} tasks")
        
        # Count task scheduling with segmentation info
        task_schedule_count = defaultdict(int)
        task_latencies = defaultdict(list)
        task_overheads = defaultdict(list)
        
        for schedule in self.schedule_history:
            task_schedule_count[schedule.task_id] += 1
            task_latencies[schedule.task_id].append(schedule.actual_latency)
            task_overheads[schedule.task_id].append(schedule.segmentation_overhead)
        
        print("\nTask Scheduling Details with Segmentation:")
        # Group by priority
        for priority in TaskPriority:
            priority_tasks = [t for t in self.tasks.values() if t.priority == priority]
            if priority_tasks:
                print(f"\n  {priority.name} Priority Tasks:")
                for task in priority_tasks:
                    count = task_schedule_count[task.task_id]
                    avg_latency = sum(task_latencies[task.task_id]) / len(task_latencies[task.task_id]) if task_latencies[task.task_id] else 0
                    avg_overhead = sum(task_overheads[task.task_id]) / len(task_overheads[task.task_id]) if task_overheads[task.task_id] else 0
                    achieved_fps = count / (self.schedule_history[-1].end_time / 1000) if self.schedule_history else 0
                    
                    seg_indicator = "🔗" if task.is_segmented else "🔒"
                    print(f"    {task.task_id} ({task.name}) [{task.runtime_type.value}] {seg_indicator}:")
                    print(f"      Scheduled Count: {count}")
                    print(f"      Average Latency: {avg_latency:.1f}ms (Required: {task.latency_requirement}ms)")
                    print(f"      Segmentation Overhead: {avg_overhead:.2f}ms")
                    print(f"      Achieved FPS: {achieved_fps:.1f} (Required: {task.fps_requirement})")
        
        # Resource utilization
        if self.schedule_history:
            time_window = self.schedule_history[-1].end_time
            utilization = self.get_resource_utilization(time_window)
            print("\nResource Utilization:")
            for resource_id, util in utilization.items():
                print(f"  {resource_id}: {util:.1f}%")
        
        # Resource binding statistics
        binding_count = defaultdict(int)
        for schedule in self.schedule_history:
            if schedule.runtime_type == RuntimeType.DSP_RUNTIME:
                binding_count[schedule.task_id] += 1
        
        if binding_count:
            print("\nDSP_Runtime Binding Statistics:")
            for task_id, count in binding_count.items():
                task = self.tasks[task_id]
                print(f"  {task_id} ({task.name}): {count} bound executions")
        
        # Print segmentation summary
        self.print_segmentation_summary()
        
        
    """
    Optional enhancements for scheduler.py to better support optimization
    Add these methods to your existing MultiResourceScheduler class
    """

    # Add these methods to your MultiResourceScheduler class:

    def reset_for_optimization(self):
        """Reset scheduler state while preserving task configurations"""
        # Reset resource queues
        for queue in self.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            queue.sub_segment_reservations.clear()
            queue.pending_sub_segments.clear()
            for p in TaskPriority:
                queue.queues[p].clear()
        
        # Reset task execution states but preserve configurations
        for task in self.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
            # Preserve segmentation configurations for CUSTOM_SEGMENTATION
            if task.segmentation_strategy != SegmentationStrategy.CUSTOM_SEGMENTATION:
                task.current_segmentation = {}
                task.selected_cut_config_index = {}
            task.total_segmentation_overhead = 0.0
            # Clear segmentation decisions but not configurations
            task.segmentation_decisions.clear()
        
        # Clear history
        self.schedule_history.clear()
        self.active_bindings.clear()
        self.segmentation_decisions_history.clear()
        
        # Reset statistics
        self.segmentation_stats = {
            'total_decisions': 0,
            'segmented_tasks': 0,
            'total_overhead': 0.0,
            'average_benefit': 0.0
        }

    def get_scheduling_metrics(self, time_window: float = None) -> Dict[str, float]:
        """Get comprehensive scheduling metrics for optimization"""
        if not self.schedule_history:
            return {}
        
        if time_window is None and self.schedule_history:
            time_window = self.schedule_history[-1].end_time
        
        metrics = {}
        
        # Task execution metrics
        task_counts = defaultdict(int)
        task_latencies = defaultdict(list)
        task_violations = defaultdict(int)
        
        for schedule in self.schedule_history:
            task_counts[schedule.task_id] += 1
            task_latencies[schedule.task_id].append(schedule.actual_latency)
        
        # Calculate per-task metrics
        total_fps_violations = 0
        total_latency_violations = 0
        
        for task_id, task in self.tasks.items():
            count = task_counts[task_id]
            
            # FPS metrics
            achieved_fps = count / (time_window / 1000.0) if time_window > 0 else 0
            fps_ratio = achieved_fps / task.fps_requirement if task.fps_requirement > 0 else 1.0
            if fps_ratio < 0.95:  # 5% tolerance
                total_fps_violations += 1
                task_violations[task_id] += 1
            
            # Latency metrics
            if task_id in task_latencies:
                avg_latency = sum(task_latencies[task_id]) / len(task_latencies[task_id])
                if avg_latency > task.latency_requirement * 1.05:  # 5% tolerance
                    total_latency_violations += 1
                    task_violations[task_id] += 1
        
        # Resource utilization
        utilization = self.get_resource_utilization(time_window)
        
        # Aggregate metrics
        metrics['total_tasks'] = len(self.tasks)
        metrics['scheduled_events'] = len(self.schedule_history)
        metrics['fps_violations'] = total_fps_violations
        metrics['latency_violations'] = total_latency_violations
        metrics['total_violations'] = total_fps_violations + total_latency_violations
        metrics['avg_utilization'] = sum(utilization.values()) / len(utilization) if utilization else 0
        metrics['min_utilization'] = min(utilization.values()) if utilization else 0
        metrics['max_utilization'] = max(utilization.values()) if utilization else 0
        metrics['utilization_variance'] = np.var(list(utilization.values())) if len(utilization) > 1 else 0
        
        # Segmentation metrics
        metrics['segmented_tasks'] = self.segmentation_stats['segmented_tasks']
        metrics['total_overhead'] = self.segmentation_stats['total_overhead']
        metrics['avg_benefit'] = self.segmentation_stats['average_benefit']
        
        # Priority-based metrics
        priority_counts = defaultdict(int)
        for task in self.tasks.values():
            priority_counts[task.priority.value] += task_counts[task.task_id]
        
        metrics['critical_executions'] = priority_counts[TaskPriority.CRITICAL.value]
        metrics['high_executions'] = priority_counts[TaskPriority.HIGH.value]
        
        return metrics

    def apply_core_assignments(self, core_assignments: Dict[str, Dict[str, str]]):
        """Apply specific core assignments for tasks (for optimizer use)
        Args:
            core_assignments: {task_id: {segment_id: core_id}}
        """
        # This is a placeholder - in a full implementation, you would modify
        # the scheduling algorithm to respect these assignments
        # For now, we'll store them for reference
        self._optimizer_core_assignments = core_assignments

    def get_task_execution_order(self) -> List[Tuple[str, float, float]]:
        """Get the execution order of tasks for analysis"""
        execution_order = []
        for schedule in self.schedule_history:
            execution_order.append((
                schedule.task_id,
                schedule.start_time,
                schedule.end_time
            ))
        return sorted(execution_order, key=lambda x: x[1])  # Sort by start time

    def analyze_resource_contention(self) -> Dict[str, List[Tuple[float, float, int]]]:
        """Analyze resource contention over time
        Returns: {resource_id: [(time, num_waiting_tasks, avg_priority)]}
        """
        contention_data = defaultdict(list)
        
        # This is a simplified analysis - you could make it more sophisticated
        time_points = sorted(set(
            [s.start_time for s in self.schedule_history] + 
            [s.end_time for s in self.schedule_history]
        ))
        
        for t in time_points:
            for resource_id, queue in self.resource_queues.items():
                # Count waiting tasks at this time point
                waiting_count = 0
                total_priority = 0
                
                for priority in TaskPriority:
                    queue_size = len(queue.queues[priority])
                    waiting_count += queue_size
                    total_priority += queue_size * priority.value
                
                if waiting_count > 0:
                    avg_priority = total_priority / waiting_count
                    contention_data[resource_id].append((t, waiting_count, avg_priority))
        
        return dict(contention_data)

    # Optional: Add this to make_segmentation_decision to better handle CUSTOM_SEGMENTATION
    def make_segmentation_decision(self, task: NNTask, current_time: float) -> Dict[str, List[str]]:
        """Make segmentation decision based on task properties and current state"""
        if not self.enable_segmentation:
            return {seg.segment_id: [] for seg in task.segments}
        
        # Get available resources
        available_resources = self.get_available_resources_info(current_time)
        
        # For CUSTOM_SEGMENTATION, the decision is already made in the task
        if task.segmentation_strategy == SegmentationStrategy.CUSTOM_SEGMENTATION:
            # Just get the current configuration
            optimal_segmentation = {}
            for segment in task.segments:
                seg_id = segment.segment_id
                if seg_id in task.preset_cut_configurations and seg_id in task.selected_cut_config_index:
                    config_idx = task.selected_cut_config_index[seg_id]
                    optimal_segmentation[seg_id] = task.preset_cut_configurations[seg_id][config_idx]
                else:
                    optimal_segmentation[seg_id] = task.current_segmentation.get(seg_id, [])
        else:
            # Use the task's get_optimal_segmentation method for other strategies
            optimal_segmentation = task.get_optimal_segmentation(available_resources, current_time)
        
        # Validate segmentation doesn't exceed overhead limits
        validated_segmentation = self._validate_segmentation_overhead(task, optimal_segmentation)
        
        # Record decision with more detailed information
        for segment_id, cuts in validated_segmentation.items():
            if cuts or segment_id in optimal_segmentation:
                segment = task.get_segment_by_id(segment_id)
                if segment:
                    decision = SegmentationDecision(
                        segment_id=segment_id,
                        task_id=task.task_id,
                        available_cuts=segment.get_available_cuts(),
                        selected_cuts=cuts,
                        decision_reason=f"{task.segmentation_strategy.value} strategy for {task.priority.name} priority {task.runtime_type.value} task",
                        estimated_benefit=self._estimate_segmentation_benefit_with_context(task, segment_id, cuts, available_resources),
                        actual_overhead=sum(cp.overhead_ms for cp in segment.cut_points if cp.op_id in cuts)
                    )
                    self.segmentation_decisions_history.append(decision)
        
        return validated_segmentation

    def _estimate_segmentation_benefit_with_context(self, task: NNTask, segment_id: str, cuts: List[str], 
                                                available_resources: Dict[ResourceType, List[float]]) -> float:
        """Enhanced benefit estimation with more context"""
        if not cuts:
            return 0.0
        
        segment = task.get_segment_by_id(segment_id)
        if not segment:
            return 0.0
        
        # Base benefit from parallelism potential
        num_available = len(available_resources.get(segment.resource_type, []))
        num_cuts = len(cuts)
        parallelism_benefit = min(num_cuts + 1, num_available) / max(num_available, 1)
        
        # Priority-based multiplier
        priority_multipliers = {
            TaskPriority.CRITICAL: 2.0,
            TaskPriority.HIGH: 1.5,
            TaskPriority.NORMAL: 1.0,
            TaskPriority.LOW: 0.5
        }
        priority_mult = priority_multipliers.get(task.priority, 1.0)
        
        # Runtime type consideration
        runtime_mult = 0.8 if task.runtime_type == RuntimeType.DSP_RUNTIME else 1.2
        
        # Resource quality consideration (bandwidth)
        if available_resources.get(segment.resource_type):
            avg_bandwidth = sum(available_resources[segment.resource_type]) / len(available_resources[segment.resource_type])
            max_possible_bandwidth = 8.0  # Assuming 8.0 is max
            quality_mult = avg_bandwidth / max_possible_bandwidth
        else:
            quality_mult = 0.5
        
        # Calculate total benefit
        total_benefit = parallelism_benefit * priority_mult * runtime_mult * quality_mult
        
        # Subtract overhead penalty
        if task.latency_requirement > 0:
            overhead_penalty = sum(cp.overhead_ms for cp in segment.cut_points if cp.op_id in cuts) / task.latency_requirement
            total_benefit *= (1 - overhead_penalty)
        
        return max(0, total_benefit)