from typing import Optional, Tuple, List, Dict
from collections import deque
from enums import TaskPriority
from task import NNTask

class ResourcePriorityQueues:
    """Enhanced priority queues for a single resource with binding support and segmentation awareness"""
    
    def __init__(self, resource_id: str):
        self.resource_id = resource_id
        # Create a queue for each priority level
        self.queues = {
            priority: deque() for priority in TaskPriority
        }
        self.available_time = 0.0  # When resource becomes available
        self.bound_until = 0.0     # When resource binding expires
        self.binding_task_id = None  # Task ID that has bound this resource
        
        # Segmentation related tracking
        self.sub_segment_reservations: List[Tuple[str, float, float]] = []  # [(sub_seg_id, start, end)]
        self.pending_sub_segments: Dict[str, List[Tuple]] = {}  # {task_id: [(sub_seg, ready_time)]}
    
    def add_task(self, task: NNTask, ready_time: float):
        """Add task to appropriate priority queue"""
        self.queues[task.priority].append((task, ready_time))
    
    def add_sub_segment_reservation(self, sub_segment_id: str, start_time: float, end_time: float):
        """Reserve resource for a specific sub-segment"""
        self.sub_segment_reservations.append((sub_segment_id, start_time, end_time))
        # Keep reservations sorted by start time
        self.sub_segment_reservations.sort(key=lambda x: x[1])
    
    def is_available_for_sub_segment(self, start_time: float, duration: float) -> bool:
        """Check if resource is available for a sub-segment during specified time window"""
        end_time = start_time + duration
        
        # Check against binding constraints
        if self.bound_until > start_time and self.binding_task_id:
            return False
        
        # Check against existing reservations
        for _, res_start, res_end in self.sub_segment_reservations:
            if not (end_time <= res_start or start_time >= res_end):
                return False  # Overlap detected
        
        return True
    
    def get_earliest_available_time_for_duration(self, duration: float, after_time: float = 0.0) -> float:
        """Find the earliest time this resource can accommodate a task of given duration"""
        candidate_start = max(after_time, self.available_time)
        
        # If resource is bound, wait until binding expires
        if self.bound_until > candidate_start:
            candidate_start = self.bound_until
        
        # Check for gaps in reservations
        for _, res_start, res_end in self.sub_segment_reservations:
            if res_start >= candidate_start:
                # Check if we can fit before this reservation
                if candidate_start + duration <= res_start:
                    return candidate_start
                else:
                    # Move candidate start to after this reservation
                    candidate_start = res_end
            elif res_end > candidate_start:
                # This reservation overlaps with our candidate start
                candidate_start = res_end
        
        return candidate_start
    
    def is_bound_to_other_task(self, task_id: str, current_time: float) -> bool:
        """Check if resource is bound to a different task"""
        return (self.bound_until > current_time and 
                self.binding_task_id is not None and 
                self.binding_task_id != task_id)
    
    def bind_resource(self, task_id: str, end_time: float):
        """Bind resource to a specific task until end_time"""
        self.binding_task_id = task_id
        self.bound_until = end_time
    
    def release_binding(self):
        """Release resource binding"""
        self.binding_task_id = None
        self.bound_until = 0.0
    
    def cleanup_expired_reservations(self, current_time: float):
        """Remove expired sub-segment reservations"""
        self.sub_segment_reservations = [
            (sub_seg_id, start, end) for sub_seg_id, start, end in self.sub_segment_reservations
            if end > current_time
        ]
    
    def get_next_task(self, current_time: float) -> Optional[Tuple[NNTask, float]]:
        """Get next task to execute based on priority with segmentation consideration"""
        # Clean up expired reservations first
        self.cleanup_expired_reservations(current_time)
        
        # Check queues from highest to lowest priority
        for priority in TaskPriority:
            queue = self.queues[priority]
            # Remove tasks that are not ready yet
            ready_tasks = []
            while queue:
                task, ready_time = queue.popleft()
                
                # Enhanced readiness check with segmentation consideration
                if self._is_task_ready_for_execution(task, ready_time, current_time):
                    ready_tasks.append((task, ready_time))
                else:
                    # Put back if not ready
                    queue.append((task, ready_time))
                    break
            
            # If we found ready tasks at this priority level, return the first one
            if ready_tasks:
                # Put remaining ready tasks back
                for t in ready_tasks[1:]:
                    queue.append(t)
                return ready_tasks[0]
        
        return None
    
    def _is_task_ready_for_execution(self, task: NNTask, ready_time: float, current_time: float) -> bool:
        """Enhanced task readiness check considering segmentation and binding constraints"""
        # Basic readiness checks
        if ready_time > current_time:
            return False
        
        if task.last_execution_time + task.min_interval_ms > current_time:
            return False
        
        if self.is_bound_to_other_task(task.task_id, current_time):
            return False
        
        # Additional check for segmented tasks
        if task.is_segmented:
            # For segmented tasks, check if any sub-segment can be scheduled
            sub_segments = task.get_sub_segments_for_scheduling()
            resource_type = None
            
            # Find sub-segments that use this resource type
            for sub_seg in sub_segments:
                if not resource_type:
                    # Determine resource type from first sub-segment (assuming homogeneous resource)
                    resource_type = sub_seg.resource_type
                
                if sub_seg.resource_type == resource_type:
                    # Check if this sub-segment can be scheduled now
                    earliest_start = self.get_earliest_available_time_for_duration(
                        sub_seg.get_duration(4.0), current_time + sub_seg.start_time
                    )
                    if earliest_start <= current_time + sub_seg.start_time + 1.0:  # Allow 1ms tolerance
                        return True
            
            return False  # No sub-segments can be scheduled immediately
        
        return True  # Non-segmented task is ready
    
    def has_higher_priority_tasks(self, priority: TaskPriority, current_time: float, task_id: str = None) -> bool:
        """Check if there are higher priority tasks waiting with segmentation awareness"""
        for p in TaskPriority:
            if p < priority:  # Higher priority
                # Check if any task in this queue is ready
                for task, ready_time in self.queues[p]:
                    if self._is_task_ready_for_execution(task, ready_time, current_time):
                        return True
        return False
    
    def get_resource_availability_windows(self, time_horizon: float, current_time: float) -> List[Tuple[float, float]]:
        """Get list of time windows when resource is available for new allocations"""
        windows = []
        
        # Start from current available time or current time, whichever is later
        search_start = max(current_time, self.available_time)
        
        # If resource is bound, start search after binding expires
        if self.bound_until > search_start:
            search_start = self.bound_until
        
        # Create list of all occupied time periods
        occupied_periods = [(res_start, res_end) for _, res_start, res_end in self.sub_segment_reservations
                           if res_end > search_start]
        
        # Sort by start time
        occupied_periods.sort()
        
        # Find gaps between occupied periods
        window_start = search_start
        
        for period_start, period_end in occupied_periods:
            if period_start > window_start:
                # Found a gap
                windows.append((window_start, period_start))
            window_start = max(window_start, period_end)
        
        # Add final window from last occupied period to time horizon
        if window_start < current_time + time_horizon:
            windows.append((window_start, current_time + time_horizon))
        
        return windows
    
    def estimate_scheduling_delay(self, task: NNTask, current_time: float, required_duration: float) -> float:
        """Estimate delay before task can be scheduled on this resource"""
        if not self.is_bound_to_other_task(task.task_id, current_time):
            earliest_time = self.get_earliest_available_time_for_duration(required_duration, current_time)
            return max(0, earliest_time - current_time)
        else:
            # Resource is bound to another task, wait until binding expires
            return max(0, self.bound_until - current_time)
    
    def get_utilization_metrics(self, time_window: float, current_time: float) -> Dict[str, float]:
        """Calculate resource utilization metrics considering segmentation"""
        total_busy_time = 0.0
        fragmentation_time = 0.0
        
        # Calculate busy time from reservations
        for _, start, end in self.sub_segment_reservations:
            if start < current_time + time_window and end > current_time:
                # Calculate overlap with time window
                overlap_start = max(start, current_time)
                overlap_end = min(end, current_time + time_window)
                total_busy_time += overlap_end - overlap_start
        
        # Calculate fragmentation (small gaps between reservations)
        windows = self.get_resource_availability_windows(time_window, current_time)
        for window_start, window_end in windows:
            window_duration = window_end - window_start
            if 0 < window_duration < 5.0:  # Consider gaps smaller than 5ms as fragmentation
                fragmentation_time += window_duration
        
        utilization = (total_busy_time / time_window) * 100 if time_window > 0 else 0
        fragmentation_ratio = (fragmentation_time / time_window) * 100 if time_window > 0 else 0
        
        return {
            'utilization_percent': min(utilization, 100),
            'fragmentation_percent': fragmentation_ratio,
            'effective_utilization_percent': min(utilization - fragmentation_ratio, 100),
            'num_reservations': len(self.sub_segment_reservations),
            'num_available_windows': len(windows)
        }
    
    def optimize_sub_segment_placement(self, sub_segments: List[Tuple], current_time: float) -> List[Tuple[str, float, float]]:
        """Optimize placement of multiple sub-segments to minimize fragmentation"""
        if not sub_segments:
            return []
        
        optimized_schedule = []
        
        # Sort sub-segments by duration (longest first) for better packing
        sorted_segments = sorted(sub_segments, key=lambda x: x[1], reverse=True)  # x[1] is duration
        
        for sub_seg_id, duration, preferred_start in sorted_segments:
            # Find best placement considering existing schedule
            best_start = self.get_earliest_available_time_for_duration(duration, 
                                                                     max(preferred_start, current_time))
            best_end = best_start + duration
            
            # Add to optimized schedule
            optimized_schedule.append((sub_seg_id, best_start, best_end))
            
            # Temporarily add this reservation to check future placements
            self.sub_segment_reservations.append((sub_seg_id, best_start, best_end))
            self.sub_segment_reservations.sort(key=lambda x: x[1])
        
        # Remove temporary reservations
        for sub_seg_id, _, _ in optimized_schedule:
            self.sub_segment_reservations = [(sid, s, e) for sid, s, e in self.sub_segment_reservations 
                                           if sid != sub_seg_id]
        
        return optimized_schedule
    
    def get_status_summary(self, current_time: float) -> str:
        """Get a human-readable summary of resource status"""
        status_parts = []
        
        # Basic availability
        if self.available_time > current_time:
            status_parts.append(f"busy until {self.available_time:.1f}ms")
        else:
            status_parts.append("available")
        
        # Binding status
        if self.bound_until > current_time:
            status_parts.append(f"bound to {self.binding_task_id} until {self.bound_until:.1f}ms")
        
        # Reservation count
        active_reservations = len([r for r in self.sub_segment_reservations 
                                 if r[2] > current_time])  # r[2] is end_time
        if active_reservations > 0:
            status_parts.append(f"{active_reservations} active reservations")
        
        # Queue status
        total_queued = sum(len(queue) for queue in self.queues.values())
        if total_queued > 0:
            status_parts.append(f"{total_queued} tasks queued")
        
        return f"{self.resource_id}: {', '.join(status_parts)}"