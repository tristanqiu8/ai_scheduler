#!/usr/bin/env python3
"""
Base scheduler interface and simple scheduling implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import heapq
from collections import defaultdict, deque
import numpy as np

from enums import (
    ResourceType, TaskPriority, RuntimeType, TaskState,
    SchedulingAlgorithm, OptimizationObjective, SchedulerConfig
)
from models import (
    ResourceUnit, TaskScheduleInfo, ResourceBinding, 
    SchedulingDecision, SystemState, SchedulingMetrics
)
from task import NNTask, TaskSet


class BaseScheduler(ABC):
    """Abstract base class for all schedulers"""
    
    def __init__(self, resources: Dict[str, ResourceUnit], 
                 objectives: List[OptimizationObjective] = None):
        self.resources = resources
        self.objectives = objectives or [OptimizationObjective.MINIMIZE_MAKESPAN]
        
        # System state
        self.system_state = SystemState()
        self.system_state.resource_states = resources.copy()
        
        # Scheduling history
        self.schedule_history: List[TaskScheduleInfo] = []
        self.decision_history: List[SchedulingDecision] = []
        
        # Performance tracking
        self.metrics = SchedulingMetrics()
        
    @abstractmethod
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Main scheduling method - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get the name of the scheduling algorithm"""
        pass
    
    def reset(self):
        """Reset scheduler state"""
        self.system_state = SystemState()
        self.system_state.resource_states = self.resources.copy()
        self.schedule_history.clear()
        self.decision_history.clear()
        self.metrics = SchedulingMetrics()
    
    def calculate_metrics(self, schedule: List[TaskScheduleInfo]) -> SchedulingMetrics:
        """Calculate performance metrics for a schedule"""
        metrics = SchedulingMetrics()
        
        if not schedule:
            return metrics
        
        # Timing metrics
        metrics.makespan_ms = max(s.end_time_ms for s in schedule)
        
        latencies = [s.get_latency() for s in schedule]
        metrics.average_latency_ms = np.mean(latencies) if latencies else 0
        
        # Throughput
        metrics.completed_tasks = len([s for s in schedule if s.state == TaskState.COMPLETED])
        if metrics.makespan_ms > 0:
            metrics.throughput_tasks_per_sec = metrics.completed_tasks / (metrics.makespan_ms / 1000)
        
        # Resource utilization
        resource_usage = defaultdict(float)
        resource_time = defaultdict(float)
        
        for sched in schedule:
            for seg_info in sched.segment_schedule:
                seg_id, start_ms, end_ms, res_id = seg_info
                resource_usage[res_id] += end_ms - start_ms
                resource_time[res_id] = max(resource_time[res_id], end_ms)
        
        # Calculate utilization by resource type
        for res_type in ResourceType:
            type_usage = 0
            type_time = 0
            type_count = 0
            
            for res_id, resource in self.resources.items():
                if resource.resource_type == res_type:
                    type_usage += resource_usage.get(res_id, 0)
                    type_time += resource_time.get(res_id, metrics.makespan_ms)
                    type_count += 1
            
            if type_count > 0 and type_time > 0:
                metrics.average_utilization[res_type] = type_usage / type_time
        
        self.metrics = metrics
        return metrics
    
    def validate_schedule(self, schedule: List[TaskScheduleInfo]) -> Tuple[bool, List[str]]:
        """Validate a schedule for correctness"""
        errors = []
        
        # Check resource conflicts
        resource_timeline = defaultdict(list)
        
        for sched in schedule:
            for seg_info in sched.segment_schedule:
                seg_id, start_ms, end_ms, res_id = seg_info
                
                # Check for overlaps
                for existing_start, existing_end in resource_timeline[res_id]:
                    if not (end_ms <= existing_start or start_ms >= existing_end):
                        errors.append(f"Resource conflict on {res_id} at {start_ms}-{end_ms}")
                
                resource_timeline[res_id].append((start_ms, end_ms))
        
        # Check task constraints
        # (Additional validation can be added here)
        
        return len(errors) == 0, errors


class SimpleScheduler(BaseScheduler):
    """Simple priority-based scheduler with basic algorithms"""
    
    def __init__(self, resources: Dict[str, ResourceUnit], 
                 algorithm: SchedulingAlgorithm = SchedulingAlgorithm.PRIORITY_BASED):
        super().__init__(resources)
        self.algorithm = algorithm
        
        # Priority queues for each priority level
        self.priority_queues = {
            priority: deque() for priority in TaskPriority
        }
        
    def get_algorithm_name(self) -> str:
        return f"Simple_{self.algorithm.value}"
    
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Schedule tasks using simple algorithms"""
        if self.algorithm == SchedulingAlgorithm.PRIORITY_BASED:
            return self._priority_based_schedule(tasks, time_limit_ms)
        elif self.algorithm == SchedulingAlgorithm.FIFO:
            return self._fifo_schedule(tasks, time_limit_ms)
        elif self.algorithm == SchedulingAlgorithm.EDF:
            return self._edf_schedule(tasks, time_limit_ms)
        elif self.algorithm == SchedulingAlgorithm.ROUND_ROBIN:
            return self._round_robin_schedule(tasks, time_limit_ms)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def _priority_based_schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Simple priority-based scheduling"""
        schedule = []
        current_time = 0.0
        completed_tasks = set()
        
        # Initialize priority queues
        for priority in TaskPriority:
            self.priority_queues[priority].clear()
        
        # Main scheduling loop
        while current_time < time_limit_ms:
            # Get ready tasks
            ready_tasks = tasks.get_ready_tasks(current_time, completed_tasks)
            
            # Add to priority queues
            for task in ready_tasks:
                if task.id not in [t[1] for q in self.priority_queues.values() for t in q]:
                    self.priority_queues[task.priority].append((current_time, task.id))
            
            # Get next task to schedule
            next_task = None
            for priority in TaskPriority:
                if self.priority_queues[priority]:
                    # Check wait time for non-critical priorities
                    if priority != TaskPriority.CRITICAL:
                        wait_time = priority.wait_time_ms()
                        # Simple check - in real implementation would track actual wait
                        if current_time < wait_time:
                            continue
                    
                    enqueue_time, task_id = self.priority_queues[priority].popleft()
                    next_task = tasks.get_task(task_id)
                    break
            
            if not next_task:
                # No task ready, advance time
                current_time += 0.1
                continue
            
            # Schedule the task
            schedule_info = self._schedule_task_simple(next_task, current_time)
            if schedule_info:
                schedule.append(schedule_info)
                current_time = schedule_info.end_time_ms
                completed_tasks.add(next_task.id)
                
                # Update task state
                next_task.last_scheduled_ms = schedule_info.start_time_ms
                next_task.completion_count += 1
                next_task.update_performance_history(schedule_info.get_latency())
            else:
                # Cannot schedule, try next time
                current_time += 0.1
        
        return schedule
    
    def _schedule_task_simple(self, task: NNTask, start_time: float) -> Optional[TaskScheduleInfo]:
        """Simple task scheduling without optimization"""
        schedule_info = TaskScheduleInfo(
            task_id=task.id,
            start_time_ms=start_time,
            state=TaskState.RUNNING
        )
        
        current_time = start_time
        segment_schedule = []
        resource_assignments = {}
        
        # Simple segmentation decision
        available_npus = len([r for r in self.resources.values() 
                            if r.resource_type == ResourceType.NPU and r.available_at_ms <= current_time])
        segmentation_decisions = task.apply_segmentation_strategy({ResourceType.NPU: available_npus})
        
        # Schedule each segment
        for segment in task.segments:
            # Find available resource
            resource = self._find_available_resource(segment.resource_type, current_time)
            if not resource:
                return None  # Cannot schedule
            
            # Calculate duration
            duration = segment.get_total_duration(resource.bandwidth)
            
            # Add to schedule
            segment_schedule.append((segment.id, current_time, current_time + duration, resource.id))
            resource_assignments[segment.id] = resource.id
            
            # Update resource availability
            resource.available_at_ms = current_time + duration
            resource.total_usage_ms += duration
            
            # Handle sub-segments if segmented
            if segment.is_segmented:
                sub_schedule = []
                sub_time = current_time
                
                for sub_seg in segment.sub_segments:
                    sub_duration = sub_seg.get_duration(resource.bandwidth) + sub_seg.cut_overhead_ms
                    sub_schedule.append((sub_seg.id, sub_time, sub_time + sub_duration, resource.id))
                    sub_time += sub_duration
                
                schedule_info.sub_segment_schedule = sub_schedule
            
            current_time += duration
        
        schedule_info.end_time_ms = current_time
        schedule_info.actual_duration_ms = current_time - start_time
        schedule_info.segment_schedule = segment_schedule
        schedule_info.resource_assignments = resource_assignments
        schedule_info.state = TaskState.COMPLETED
        
        return schedule_info
    
    def _find_available_resource(self, resource_type: ResourceType, 
                               current_time: float) -> Optional[ResourceUnit]:
        """Find first available resource of given type"""
        candidates = [
            r for r in self.resources.values()
            if r.resource_type == resource_type and r.available_at_ms <= current_time
        ]
        
        if not candidates:
            return None
        
        # Return highest bandwidth resource
        return max(candidates, key=lambda r: r.bandwidth)
    
    def _fifo_schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """First-In-First-Out scheduling"""
        # Convert to list and sort by task creation order
        task_list = sorted(tasks.tasks.values(), key=lambda t: t.id)
        
        schedule = []
        current_time = 0.0
        
        for task in task_list:
            if current_time >= time_limit_ms:
                break
            
            schedule_info = self._schedule_task_simple(task, current_time)
            if schedule_info:
                schedule.append(schedule_info)
                current_time = schedule_info.end_time_ms
        
        return schedule
    
    def _edf_schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Earliest Deadline First scheduling"""
        schedule = []
        current_time = 0.0
        completed_tasks = set()
        
        while current_time < time_limit_ms:
            # Get ready tasks
            ready_tasks = tasks.get_ready_tasks(current_time, completed_tasks)
            
            if not ready_tasks:
                current_time += 0.1
                continue
            
            # Sort by deadline (approximated by latency requirement)
            ready_tasks.sort(key=lambda t: t.constraints.latency_requirement_ms)
            
            # Schedule task with earliest deadline
            next_task = ready_tasks[0]
            schedule_info = self._schedule_task_simple(next_task, current_time)
            
            if schedule_info:
                schedule.append(schedule_info)
                current_time = schedule_info.end_time_ms
                completed_tasks.add(next_task.id)
                next_task.last_scheduled_ms = schedule_info.start_time_ms
                next_task.completion_count += 1
            else:
                current_time += 0.1
        
        return schedule
    
    def _round_robin_schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Round-robin scheduling with priority consideration"""
        schedule = []
        current_time = 0.0
        task_queue = deque()
        completed_tasks = set()
        
        # Initialize with all tasks
        for priority in TaskPriority:
            priority_tasks = tasks.get_tasks_by_priority(priority)
            task_queue.extend(priority_tasks)
        
        while current_time < time_limit_ms and task_queue:
            task = task_queue.popleft()
            
            # Check if task is ready
            if task.last_scheduled_ms + task.constraints.get_period_ms() > current_time:
                task_queue.append(task)  # Re-queue
                current_time += 0.1
                continue
            
            # Check dependencies
            if not task.check_dependencies_met(completed_tasks):
                task_queue.append(task)  # Re-queue
                current_time += 0.1
                continue
            
            # Schedule the task
            schedule_info = self._schedule_task_simple(task, current_time)
            
            if schedule_info:
                schedule.append(schedule_info)
                current_time = schedule_info.end_time_ms
                task.last_scheduled_ms = schedule_info.start_time_ms
                task.completion_count += 1
                
                # Re-queue for next iteration
                task_queue.append(task)
            else:
                current_time += 0.1
                task_queue.append(task)
        
        return schedule


class PriorityQueueScheduler(BaseScheduler):
    """Advanced priority queue scheduler with preemption support"""
    
    def __init__(self, resources: Dict[str, ResourceUnit]):
        super().__init__(resources)
        
        # Multi-level priority queues
        self.ready_queues = {
            priority: [] for priority in TaskPriority  # Using heapq
        }
        
        # Resource bindings for DSP_RUNTIME
        self.active_bindings: List[ResourceBinding] = []
        
        # Preemption tracking
        self.preempted_tasks: List[Tuple[float, NNTask]] = []  # (resume_time, task)
        
    def get_algorithm_name(self) -> str:
        return "Advanced_PriorityQueue"
    
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Advanced scheduling with preemption and resource binding"""
        schedule = []
        current_time = 0.0
        completed_tasks = set()
        running_tasks = {}  # task_id -> TaskScheduleInfo
        
        # Discrete event simulation
        events = []  # (time, event_type, data)
        
        # Initialize with task arrivals
        for task in tasks.tasks.values():
            heapq.heappush(events, (0.0, 'arrival', task.id))
        
        while events and current_time < time_limit_ms:
            event_time, event_type, event_data = heapq.heappop(events)
            current_time = event_time
            
            if current_time >= time_limit_ms:
                break
            
            if event_type == 'arrival':
                # Task arrival event
                task_id = event_data
                task = tasks.get_task(task_id)
                
                if task and task.check_dependencies_met(completed_tasks):
                    # Add to ready queue
                    priority_value = task.priority.value
                    heapq.heappush(self.ready_queues[task.priority], 
                                 (priority_value, current_time, task_id))
                    
                    # Try to schedule
                    self._try_schedule_tasks(tasks, current_time, schedule, 
                                           running_tasks, events, completed_tasks)
                
                # Schedule next arrival
                next_arrival = current_time + task.constraints.get_period_ms()
                if next_arrival < time_limit_ms:
                    heapq.heappush(events, (next_arrival, 'arrival', task_id))
            
            elif event_type == 'completion':
                # Task completion event
                task_id, schedule_info = event_data
                
                # Mark as completed
                completed_tasks.add(task_id)
                schedule_info.state = TaskState.COMPLETED
                
                # Release resources
                self._release_resources(schedule_info, current_time)
                
                # Remove from running tasks
                if task_id in running_tasks:
                    del running_tasks[task_id]
                
                # Try to schedule waiting tasks
                self._try_schedule_tasks(tasks, current_time, schedule,
                                       running_tasks, events, completed_tasks)
            
            elif event_type == 'preemption_check':
                # Check if preemption is needed
                self._check_preemption(tasks, current_time, schedule,
                                     running_tasks, events)
        
        return schedule
    
    def _try_schedule_tasks(self, tasks: TaskSet, current_time: float,
                          schedule: List[TaskScheduleInfo],
                          running_tasks: Dict[str, TaskScheduleInfo],
                          events: List, completed_tasks: Set[str]):
        """Try to schedule ready tasks"""
        
        # Check each priority level
        for priority in TaskPriority:
            while self.ready_queues[priority]:
                # Peek at highest priority task
                _, arrival_time, task_id = self.ready_queues[priority][0]
                task = tasks.get_task(task_id)
                
                if not task:
                    heapq.heappop(self.ready_queues[priority])
                    continue
                
                # Check if we can schedule this task
                can_schedule, resources = self._can_schedule_task(task, current_time)
                
                if can_schedule:
                    # Remove from ready queue
                    heapq.heappop(self.ready_queues[priority])
                    
                    # Create schedule
                    schedule_info = self._create_schedule(task, current_time, resources)
                    schedule.append(schedule_info)
                    running_tasks[task.id] = schedule_info
                    
                    # Update task state
                    task.state = TaskState.RUNNING
                    task.last_scheduled_ms = current_time
                    
                    # Schedule completion event
                    completion_time = schedule_info.end_time_ms
                    heapq.heappush(events, (completion_time, 'completion', 
                                          (task.id, schedule_info)))
                    
                    # Schedule preemption checks if ACPU runtime
                    if task.runtime_type == RuntimeType.ACPU_RUNTIME:
                        check_time = current_time + SchedulerConfig.DEFAULT_TIME_QUANTUM_MS
                        heapq.heappush(events, (check_time, 'preemption_check', task.id))
                else:
                    # Cannot schedule tasks at this priority
                    break
    
    def _can_schedule_task(self, task: NNTask, current_time: float) -> Tuple[bool, Dict[str, ResourceUnit]]:
        """Check if task can be scheduled and return resources"""
        required_resources = {}
        
        if task.runtime_type == RuntimeType.DSP_RUNTIME:
            # Need to allocate all resources at once
            all_available = True
            
            for segment in task.segments:
                resource = self._find_best_resource(segment.resource_type, current_time)
                if not resource or resource.available_at_ms > current_time:
                    all_available = False
                    break
                required_resources[segment.id] = resource
            
            return all_available, required_resources
        
        else:  # ACPU_RUNTIME
            # Just need resource for first segment
            if task.segments:
                first_segment = task.segments[0]
                resource = self._find_best_resource(first_segment.resource_type, current_time)
                
                if resource:
                    # Check if we should preempt
                    if resource.current_task_id:
                        current_task = self._get_running_task(resource.current_task_id)
                        if current_task and current_task.priority > task.priority:
                            # Can preempt
                            required_resources[first_segment.id] = resource
                            return True, required_resources
                    elif resource.available_at_ms <= current_time:
                        required_resources[first_segment.id] = resource
                        return True, required_resources
            
            return False, {}
    
    def _find_best_resource(self, resource_type: ResourceType, 
                          current_time: float) -> Optional[ResourceUnit]:
        """Find best available resource considering thermal and performance"""
        candidates = [
            r for r in self.resources.values()
            if r.resource_type == resource_type
        ]
        
        if not candidates:
            return None
        
        # Score each candidate
        best_score = -float('inf')
        best_resource = None
        
        for resource in candidates:
            # Base score on bandwidth
            score = resource.bandwidth
            
            # Thermal penalty
            thermal_factor = resource.get_thermal_throttle_factor()
            score *= thermal_factor
            
            # Availability bonus
            if resource.available_at_ms <= current_time:
                score *= 1.2
            
            # Utilization penalty (load balancing)
            if self.system_state.current_time_ms > 0:
                utilization = resource.total_usage_ms / self.system_state.current_time_ms
                score *= (1.0 - utilization * 0.5)
            
            if score > best_score:
                best_score = score
                best_resource = resource
        
        return best_resource
    
    def _create_schedule(self, task: NNTask, start_time: float,
                       resources: Dict[str, ResourceUnit]) -> TaskScheduleInfo:
        """Create detailed schedule for task"""
        schedule_info = TaskScheduleInfo(
            task_id=task.id,
            start_time_ms=start_time,
            state=TaskState.RUNNING
        )
        
        current_time = start_time
        segment_schedule = []
        sub_segment_schedule = []
        
        # Apply segmentation
        available_resources = defaultdict(int)
        for res in self.resources.values():
            available_resources[res.resource_type] += 1
        
        segmentation_decisions = task.apply_segmentation_strategy(dict(available_resources))
        
        # Schedule segments
        for segment in task.segments:
            if task.runtime_type == RuntimeType.DSP_RUNTIME:
                # Use pre-allocated resource
                resource = resources.get(segment.id)
            else:
                # Find resource dynamically
                resource = self._find_best_resource(segment.resource_type, current_time)
            
            if not resource:
                # This shouldn't happen if _can_schedule_task worked correctly
                continue
            
            # Calculate duration with thermal throttling
            base_duration = segment.get_total_duration(resource.bandwidth)
            thermal_factor = resource.get_thermal_throttle_factor()
            actual_duration = base_duration / thermal_factor
            
            # Update resource state
            resource.available_at_ms = current_time + actual_duration
            resource.current_task_id = task.id
            resource.total_usage_ms += actual_duration
            
            # Update thermal model
            power = resource.bandwidth * 2.0  # Simplified power model
            resource.update_temperature(power, actual_duration)
            
            # Add to schedule
            segment_schedule.append((segment.id, current_time, 
                                   current_time + actual_duration, resource.id))
            
            # Handle sub-segments
            if segment.is_segmented:
                sub_time = current_time
                for sub_seg in segment.sub_segments:
                    sub_duration = sub_seg.get_duration(resource.bandwidth) / thermal_factor
                    sub_duration += sub_seg.cut_overhead_ms
                    
                    sub_segment_schedule.append((sub_seg.id, sub_time,
                                               sub_time + sub_duration, resource.id))
                    sub_time += sub_duration
            
            current_time += actual_duration
        
        # Create resource binding for DSP_RUNTIME
        if task.runtime_type == RuntimeType.DSP_RUNTIME:
            binding = ResourceBinding(
                task_id=task.id,
                resource_ids=set(r.id for r in resources.values()),
                binding_start_ms=start_time,
                binding_end_ms=current_time,
                is_active=True
            )
            self.active_bindings.append(binding)
        
        schedule_info.end_time_ms = current_time
        schedule_info.actual_duration_ms = current_time - start_time
        schedule_info.segment_schedule = segment_schedule
        schedule_info.sub_segment_schedule = sub_segment_schedule
        
        return schedule_info
    
    def _release_resources(self, schedule_info: TaskScheduleInfo, current_time: float):
        """Release resources after task completion"""
        # Release individual resources
        for seg_id, start, end, res_id in schedule_info.segment_schedule:
            if res_id in self.resources:
                self.resources[res_id].current_task_id = None
        
        # Release bindings
        self.active_bindings = [
            b for b in self.active_bindings 
            if b.task_id != schedule_info.task_id or not b.is_active
        ]
    
    def _check_preemption(self, tasks: TaskSet, current_time: float,
                        schedule: List[TaskScheduleInfo],
                        running_tasks: Dict[str, TaskScheduleInfo],
                        events: List):
        """Check if any running task should be preempted"""
        # Only check high priority ready tasks
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
            if not self.ready_queues[priority]:
                continue
            
            _, _, ready_task_id = self.ready_queues[priority][0]
            ready_task = tasks.get_task(ready_task_id)
            
            if not ready_task:
                continue
            
            # Check each running lower priority task
            for running_id, running_schedule in list(running_tasks.items()):
                running_task = tasks.get_task(running_id)
                
                if not running_task:
                    continue
                
                # Can only preempt ACPU_RUNTIME tasks with lower priority
                if (running_task.runtime_type == RuntimeType.ACPU_RUNTIME and
                    running_task.priority > ready_task.priority):
                    
                    # Preempt the task
                    self._preempt_task(running_task, running_schedule, current_time)
                    del running_tasks[running_id]
                    
                    # Add to preempted list
                    self.preempted_tasks.append((current_time + 1.0, running_task))
                    
                    # Try to schedule the high priority task
                    self._try_schedule_tasks(tasks, current_time, schedule,
                                           running_tasks, events, set())
                    break
    
    def _preempt_task(self, task: NNTask, schedule_info: TaskScheduleInfo, 
                     current_time: float):
        """Preempt a running task"""
        task.state = TaskState.PREEMPTED
        schedule_info.state = TaskState.PREEMPTED
        schedule_info.preemption_count += 1
        
        # Calculate completion ratio
        total_duration = schedule_info.actual_duration_ms
        elapsed = current_time - schedule_info.start_time_ms
        schedule_info.completion_ratio = elapsed / total_duration if total_duration > 0 else 0
        
        # Release resources
        self._release_resources(schedule_info, current_time)
    
    def _get_running_task(self, task_id: str) -> Optional[NNTask]:
        """Get currently running task by ID"""
        # This would be implemented with proper task tracking
        return None