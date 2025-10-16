#!/usr/bin/env python3
"""
调度追踪器 - 用于追踪和可视化整个系统的任务执行时间线
支持生成甘特图和Chrome Tracing格式
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import copy

from .enums import ResourceType, TaskPriority
from .resource_queue import ResourceQueueManager


class EventType(Enum):
    """事件类型"""
    TASK_ENQUEUE = "enqueue"
    TASK_START = "start"
    TASK_END = "end"
    TASK_PREEMPT = "preempt"
    TASK_RESUME = "resume"
    RESOURCE_IDLE = "idle"
    RESOURCE_BUSY = "busy"


@dataclass
class TraceEvent:
    """追踪事件"""
    timestamp: float
    event_type: EventType
    task_id: str
    resource_id: str
    resource_type: ResourceType
    priority: TaskPriority
    
    # 额外信息
    segment_id: Optional[str] = None
    bandwidth: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """任务执行记录"""
    task_id: str
    root_task_id: str
    resource_id: str
    resource_type: ResourceType
    priority: TaskPriority
    start_time: float
    end_time: float
    bandwidth: float
    task_name: Optional[str] = None
    instance_id: Optional[int] = None
    segment_id: Optional[str] = None
    segment_index: Optional[int] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ScheduleTracer:
    """调度追踪器 - 记录和分析整个系统的执行时间线"""
    
    def __init__(self, queue_manager: ResourceQueueManager):
        self.queue_manager = queue_manager
        self.events: List[TraceEvent] = []
        self.executions: List[TaskExecution] = []
        
        # 任务相关信息
        self.task_info: Dict[str, Dict[str, Any]] = {}
        self.segment_to_task_key: Dict[str, str] = {}
        self.priority_overrides: Dict[str, TaskPriority] = {}
        
        # 资源利用率追踪
        self.resource_busy_time: Dict[str, float] = defaultdict(float)
        self.resource_last_busy: Dict[str, float] = defaultdict(float)
        
        # 时间范围
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def record_enqueue(self, task_id: str, resource_id: str, priority: TaskPriority,
                      ready_time: float, segments: List[Any],
                      original_task_id: Optional[str] = None,
                      task_name: Optional[str] = None,
                      instance_id: Optional[int] = None,
                      segment_index: Optional[int] = None,
                      jitter_ms: Optional[float] = None):
        """记录任务入队事件"""
        root_task_id, parsed_instance, parsed_segment_index = self._parse_task_identifier(task_id)
        resolved_root_task_id = original_task_id or root_task_id
        resolved_instance_id = instance_id if instance_id is not None else parsed_instance
        resolved_segment_index = segment_index if segment_index is not None else parsed_segment_index
        
        if resolved_root_task_id is None:
            resolved_root_task_id = task_id
        if resolved_instance_id is None:
            resolved_instance_id = 0
        
        task_key = self._make_task_key(resolved_root_task_id, resolved_instance_id)
        self.segment_to_task_key[task_id] = task_key
        self.segment_to_task_key.setdefault(resolved_root_task_id, task_key)
        
        metadata = {"segment_count": len(segments)}
        if jitter_ms is not None:
            metadata["jitter_ms"] = jitter_ms

        event = TraceEvent(
            timestamp=ready_time,
            event_type=EventType.TASK_ENQUEUE,
            task_id=task_id,
            resource_id=resource_id,
            resource_type=self._get_resource_type(resource_id),
            priority=priority,
            metadata=metadata
        )
        self.events.append(event)
        
        # 保存任务信息
        info = self.task_info.get(task_key)
        if info is None:
            info = {
                "root_task_id": resolved_root_task_id,
                "task_name": task_name,
                "priority": priority,
                "ready_time": ready_time,
                "instance_id": resolved_instance_id,
                "segment_ids": set([task_id]),
                "resource_ids": {resource_id},
                "first_segment_index": resolved_segment_index,
            }
            if jitter_ms is not None:
                info["jitter_ms"] = jitter_ms
            self.task_info[task_key] = info
        else:
            info["priority"] = priority
            info.setdefault("segment_ids", set()).add(task_id)
            info.setdefault("resource_ids", set()).add(resource_id)
            if resolved_segment_index is not None:
                current = info.get("first_segment_index")
                if current is None or resolved_segment_index < current:
                    info["first_segment_index"] = resolved_segment_index
            ref_index = info.get("first_segment_index")
            if ref_index is None or resolved_segment_index == ref_index:
                info["ready_time"] = min(info.get("ready_time", ready_time), ready_time)
                if jitter_ms is not None:
                    info["jitter_ms"] = jitter_ms
            if task_name:
                info["task_name"] = task_name
        
        if task_name:
            info["task_name"] = task_name
    
    def record_execution(self, task_id: str, resource_id: str, 
                        start_time: float, end_time: float,
                        bandwidth: float, segment_id: Optional[str] = None,
                        original_task_id: Optional[str] = None,
                        task_name: Optional[str] = None,
                        instance_id: Optional[int] = None,
                        segment_index: Optional[int] = None):
        """记录任务执行"""
        root_task_id, parsed_instance, parsed_segment_index = self._parse_task_identifier(task_id)
        resolved_root_task_id = original_task_id or root_task_id or task_id
        resolved_instance_id = instance_id if instance_id is not None else parsed_instance
        resolved_segment_index = segment_index if segment_index is not None else parsed_segment_index
        if resolved_instance_id is None:
            resolved_instance_id = 0
        
        task_key = self.segment_to_task_key.get(task_id)
        if not task_key:
            task_key = self._make_task_key(resolved_root_task_id, resolved_instance_id)
            self.segment_to_task_key[task_id] = task_key
        self.segment_to_task_key.setdefault(resolved_root_task_id, task_key)
        
        info = self.task_info.get(task_key)
        if info is None:
            info = {
                "root_task_id": resolved_root_task_id,
                "task_name": task_name,
                "priority": TaskPriority.NORMAL,
                "ready_time": start_time,
                "instance_id": resolved_instance_id,
                "segment_ids": set([task_id]),
                "resource_ids": {resource_id},
                "first_segment_index": resolved_segment_index,
            }
            self.task_info[task_key] = info
        else:
            info.setdefault("segment_ids", set()).add(task_id)
            info.setdefault("resource_ids", set()).add(resource_id)
            info["ready_time"] = min(info.get("ready_time", start_time), start_time)
            if task_name:
                info["task_name"] = task_name
            if resolved_segment_index is not None:
                current = info.get("first_segment_index")
                if current is None or resolved_segment_index < current:
                    info["first_segment_index"] = resolved_segment_index
        
        info.setdefault("priority", TaskPriority.NORMAL)
        
        if task_name and not info.get("task_name"):
            info["task_name"] = task_name
        if original_task_id and not info.get("root_task_id"):
            info["root_task_id"] = original_task_id
        if "root_task_id" not in info:
            info["root_task_id"] = resolved_root_task_id
        if "instance_id" not in info:
            info["instance_id"] = resolved_instance_id
        priority = info.get("priority", TaskPriority.NORMAL)
        
        resource_type = self._get_resource_type(resource_id)
        
        # 记录开始事件
        self.events.append(TraceEvent(
            timestamp=start_time,
            event_type=EventType.TASK_START,
            task_id=task_id,
            resource_id=resource_id,
            resource_type=resource_type,
            priority=priority,
            segment_id=segment_id,
            bandwidth=bandwidth,
            duration=end_time - start_time,
            metadata={
                "root_task_id": resolved_root_task_id,
                "instance_id": resolved_instance_id,
                "task_name": info.get("task_name"),
                "segment_index": resolved_segment_index
            }
        ))
        
        # 记录结束事件
        self.events.append(TraceEvent(
            timestamp=end_time,
            event_type=EventType.TASK_END,
            task_id=task_id,
            resource_id=resource_id,
            resource_type=resource_type,
            priority=priority,
            segment_id=segment_id,
            metadata={
                "root_task_id": resolved_root_task_id,
                "instance_id": resolved_instance_id,
                "segment_index": resolved_segment_index
            }
        ))
        
        # 记录执行
        execution = TaskExecution(
            task_id=task_id,
            root_task_id=resolved_root_task_id,
            resource_id=resource_id,
            resource_type=resource_type,
            priority=priority,
            start_time=start_time,
            end_time=end_time,
            bandwidth=bandwidth,
            task_name=info.get("task_name"),
            instance_id=resolved_instance_id,
            segment_id=segment_id,
            segment_index=resolved_segment_index
        )
        self.executions.append(execution)
        
        # 更新资源利用率
        self.resource_busy_time[resource_id] += (end_time - start_time)
        self.resource_last_busy[resource_id] = end_time
        
        # 更新时间范围 - 确保值被正确设置
        if self.start_time is None:
            self.start_time = start_time
        else:
            self.start_time = min(self.start_time, start_time)
            
        if self.end_time is None:
            self.end_time = end_time
        else:
            self.end_time = max(self.end_time, end_time)
    
    def _parse_task_identifier(self, identifier: Optional[str]) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """解析任务标识符，返回 (root_task_id, instance_id, segment_index)"""
        if not identifier:
            return None, None, None
        
        root_task_id: Optional[str] = identifier
        instance_id: Optional[int] = None
        segment_index: Optional[int] = None
        
        prefix = identifier
        if "_seg" in identifier:
            prefix, seg_suffix = identifier.split("_seg", 1)
            digits = ""
            for ch in seg_suffix:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                segment_index = int(digits)
        if "#" in prefix:
            root_part, instance_suffix = prefix.split("#", 1)
            digits = ""
            for ch in instance_suffix:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                instance_id = int(digits)
            root_task_id = root_part
        else:
            root_task_id = prefix
        
        return root_task_id, instance_id, segment_index
    
    def _make_task_key(self, root_task_id: str, instance_id: Optional[int]) -> str:
        """根据根任务ID和实例ID生成唯一键"""
        normalized_instance = instance_id if instance_id is not None else 0
        return f"{root_task_id}#{normalized_instance}"
    
    def _resolve_root_task_id(self, task_id: str) -> Optional[str]:
        task_key = self.segment_to_task_key.get(task_id)
        if task_key:
            info = self.task_info.get(task_key)
            if info:
                return info.get("root_task_id")
        root, _, _ = self._parse_task_identifier(task_id)
        return root
    
    def _get_resource_type(self, resource_id: str) -> ResourceType:
        """获取资源类型"""
        queue = self.queue_manager.get_queue(resource_id)
        return queue.resource_type if queue else ResourceType.NPU
    
    def get_timeline(self) -> Dict[str, List[TaskExecution]]:
        """获取按资源分组的执行时间线"""
        timeline = defaultdict(list)
        for execution in self.executions:
            timeline[execution.resource_id].append(execution)
        
        # 按时间排序
        for resource_id in timeline:
            timeline[resource_id].sort(key=lambda x: x.start_time)
        
        return dict(timeline)
    
    def get_task_timeline(self, task_id: str) -> List[TaskExecution]:
        """获取特定任务的执行时间线（支持根任务ID或段ID）"""
        return [
            e for e in self.executions
            if e.root_task_id == task_id or e.task_id == task_id
        ]
    
    def get_task_groups(self) -> Dict[str, List[TaskExecution]]:
        """按任务分组获取执行记录"""
        task_groups: Dict[str, List[TaskExecution]] = defaultdict(list)
        for execution in self.executions:
            task_key = self.segment_to_task_key.get(execution.task_id)
            if not task_key:
                task_key = self._make_task_key(
                    execution.root_task_id or execution.task_id,
                    execution.instance_id
                )
                self.segment_to_task_key[execution.task_id] = task_key
            task_groups[task_key].append(execution)
        
        for task_key in task_groups:
            task_groups[task_key].sort(key=lambda x: x.start_time)
        return dict(task_groups)
    
    def get_task_latency_summary(self) -> Dict[str, Dict[str, Any]]:
        """汇总每个任务的首段、尾段与时延信息"""
        summaries: Dict[str, Dict[str, Any]] = {}
        task_groups = self.get_task_groups()
        
        for task_key, executions in task_groups.items():
            if not executions:
                continue
            
            info = self.task_info.get(task_key, {})
            root_task_id = info.get("root_task_id") or executions[0].root_task_id
            task_name = info.get("task_name") or executions[0].task_name
            priority = info.get("priority", executions[0].priority)
            instance_id = info.get("instance_id", executions[0].instance_id)
            
            sorted_exec = executions
            first_start = sorted_exec[0].start_time
            last_end = max(exec.end_time for exec in executions)
            latency = max(last_end - first_start, 0.0)
            
            display_name = root_task_id
            if task_name:
                display_name = f"{root_task_id}.{task_name}"
            
            segments = []
            for index, exec in enumerate(sorted_exec):
                segments.append({
                    "start": exec.start_time,
                    "end": exec.end_time,
                    "duration": exec.duration,
                    "resource_id": exec.resource_id,
                    "resource_type": exec.resource_type.value,
                    "segment_id": exec.segment_id,
                    "segment_index": exec.segment_index,
                    "bandwidth": exec.bandwidth,
                    "priority": exec.priority.name,
                    "is_first_segment": index == 0
                })
            
            gaps = []
            for prev, nxt in zip(sorted_exec, sorted_exec[1:]):
                if nxt.start_time > prev.end_time:
                    gaps.append({
                        "start": prev.end_time,
                        "end": nxt.start_time,
                        "duration": nxt.start_time - prev.end_time
                    })
            
            first_segment_exec = sorted_exec[0]
            ready_time = info.get("ready_time")
            wait_time = None
            if ready_time is not None and first_start is not None:
                wait_time = max(first_start - ready_time, 0.0)

            execution_latency = latency
            total_latency = latency
            if ready_time is not None:
                total_latency = max(last_end - ready_time, 0.0)

            summaries[task_key] = {
                "task_key": task_key,
                "task_id": root_task_id,
                "task_name": task_name,
                "display_name": display_name,
                "priority": priority,
                "instance_id": instance_id,
                "first_start": first_start,
                "last_end": last_end,
                "latency": total_latency,
                "total_latency": total_latency,
                "execution_latency": execution_latency,
                "wait_time": wait_time,
                "segment_count": len(sorted_exec),
                "segments": segments,
                "gaps": gaps,
                "ready_time": ready_time,
                "jitter_ms": info.get("jitter_ms"),
                "first_resource_id": first_segment_exec.resource_id,
                "first_segment_id": first_segment_exec.segment_id,
            }
        
        return summaries

    def to_snapshot(self) -> Dict[str, Any]:
        """序列化当前追踪器状态用于复现可视化"""
        resource_specs = []
        for res_id, queue in self.queue_manager.resource_queues.items():
            resource_specs.append({
                "resource_id": res_id,
                "resource_type": queue.resource_type.name,
                "bandwidth": queue.bandwidth
            })

        def _serialize_event(event: TraceEvent) -> Dict[str, Any]:
            return {
                "timestamp": event.timestamp,
                "event_type": event.event_type.value,
                "task_id": event.task_id,
                "resource_id": event.resource_id,
                "resource_type": event.resource_type.name,
                "priority": event.priority.name,
                "segment_id": event.segment_id,
                "bandwidth": event.bandwidth,
                "duration": event.duration,
                "metadata": copy.deepcopy(event.metadata)
            }

        def _serialize_execution(execution: TaskExecution) -> Dict[str, Any]:
            return {
                "task_id": execution.task_id,
                "root_task_id": execution.root_task_id,
                "resource_id": execution.resource_id,
                "resource_type": execution.resource_type.name,
                "priority": execution.priority.name,
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "bandwidth": execution.bandwidth,
                "task_name": execution.task_name,
                "instance_id": execution.instance_id,
                "segment_id": execution.segment_id,
                "segment_index": execution.segment_index
            }

        def _serialize_task_info(info: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in info.items():
                if isinstance(value, set):
                    result[key] = sorted(value)
                elif isinstance(value, TaskPriority):
                    result[key] = {"__enum__": value.name}
                else:
                    result[key] = copy.deepcopy(value)
            return result

        snapshot = {
            "resources": resource_specs,
            "events": [_serialize_event(ev) for ev in self.events],
            "executions": [_serialize_execution(ex) for ex in self.executions],
            "task_info": {k: _serialize_task_info(v) for k, v in self.task_info.items()},
            "segment_to_task_key": dict(self.segment_to_task_key),
            "priority_overrides": {
                k: v.name for k, v in self.priority_overrides.items()
            },
            "resource_busy_time": dict(self.resource_busy_time),
            "resource_last_busy": dict(self.resource_last_busy),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }
        return snapshot

    @classmethod
    def from_snapshot(cls, snapshot: Dict[str, Any]) -> 'ScheduleTracer':
        """根据序列化快照还原追踪器"""
        queue_manager = ResourceQueueManager()
        for spec in snapshot.get("resources", []):
            resource_type = ResourceType[spec["resource_type"]]
            queue_manager.add_resource(spec["resource_id"], resource_type, spec.get("bandwidth", 0.0))

        tracer = cls(queue_manager)

        def _deserialize_event(data: Dict[str, Any]) -> TraceEvent:
            return TraceEvent(
                timestamp=data["timestamp"],
                event_type=EventType(data["event_type"]),
                task_id=data["task_id"],
                resource_id=data["resource_id"],
                resource_type=ResourceType[data["resource_type"]],
                priority=TaskPriority[data["priority"]],
                segment_id=data.get("segment_id"),
                bandwidth=data.get("bandwidth"),
                duration=data.get("duration"),
                metadata=copy.deepcopy(data.get("metadata", {}))
            )

        def _deserialize_execution(data: Dict[str, Any]) -> TaskExecution:
            return TaskExecution(
                task_id=data["task_id"],
                root_task_id=data["root_task_id"],
                resource_id=data["resource_id"],
                resource_type=ResourceType[data["resource_type"]],
                priority=TaskPriority[data["priority"]],
                start_time=data["start_time"],
                end_time=data["end_time"],
                bandwidth=data.get("bandwidth", 0.0),
                task_name=data.get("task_name"),
                instance_id=data.get("instance_id"),
                segment_id=data.get("segment_id"),
                segment_index=data.get("segment_index")
            )

        tracer.events = [_deserialize_event(ev) for ev in snapshot.get("events", [])]
        tracer.executions = [_deserialize_execution(ex) for ex in snapshot.get("executions", [])]

        def _deserialize_task_info(data: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in data.items():
                if isinstance(value, list):
                    result[key] = set(value)
                elif isinstance(value, dict) and value.get("__enum__"):
                    result[key] = TaskPriority[value["__enum__"]]
                else:
                    result[key] = copy.deepcopy(value)
            return result

        tracer.task_info = {
            key: _deserialize_task_info(val)
            for key, val in snapshot.get("task_info", {}).items()
        }
        tracer.segment_to_task_key = dict(snapshot.get("segment_to_task_key", {}))
        tracer.priority_overrides = {
            k: TaskPriority[v] for k, v in snapshot.get("priority_overrides", {}).items()
        }
        tracer.resource_busy_time = defaultdict(float, snapshot.get("resource_busy_time", {}))
        tracer.resource_last_busy = defaultdict(float, snapshot.get("resource_last_busy", {}))
        tracer.start_time = snapshot.get("start_time")
        tracer.end_time = snapshot.get("end_time")
        return tracer
    
    def get_resource_utilization(self, time_window: float = None) -> Dict[str, float]:
        """
        计算资源利用率
        
        Args:
            time_window: 时间窗口，如果为None则使用实际执行时间跨度
            
        Returns:
            资源利用率字典
        """
        if not self.executions and not self.queue_manager:
            return {}
        
        # 确定时间基准
        if time_window is None:
            # 使用实际执行时间跨度
            if self.executions:
                actual_start = min(e.start_time for e in self.executions)
                actual_end = max(e.end_time for e in self.executions)
                total_time = actual_end - actual_start
            else:
                total_time = 0
        else:
            # 使用指定的时间窗口
            total_time = time_window
        
        if total_time <= 0:
            return {}
        
        utilization = {}
        # 包括所有资源（即使没有执行任务的）
        all_resources = set()
        if self.queue_manager:
            all_resources.update(self.queue_manager.resource_queues.keys())
        all_resources.update(self.resource_busy_time.keys())
        
        for resource_id in all_resources:
            busy_time = self.resource_busy_time.get(resource_id, 0.0)
            utilization[resource_id] = (busy_time / total_time) * 100
        
        return utilization
    
    def get_statistics(self, time_window: float = None) -> Dict[str, Any]:
        """
        获取调度统计信息
        
        Args:
            time_window: 时间窗口，用于计算资源利用率
        """
        # 计算实际时间跨度
        if self.executions:
            actual_start = min(e.start_time for e in self.executions)
            actual_end = max(e.end_time for e in self.executions)
            time_span = actual_end - actual_start
        else:
            time_span = 0
            
        stats = {
            "total_tasks": len(self.task_info),
            "total_executions": len(self.executions),
            "time_span": time_span,
            "resource_utilization": self.get_resource_utilization(time_window),
            "tasks_by_priority": defaultdict(int),
            "average_wait_time": 0,
            "average_execution_time": 0
        }
        
        # 按优先级统计任务数
        for task_info in self.task_info.values():
            priority = task_info.get("priority", TaskPriority.NORMAL)
            stats["tasks_by_priority"][priority.name] += 1
        
        # 计算平均等待时间和执行时间
        wait_times = []
        exec_times = []
        
        for execution in self.executions:
            exec_times.append(execution.duration)
            
            # 等待时间 = 开始时间 - 就绪时间
            task_key = self.segment_to_task_key.get(execution.task_id)
            task_info = self.task_info.get(task_key) if task_key else None
            if task_info and "ready_time" in task_info:
                wait_time = execution.start_time - task_info["ready_time"]
                wait_times.append(wait_time)
        
        if wait_times:
            stats["average_wait_time"] = sum(wait_times) / len(wait_times)
        if exec_times:
            stats["average_execution_time"] = sum(exec_times) / len(exec_times)
        
        stats["tasks_by_priority"] = dict(stats["tasks_by_priority"])
        return stats

    def apply_priority_overrides(self, priority_map: Dict[str, TaskPriority]):
        """根据最终配置覆盖任务优先级，用于可视化颜色统一"""
        if not priority_map:
            return
        self.priority_overrides = priority_map.copy()
        
        # 更新任务信息记录
        for task_key, info in self.task_info.items():
            root_id = info.get("root_task_id")
            if not root_id:
                root_id, _, _ = self._parse_task_identifier(task_key)
            if root_id and root_id in priority_map:
                info["priority"] = priority_map[root_id]
        
        # 更新执行记录
        for execution in self.executions:
            root_id = execution.root_task_id or execution.task_id
            if root_id in priority_map:
                new_priority = priority_map[root_id]
                execution.priority = new_priority
        
        # 更新事件记录
        for event in self.events:
            root_id = self._resolve_root_task_id(event.task_id)
            if root_id and root_id in priority_map:
                event.priority = priority_map[root_id]
