#!/usr/bin/env python3
"""
事件驱动执行器（验证版）

核心思想：
- 不依赖绝对时间计划，而由“资源完成/空闲事件”和“实例到达/依赖完成事件”驱动调度。
- 在不可抢占的FIFO资源上，通过小幅的 slack 保护即将到来的更高优先级段，尽量避免被低优长段压住。

注意：这是验证版，估时与ETA预测做了保守简化，后续可按需要增强。
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import heapq

from NNScheduler.core.enums import TaskPriority, ResourceType
from NNScheduler.core.models import SubSegment
from NNScheduler.core.task import NNTask
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import LaunchEvent  # 复用用于评估的事件结构


@dataclass
class ReadyJob:
    task_id: str
    instance_id: int
    segment_index: int
    segment: SubSegment
    priority: TaskPriority


class EventDrivenExecutor:
    """事件驱动执行器（验证版）"""

    def __init__(self, queue_manager: ResourceQueueManager, tracer: ScheduleTracer, tasks: Dict[str, NNTask]):
        self.queue_manager = queue_manager
        self.tracer = tracer
        self.tasks = tasks

        # 运行状态
        self.current_time: float = 0.0

        # (resource_type -> ready list)
        self.ready_pools: Dict[ResourceType, List[ReadyJob]] = {
            ResourceType.NPU: [],
            ResourceType.DSP: [],
        }

        # 事件堆： (time, type, payload)
        # type: 'instance', 'segment_ready', 'completion'
        self.events: List[Tuple[float, str, Any]] = []

        # 任务实例完成时间：用于依赖判断
        self.instance_completion: Dict[Tuple[str, int], float] = {}

        # 实例的执行进度： (task_id, instance_id) -> next_index
        self.instance_progress: Dict[Tuple[str, int], int] = {}

        # 任务的下一次实例到达时间
        self.task_next_arrival: Dict[str, float] = {}

        # 记录发射事件（用于评估等待/延迟）
        self.launch_events: List[LaunchEvent] = []

    def _prepare_segments(self, task: NNTask, segment_mode: bool) -> List[SubSegment]:
        """生成子段列表（与传统执行器一致的行为）"""
        sub_segments = task.apply_segmentation() if segment_mode else []
        if sub_segments:
            return sub_segments
        else:
            # 将 ResourceSegment 转换为 SubSegment
            ret = []
            for seg in task.segments:
                ret.append(SubSegment(
                    sub_id=seg.segment_id,
                    resource_type=seg.resource_type,
                    duration_table=seg.duration_table,
                    cut_overhead=0.0,
                    original_segment_id=seg.segment_id
                ))
            return ret

    def _min_interval(self, task: NNTask) -> float:
        return 1000.0 / task.fps_requirement if task.fps_requirement > 0 else float('inf')

    def _get_dep_instance(self, task: NNTask, instance_id: int, dep_id: str) -> int:
        """与 launcher 中一致的FPS感知实例映射"""
        dep_task = self.tasks.get(dep_id)
        if not dep_task:
            return instance_id
        if dep_task.fps_requirement < task.fps_requirement and dep_task.fps_requirement > 0:
            ratio = task.fps_requirement / dep_task.fps_requirement
            return int(instance_id / ratio)
        return instance_id

    def _dependencies_satisfied(self, task: NNTask, instance_id: int) -> bool:
        for dep in task.dependencies:
            dep_inst = self._get_dep_instance(task, instance_id, dep)
            if (dep, dep_inst) not in self.instance_completion:
                return False
            if self.instance_completion[(dep, dep_inst)] > self.current_time:
                return False
        return True

    def _estimate_duration(self, seg: SubSegment, resource_id: str) -> float:
        queue = self.queue_manager.get_queue(resource_id)
        bw = queue.bandwidth if queue else 40.0
        return seg.get_duration(bw)

    def _pick_resource(self, rtype: ResourceType) -> Optional[str]:
        # 选择最早可用的资源ID
        best = None
        best_t = float('inf')
        for rid, q in self.queue_manager.resource_queues.items():
            if q.resource_type == rtype:
                t = q.get_next_available_time()
                if t < best_t:
                    best_t = t
                    best = rid
        return best

    def _eta_higher_priority(self, rtype: ResourceType, prio: TaskPriority,
                              slack_map: Dict[ResourceType, float]) -> Optional[float]:
        """预测更高优先级段的最早到达时间（粗略）"""
        # 就绪池中若有更高优先级则视为立即到达
        for hp in TaskPriority:
            if hp.value < prio.value:
                for job in self.ready_pools.get(rtype, []):
                    if job.priority == hp:
                        return self.current_time  # 立即可用
        # 任务下一次实例到达（仅考虑无依赖或依赖已满足的简化）
        eta = None
        for tid, task in self.tasks.items():
            if task.priority.value < prio.value:
                # 仅考虑首段资源匹配的任务
                segs = self._prepare_segments(task, segment_mode=True)
                if not segs:
                    continue
                if segs[0].resource_type != rtype:
                    continue
                t_arr = self.task_next_arrival.get(tid, None)
                if t_arr is None:
                    continue
                # 依赖检查（粗略）：若依赖不满足，忽略
                if task.dependencies:
                    continue
                eta = t_arr if eta is None else min(eta, t_arr)
        return eta

    def execute(self, max_time: float, segment_mode: bool = True,
                launch_strategy: str = "balanced",
                slack_ms: Optional[Dict[str, float]] = None,
                max_idle_ms: float = 0.5,
                max_queue_depth: int = 1) -> Dict[str, Any]:
        """执行事件驱动调度"""
        slack_ms = slack_ms or {"NPU": 1.0, "DSP": 0.8}
        slack_map = {
            ResourceType.NPU: float(slack_ms.get("NPU", 1.0)),
            ResourceType.DSP: float(slack_ms.get("DSP", 0.8)),
        }

        # 初始化：为每个任务安排第一轮实例到达事件
        for tid, task in self.tasks.items():
            self.task_next_arrival[tid] = 0.0
            heapq.heappush(self.events, (0.0, 'instance', (tid, 0)))

        while self.current_time < max_time:
            # 推进到下一事件或资源完成时刻
            next_times = []
            if self.events:
                next_times.append(self.events[0][0])
            for q in self.queue_manager.resource_queues.values():
                if q.is_busy():
                    next_times.append(q.busy_until)
            if not next_times:
                break
            self.current_time = min(next_times)

            # 处理到期的事件
            while self.events and self.events[0][0] <= self.current_time:
                t, etype, payload = heapq.heappop(self.events)
                if etype == 'instance':
                    task_id, inst_id = payload
                    task = self.tasks[task_id]
                    # 依赖检查
                    if not task.dependencies or self._dependencies_satisfied(task, inst_id):
                        # 记录发射事件
                        self.launch_events.append(LaunchEvent(time=self.current_time, task_id=task_id, instance_id=inst_id))
                        # 创建实例并推送首段到就绪池
                        segs = self._prepare_segments(task, segment_mode)
                        self.instance_progress[(task_id, inst_id)] = 0
                        if segs:
                            seg0 = segs[0]
                            self.ready_pools.setdefault(seg0.resource_type, []).append(
                                ReadyJob(task_id, inst_id, 0, seg0, task.priority)
                            )
                        # 安排下一实例
                        next_t = self.task_next_arrival[task_id] + self._min_interval(task)
                        self.task_next_arrival[task_id] = next_t
                        heapq.heappush(self.events, (next_t, 'instance', (task_id, inst_id + 1)))
                    else:
                        # 依赖未满足：稍后重试（简单回退：+0.5ms）
                        heapq.heappush(self.events, (self.current_time + 0.5, 'instance', (task_id, inst_id)))
                elif etype == 'segment_ready':
                    task_id, inst_id, seg_idx, seg = payload
                    task = self.tasks[task_id]
                    self.ready_pools.setdefault(seg.resource_type, []).append(
                        ReadyJob(task_id, inst_id, seg_idx, seg, task.priority)
                    )
                elif etype == 'completion':
                    # 完成一个段，准备下一个段或结束实例
                    task_id, inst_id, seg_idx = payload
                    task = self.tasks[task_id]
                    segs = self._prepare_segments(task, segment_mode)
                    next_idx = seg_idx + 1
                    if next_idx < len(segs):
                        seg_next = segs[next_idx]
                        heapq.heappush(self.events, (self.current_time, 'segment_ready', (task_id, inst_id, next_idx, seg_next)))
                        self.instance_progress[(task_id, inst_id)] = next_idx
                    else:
                        # 实例完成
                        self.instance_completion[(task_id, inst_id)] = self.current_time

            # 对每个资源尝试分配就绪段
            for res_id, queue in self.queue_manager.resource_queues.items():
                if queue.is_busy() or self.current_time >= max_time:
                    continue
                rtype = queue.resource_type

                # 选择就绪段（按优先级）
                pool = self.ready_pools.get(rtype, [])
                if not pool:
                    continue

                # 找到最高优先级的候选
                pool.sort(key=lambda j: j.priority.value)  # CRITICAL(0) 最先
                chosen: Optional[ReadyJob] = None

                # slack gating 仅在 balanced 下生效（eager 等价于 slack=0）
                use_slack = (launch_strategy == 'balanced')
                r_slack = slack_map.get(rtype, 0.0) if use_slack else 0.0

                # 预测更高优先级的到达
                for job in pool:
                    eta_high = self._eta_higher_priority(rtype, job.priority, slack_map)
                    dur = self._estimate_duration(job.segment, res_id)
                    if use_slack and eta_high is not None and (self.current_time + dur) > (eta_high - r_slack):
                        # 尝试寻找同优更短段
                        budget = max(0.0, (eta_high - r_slack) - self.current_time)
                        same_prio = [j for j in pool if j.priority == job.priority]
                        best_fit = None
                        best_dur = None
                        for j in same_prio:
                            d = self._estimate_duration(j.segment, res_id)
                            if d <= budget and (best_dur is None or d < best_dur):
                                best_dur = d
                                best_fit = j
                        if best_fit:
                            chosen = best_fit
                            break
                        else:
                            # 放弃提交，等待更高优到达（自然由事件驱动推进）
                            chosen = None
                            break
                    else:
                        chosen = job
                        break

                if not chosen:
                    continue

                # 提交 chosen
                pool.remove(chosen)
                start = self.current_time
                dur = self._estimate_duration(chosen.segment, res_id)
                end = start + dur

                # 记录执行
                self.tracer.record_execution(
                    task_id=f"{chosen.task_id}#{chosen.instance_id}",
                    resource_id=res_id,
                    start_time=start,
                    end_time=end,
                    bandwidth=self.queue_manager.get_queue(res_id).bandwidth,
                    segment_id=chosen.segment.sub_id
                )
                # 更新资源状态
                queue.busy_until = end
                queue.current_time = start

                # 安排完成事件
                heapq.heappush(self.events, (end, 'completion', (chosen.task_id, chosen.instance_id, chosen.segment_index)))

            # 推进资源时间到 current_time
            self.queue_manager.advance_all_queues(self.current_time)

        # 返回简单统计
        return {
            'launch_events': self.launch_events,
            'completed_instances': len({(tid, iid) for (tid, iid) in self.instance_completion.keys()}),
            'current_time': self.current_time,
        }

