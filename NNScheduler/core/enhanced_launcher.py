#!/usr/bin/env python3
"""
增强的任务发射器 - 实现智能依赖预测
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import math

from NNScheduler.core.enums import TaskPriority, ResourceType
from NNScheduler.core.task import NNTask
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.models import SubSegment


@dataclass
class TaskLaunchConfig:
    """任务发射配置"""
    task_id: str
    priority: TaskPriority
    fps_requirement: float
    dependencies: List[str] = field(default_factory=list)
    initial_offset_ms: float = 0.0
    respect_dependencies: bool = False
    has_custom_offset: bool = False
    min_interval: float = field(init=False)
    
    def __post_init__(self):
        if self.fps_requirement > 0:
            self.min_interval = 1000.0 / self.fps_requirement
        else:
            self.min_interval = float('inf')


@dataclass
class LaunchEvent:
    """发射事件"""
    time: float
    task_id: str
    instance_id: int
    
    def __lt__(self, other):
        return self.time < other.time


@dataclass
class LaunchPlan:
    """发射计划"""
    events: List[LaunchEvent] = field(default_factory=list)
    task_schedules: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add_launch(self, task_id: str, time: float, instance_id: int):
        """添加发射事件"""
        event = LaunchEvent(time, task_id, instance_id)
        self.events.append(event)
        self.task_schedules[task_id].append(time)
        
    def sort_events(self):
        """按时间排序事件"""
        self.events.sort(key=lambda e: e.time)


class EnhancedTaskLauncher:
    """增强的任务发射器 - 支持智能依赖预测"""
    
    def __init__(self, queue_manager: ResourceQueueManager, tracer: Optional[ScheduleTracer] = None):
        self.queue_manager = queue_manager
        self.tracer = tracer
        
        # 任务配置
        self.task_configs: Dict[str, TaskLaunchConfig] = {}
        self.tasks: Dict[str, NNTask] = {}
        
        # 运行时状态
        self.task_last_launch: Dict[str, float] = {}
        self.task_instance_count: Dict[str, int] = defaultdict(int)
        
        # 依赖管理
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # 任务执行时间缓存（用于预测）
        self.task_duration_cache: Dict[str, float] = {}
        # 依赖完成时间缓存（用于递归估算加速）
        self._dep_completion_cache: Dict[Tuple[str, int], float] = {}
        # 任务发射时间缓存（避免重复递归计算）
        self._launch_time_cache: Dict[Tuple[str, int], float] = {}

    def register_task(self, task: NNTask):
        """注册任务"""
        config = TaskLaunchConfig(
            task_id=task.task_id,
            priority=task.priority,
            fps_requirement=task.fps_requirement,
            dependencies=sorted(task.dependencies),
            initial_offset_ms=max(0.0, getattr(task, "launch_offset_ms", 0.0) or 0.0),
            respect_dependencies=bool(getattr(task, "launch_respect_dependencies", False)),
            has_custom_offset=bool(getattr(task, "launch_offset_configured", False))
        )
        
        self.task_configs[task.task_id] = config
        self.tasks[task.task_id] = task
        
        # 构建依赖图
        for dep_id in task.dependencies:
            self.dependency_graph[dep_id].add(task.task_id)
            
        # 预计算任务执行时间
        self._cache_task_duration(task)
            
    def _cache_task_duration(self, task: NNTask):
        """缓存任务的估计执行时间

        使用 queue_manager 中的实际资源带宽（若缺失则使用默认）估算每段时长，
        并叠加 10% 裕量，作为依赖完成时间估算的经验执行时长。
        """
        total_duration = 0.0

        # 获取实际带宽映射（按资源类型挑选最高带宽的实例）
        bw_map = self._get_actual_bandwidth_map()

        # 计算所有段的执行时间
        for segment in task.segments:
            bandwidth = bw_map.get(segment.resource_type, 40.0)
            duration = segment.get_duration(bandwidth)
            total_duration += duration

        # 添加一些余量（10%）
        self.task_duration_cache[task.task_id] = total_duration * 1.1

    def _get_actual_bandwidth_map(self) -> Dict[ResourceType, float]:
        """从资源队列中推断各资源类型的代表性带宽

        若同类型有多实例，取带宽最高者；若缺失则使用默认值兜底。
        """
        bandwidth_map: Dict[ResourceType, float] = {}

        # 从资源队列管理器中获取每种资源类型的最高带宽
        for _, queue in self.queue_manager.resource_queues.items():
            rtype = queue.resource_type
            if rtype not in bandwidth_map or queue.bandwidth > bandwidth_map[rtype]:
                bandwidth_map[rtype] = queue.bandwidth

        # 默认兜底值（与 TaskLauncher 保持一致的量级）
        default_bandwidth = {
            ResourceType.NPU: 60.0,
            ResourceType.DSP: 40.0,
            ResourceType.ISP: 50.0,
            ResourceType.CPU: 35.0,
            ResourceType.GPU: 70.0,
            ResourceType.VPU: 45.0,
            ResourceType.FPGA: 50.0,
        }

        for rtype, default_bw in default_bandwidth.items():
            if rtype not in bandwidth_map:
                bandwidth_map[rtype] = default_bw

        return bandwidth_map
        
    def create_launch_plan(self, time_window: float, strategy: str = "eager") -> LaunchPlan:
        """创建发射计划 - 使用智能依赖预测"""
        # 创建计划前清空依赖完成时间缓存，避免跨次调用污染
        self._dep_completion_cache.clear()
        self._launch_time_cache.clear()
        if strategy == "eager":
            return self._create_smart_eager_plan(time_window)
        elif strategy == "lazy":
            return self._create_lazy_plan(time_window)
        elif strategy == "sync":
            try:
                return self._create_sync_plan(time_window)
            except Exception as exc:
                print(f"[WARN] Sync launch计划失败，回退到balanced策略: {exc}")
                return self._create_balanced_plan(time_window)
        elif strategy == "fixed":
            try:
                return self._create_fixed_plan(time_window)
            except Exception as exc:
                print(f"[WARN] Fixed launch计划失败，回退到balanced策略: {exc}")
                return self._create_balanced_plan(time_window)
        else:
            return self._create_balanced_plan(time_window)
            
    def _create_smart_eager_plan(self, time_window: float) -> LaunchPlan:
        """创建智能的急切发射计划 - 正确处理依赖关系"""
        plan = LaunchPlan()
        
        # 对任务进行拓扑排序（考虑依赖关系）
        sorted_tasks = self._topological_sort_tasks()
        
        # 为每个任务规划发射时间
        for task_id in sorted_tasks:
            config = self.task_configs[task_id]

            if not math.isfinite(config.min_interval) or config.min_interval <= 0:
                continue

            if config.has_custom_offset:
                available_window = time_window - config.initial_offset_ms
                if available_window <= 0:
                    continue
                max_instances = int(available_window / config.min_interval) + 1
            elif config.min_interval >= time_window:
                max_instances = 1
            else:
                max_instances = int(time_window / config.min_interval) + 1

            # 为每个实例找到合适的发射时间
            for instance in range(max_instances):
                if config.has_custom_offset:
                    base_time = config.initial_offset_ms + instance * config.min_interval
                else:
                    base_time = instance * config.min_interval

                if base_time >= time_window:
                    break

                if config.has_custom_offset and not config.respect_dependencies:
                    launch_time = base_time
                else:
                    launch_time = self._calculate_launch_time_with_dependencies(
                        task_id, instance, base_time, time_window
                    )

                if launch_time < time_window:
                    plan.add_launch(task_id, launch_time, instance)
                    
        plan.sort_events()
        return plan
        
    def _topological_sort_tasks(self) -> List[str]:
        """对任务进行拓扑排序

        目的：保证依赖的提供者先于消费者出现在顺序中。
        说明：此排序用于 Balanced/Lazy/Eager 组内的相对顺序基线；
             Balanced 最终仍会在实例层面通过依赖感知发射时间进行精调。
        """
        # 计算入度
        in_degree = {task_id: 0 for task_id in self.task_configs}
        
        for task_id, config in self.task_configs.items():
            for dep in config.dependencies:
                if dep in self.task_configs:
                    in_degree[task_id] += 1
                    
        # 拓扑排序
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        sorted_tasks = []
        
        while queue:
            # 如果同一轮有多个入度为0的候选，按优先级进行择序
            # 注：TaskPriority.CRITICAL 的 value 更小（优先级更高）
            queue.sort(key=lambda t: -self.task_configs[t].priority.value)
            task_id = queue.pop(0)
            sorted_tasks.append(task_id)
            
            # 更新依赖此任务的其他任务
            for dependent in self.dependency_graph[task_id]:
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
                        
        return sorted_tasks
        
    def _calculate_launch_time_with_dependencies(
        self, task_id: str, instance: int, base_time: float, time_window: float
    ) -> float:
        """计算考虑依赖关系的发射时间（支持帧率感知的依赖映射）

        步骤：
        1) 以组内偏移+实例间隔得到 base_time；
        2) 对每个依赖：
           - 将当前实例映射到依赖任务的实例（按 FPS 比例向下取整）；
           - 递归估算该依赖实例的完成时刻（其自身也会考虑再往前的依赖）；
           - 将发射时间推迟到依赖完成之后（+1ms 安全裕量）；
        3) 约束不早于自身的理论最小时间（instance * min_interval）。
        """
        config = self.task_configs[task_id]
        launch_time = base_time
        
        # 检查每个依赖
        for dep_task_id in config.dependencies:
            if dep_task_id not in self.task_configs:
                continue
                
            # 使用帧率感知的依赖实例映射
            dep_instance = self._get_dependency_instance(task_id, instance, dep_task_id)
            
            # 估算依赖任务的完成时间（递归计算依赖的发射时间 + 经验执行时长）
            dep_completion = self._get_dependency_completion(dep_task_id, dep_instance)
            
            # 确保在依赖完成后发射（留1ms余量）
            launch_time = max(launch_time, dep_completion + 1.0)
            
        # 确保不违反最小间隔（帧率约束）
        if instance > 0:
            min_time = instance * config.min_interval
            launch_time = max(launch_time, min_time)

        cache_key = (task_id, instance)
        self._launch_time_cache[cache_key] = launch_time
        # 发射时间更新后，移除旧的完成时间缓存，避免过期
        if cache_key in self._dep_completion_cache:
            self._dep_completion_cache.pop(cache_key, None)
            
        return launch_time

    def _create_sync_plan(self, time_window: float) -> LaunchPlan:
        """创建同步交替的发射计划，确保任务按ISP阶段顺序轮流发射"""
        plan = LaunchPlan()
        if not self.tasks:
            return plan

        ordered_tasks = sorted(
            self.tasks.values(),
            key=lambda t: (self.task_configs[t.task_id].priority.value, t.task_id)
        )

        bw_map = self._get_actual_bandwidth_map()
        isp_offsets: Dict[str, float] = {}
        isp_duration_map: Dict[str, float] = {}

        offset_acc = 0.0
        for task in ordered_tasks:
            isp_segment = next((seg for seg in task.segments if seg.resource_type == ResourceType.ISP), None)
            if isp_segment is None:
                raise ValueError(f"Task {task.task_id} 缺少ISP段，无法使用sync策略")
            isp_duration = isp_segment.get_duration(bw_map[ResourceType.ISP])
            if isp_duration <= 0:
                raise ValueError(f"Task {task.task_id} 的ISP段时长无效: {isp_duration}")

            isp_duration_map[task.task_id] = isp_duration
            isp_offsets[task.task_id] = offset_acc
            offset_acc += isp_duration

        cycle_length = offset_acc
        if cycle_length <= 0:
            raise ValueError("Sync策略计算得到的cycle_length无效")

        instance_counts: Dict[str, int] = defaultdict(int)
        next_allowed: Dict[str, float] = {task_id: 0.0 for task_id in self.tasks}
        cycle_start = 0.0
        epsilon = 1e-6

        while cycle_start < time_window - epsilon:
            for task in ordered_tasks:
                task_id = task.task_id
                launch_time = cycle_start + isp_offsets[task_id]
                if launch_time >= time_window:
                    continue

                min_interval = self.task_configs[task_id].min_interval
                if launch_time + epsilon < next_allowed[task_id]:
                    launch_time = next_allowed[task_id]

                plan.add_launch(task_id, launch_time, instance_counts[task_id])
                instance_counts[task_id] += 1
                next_allowed[task_id] = launch_time + min_interval

            cycle_start += cycle_length

            if cycle_length <= epsilon:
                break

        plan.sort_events()
        return plan

    def _create_fixed_plan(self, time_window: float) -> LaunchPlan:
        """创建固定相位的发射计划"""
        plan = LaunchPlan()
        if not self.tasks or time_window <= 0:
            return plan

        epsilon = 1e-6
        ordered_tasks = sorted(
            self.task_configs.keys(),
            key=lambda tid: (self.task_configs[tid].priority.value, tid)
        )

        for task_id in ordered_tasks:
            config = self.task_configs[task_id]

            if config.fps_requirement <= 0:
                continue

            offset = max(0.0, config.initial_offset_ms)
            if offset >= time_window:
                continue

            if config.min_interval == float("inf"):
                continue

            if config.min_interval <= epsilon:
                continue

            instance = 0
            base_time = offset

            while base_time < time_window - epsilon:
                launch_time = base_time

                if config.respect_dependencies:
                    launch_time = self._calculate_launch_time_with_dependencies(
                        task_id,
                        instance,
                        base_time,
                        time_window
                    )

                if launch_time >= time_window - epsilon:
                    break

                plan.add_launch(task_id, launch_time, instance)

                instance += 1
                base_time = offset + instance * config.min_interval

        plan.sort_events()
        return plan
    
    def _get_dependency_instance(self, task_id: str, instance_id: int, dep_id: str) -> int:
        """获取依赖任务的实例号（考虑帧率差异）

        若依赖任务 FPS 更低：按比例映射到较低频率的对应实例（向下取整），
        以模拟“低频任务每次完成可供若干个高频实例使用”的关系。
        """
        config = self.task_configs[task_id]
        dep_config = self.task_configs.get(dep_id)
        
        if not dep_config:
            return instance_id
        
        # 如果依赖任务的帧率较低，需要映射到合适的实例
        if dep_config.fps_requirement < config.fps_requirement:
            # 计算帧率比例
            fps_ratio = config.fps_requirement / dep_config.fps_requirement
            # 映射到依赖任务的实例号（向下取整）
            dep_instance = int(instance_id / fps_ratio)
        else:
            # 依赖任务帧率相同或更高，使用相同的实例号
            dep_instance = instance_id
        
        return dep_instance

    def _infer_base_launch_time(self, task_id: str, instance_id: int) -> float:
        """根据任务配置推测实例的基础发射时间"""
        config = self.task_configs.get(task_id)
        if not config or not math.isfinite(config.min_interval) or config.min_interval <= 0:
            return 0.0

        base_time = instance_id * config.min_interval
        if config.has_custom_offset:
            base_time += config.initial_offset_ms
        return base_time

    def _get_dependency_completion(self, task_id: str, instance_id: int) -> float:
        """获取依赖任务实例的完成时间，带缓存"""
        cache_key = (task_id, instance_id)

        cached_completion = self._dep_completion_cache.get(cache_key)
        if cached_completion is not None:
            return cached_completion

        if task_id not in self.task_configs:
            return 0.0

        launch_time = self._launch_time_cache.get(cache_key)
        if launch_time is None:
            base_time = self._infer_base_launch_time(task_id, instance_id)
            launch_time = self._calculate_launch_time_with_dependencies(
                task_id,
                instance_id,
                base_time,
                float('inf')
            )

        execution_time = self.task_duration_cache.get(task_id, 10.0)
        completion = launch_time + execution_time
        self._dep_completion_cache[cache_key] = completion
        self._launch_time_cache[cache_key] = launch_time
        return completion
        
    def _create_lazy_plan(self, time_window: float) -> LaunchPlan:
        """创建延迟发射计划"""
        plan = LaunchPlan()
        
        for task_id, config in self.task_configs.items():
            task = self.tasks[task_id]
            
            # 估算任务执行时间
            estimated_duration = self.task_duration_cache.get(task_id, 10.0)
            
            if config.min_interval >= time_window:
                # 只发射一次，尽量晚
                launch_time = max(0, time_window - estimated_duration - 10)
                
                # 考虑依赖关系
                launch_time = self._calculate_launch_time_with_dependencies(
                    task_id, 0, launch_time, time_window
                )
                
                if launch_time < time_window:
                    plan.add_launch(task_id, launch_time, 0)
            else:
                # 周期性发射
                max_instances = int(time_window / config.min_interval)
                
                for instance in range(max_instances):
                    # 从后往前计算
                    base_time = time_window - (max_instances - instance) * config.min_interval
                    
                    launch_time = self._calculate_launch_time_with_dependencies(
                        task_id, instance, base_time, time_window
                    )
                    
                    if 0 <= launch_time < time_window:
                        plan.add_launch(task_id, launch_time, instance)
                        
        plan.sort_events()
        return plan
        
    def _create_balanced_plan(self, time_window: float) -> LaunchPlan:
        """创建均衡发射计划

        设计要点：
        - 先按优先级分组（CRITICAL→HIGH→NORMAL→LOW）；
        - 组内按拓扑序排列（供应者在前，消费者在后）；
        - 在一个组内，用固定 5ms 的交错偏移（offset_step）分散不同任务的基础发射时刻；
        - 对每个实例的基础时刻再做“依赖就绪校正”（见 _calculate_launch_time_with_dependencies）。
        """
        plan = LaunchPlan()
        
        # 按优先级分组
        priority_groups = defaultdict(list)
        for task_id, config in self.task_configs.items():
            priority_groups[config.priority].append(task_id)
            
        # 为每个优先级组创建交错的发射时间
        for priority in TaskPriority:
            tasks = priority_groups[priority]
            if not tasks:
                continue
                
            # 对组内任务进行拓扑排序（仅保留同组成员的相对顺序）
            sorted_group = [t for t in self._topological_sort_tasks() if t in tasks]
            
            valid_intervals = [
                self.task_configs[t].min_interval
                for t in sorted_group
                if math.isfinite(self.task_configs[t].min_interval) and self.task_configs[t].min_interval > 0
            ]
            if valid_intervals:
                reference_interval = min(valid_intervals)
                offset_step = max(1.0, min(reference_interval * 0.5, 10.0))
            else:
                offset_step = 5.0  # 兜底
            
            for i, task_id in enumerate(sorted_group):
                config = self.task_configs[task_id]
                if config.has_custom_offset:
                    base_offset = config.initial_offset_ms
                else:
                    base_offset = i * offset_step

                if not math.isfinite(config.min_interval) or config.min_interval <= 0:
                    continue
                if base_offset >= time_window:
                    continue

                # 计算实例数
                available_window = time_window - base_offset
                if available_window <= 0:
                    continue

                if config.min_interval >= available_window:
                    max_instances = 1
                else:
                    max_instances = int(available_window / config.min_interval) + 1

                # 为每个实例规划发射时间（基础时间 + 依赖就绪校正）
                for instance in range(max_instances):
                    base_time = base_offset + instance * config.min_interval

                    if base_time >= time_window:
                        break

                    if config.has_custom_offset and not config.respect_dependencies:
                        launch_time = base_time
                    else:
                        launch_time = self._calculate_launch_time_with_dependencies(
                            task_id, instance, base_time, time_window
                        )

                    if launch_time < time_window:
                        plan.add_launch(task_id, launch_time, instance)
                        
        plan.sort_events()
        return plan


# 测试代码
if __name__ == "__main__":
    from scenario.real_task import create_real_tasks
    
    # 创建资源管理器
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 创建增强的发射器
    launcher = EnhancedTaskLauncher(queue_manager)
    
    # 注册任务
    tasks = create_real_tasks()
    for task in tasks:
        launcher.register_task(task)
        
    # 创建发射计划
    plan = launcher.create_launch_plan(200.0, "eager")
    
    # 打印计划
    print("发射计划：")
    print(f"任务ID\t实例\t发射时间(ms)")
    print("-" * 30)
    
    for event in plan.events:
        print(f"{event.task_id}\t{event.instance_id}\t{event.time:.1f}")
        
    # 统计各任务的发射次数
    task_counts = defaultdict(int)
    for event in plan.events:
        task_counts[event.task_id] += 1
        
    print("\n任务发射统计：")
    for task_id in sorted(task_counts.keys()):
        task = next(t for t in tasks if t.task_id == task_id)
        expected = int(200.0 / (1000.0 / task.fps_requirement))
        actual = task_counts[task_id]
        status = "[OK]" if actual >= expected else "[FAIL]"
        print(f"{task_id}: {actual}/{expected} (FPS={task.fps_requirement}) {status}")
