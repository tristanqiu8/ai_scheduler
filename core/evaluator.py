#!/usr/bin/env python3
"""
性能评估器 - 全面评估调度器的执行性能
包括FPS、延迟、资源利用率、空闲时间等关键指标
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json

from .enums import TaskPriority, ResourceType
from .task import NNTask
from .schedule_tracer import ScheduleTracer, TaskExecution
from .resource_queue import ResourceQueueManager


@dataclass
class TaskPerformanceMetrics:
    """单个任务的性能指标"""
    task_id: str
    task_name: str
    priority: TaskPriority
    fps_requirement: float
    latency_requirement: float
    
    # 执行统计
    execution_count: int = 0
    instance_count: int = 0
    
    # FPS相关
    achieved_fps: float = 0.0
    fps_satisfaction: bool = False
    fps_achievement_rate: float = 0.0  # 达成率百分比
    
    # 延迟相关
    wait_times: List[float] = field(default_factory=list)  # 发射到开始调度的时延
    latencies: List[float] = field(default_factory=list)   # 发射到完成的总延迟
    avg_wait_time: float = 0.0
    avg_latency: float = 0.0
    max_latency: float = 0.0
    latency_violations: int = 0
    latency_satisfaction_rate: float = 0.0
    
    # 执行时间统计
    execution_times: List[float] = field(default_factory=list)
    avg_execution_time: float = 0.0
    total_execution_time: float = 0.0


@dataclass
class ResourceUtilizationMetrics:
    """资源利用率指标"""
    resource_id: str
    resource_type: ResourceType
    capacity: float
    
    # 时间统计
    busy_time: float = 0.0
    idle_time: float = 0.0
    total_time: float = 0.0
    
    # 利用率
    utilization_rate: float = 0.0  # busy_time / total_time
    
    # 任务执行统计
    task_executions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    segment_executions: int = 0


@dataclass
class OverallPerformanceMetrics:
    """整体性能指标"""
    # 时间窗口
    time_window: float
    actual_execution_time: float  # 实际执行时间（最后一个任务完成时间）
    idle_time: float  # 整体运行完成到整体窗口结束的idle时间
    idle_time_ratio: float  # idle时间占比
    
    # FPS统计
    total_fps_requirement: float
    achieved_total_fps: float
    fps_satisfaction_rate: float  # 满足FPS要求的任务比例
    avg_fps_achievement_rate: float  # 平均FPS达成率
    
    # 延迟统计
    avg_wait_time: float
    avg_latency: float
    max_latency: float
    latency_violation_rate: float
    
    # 资源利用率
    avg_npu_utilization: float
    avg_dsp_utilization: float
    overall_resource_utilization: float
    resource_balance_score: float  # 资源负载均衡度(0-1)
    
    # 任务完成情况
    total_tasks: int
    completed_tasks: int
    completion_rate: float
    total_segments: int
    completed_segments: int


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self, tracer: ScheduleTracer, tasks: Dict[str, NNTask], 
                 queue_manager: ResourceQueueManager):
        self.tracer = tracer
        self.tasks = tasks
        self.queue_manager = queue_manager
        self.time_window = 0.0
        
        # 评估结果
        self.task_metrics: Dict[str, TaskPerformanceMetrics] = {}
        self.resource_metrics: Dict[str, ResourceUtilizationMetrics] = {}
        self.overall_metrics: Optional[OverallPerformanceMetrics] = None
        
    def evaluate(self, time_window: float, launch_events: List = None) -> OverallPerformanceMetrics:
        """
        执行全面的性能评估
        
        Args:
            time_window: 时间窗口
            launch_events: 发射事件列表(用于计算等待时间)
            
        Returns:
            整体性能指标
        """
        self.time_window = time_window
        
        # 1. 评估任务性能
        self._evaluate_task_performance(launch_events)
        
        # 2. 评估资源利用率
        self._evaluate_resource_utilization()
        
        # 3. 计算整体指标
        self._calculate_overall_metrics()
        
        return self.overall_metrics
        
    def _evaluate_task_performance(self, launch_events: List = None):
        """评估每个任务的性能"""
        # 初始化任务指标
        for task_id, task in self.tasks.items():
            self.task_metrics[task_id] = TaskPerformanceMetrics(
                task_id=task_id,
                task_name=task.name,
                priority=task.priority,
                fps_requirement=task.fps_requirement,
                latency_requirement=task.latency_requirement
            )
        
        # 构建发射时间映射
        launch_times = defaultdict(list)  # task_id -> [(instance_id, launch_time)]
        if launch_events:
            for event in launch_events:
                # 尝试获取时间属性（兼容不同的属性名）
                launch_time = None
                if hasattr(event, 'time'):
                    launch_time = event.time
                elif hasattr(event, 'launch_time'):
                    launch_time = event.launch_time
                
                if launch_time is not None and hasattr(event, 'task_id'):
                    instance_id = getattr(event, 'instance_id', 0)
                    launch_times[event.task_id].append((instance_id, launch_time))
        
        # 分析执行历史
        task_instances = defaultdict(lambda: defaultdict(list))  # task_id -> instance -> executions
        
        for execution in self.tracer.executions:
            # 解析任务ID和实例号
            if '#' in execution.task_id:
                base_task_id, instance_info = execution.task_id.split('#', 1)
                if '_' in instance_info:
                    instance_num = int(instance_info.split('_')[0])
                else:
                    instance_num = int(instance_info)
            else:
                base_task_id = execution.task_id
                instance_num = 0
            
            if base_task_id in self.task_metrics:
                task_instances[base_task_id][instance_num].append(execution)
        
        # 计算每个任务的指标
        for task_id, instances in task_instances.items():
            metrics = self.task_metrics[task_id]
            metrics.instance_count = len(instances)
            
            # 处理每个实例
            for instance_num, executions in instances.items():
                if not executions:
                    continue
                
                # 找到该实例的第一次和最后一次执行
                first_exec = min(executions, key=lambda e: e.start_time)
                last_exec = max(executions, key=lambda e: e.end_time)
                
                # 查找对应的发射时间
                launch_time = None
                if task_id in launch_times:
                    # 查找匹配的实例发射时间
                    for inst_id, l_time in launch_times[task_id]:
                        if inst_id == instance_num:
                            launch_time = l_time
                            break
                
                # 计算延迟
                if launch_time is not None:
                    # 等待时间：发射到首次执行
                    wait_time = first_exec.start_time - launch_time
                    metrics.wait_times.append(wait_time)
                    
                    # 总延迟：发射到完成
                    total_latency = last_exec.end_time - launch_time
                    metrics.latencies.append(total_latency)
                    
                    # 检查延迟违规
                    if total_latency > metrics.latency_requirement:
                        metrics.latency_violations += 1
                
                # 累计执行时间
                for exec in executions:
                    metrics.execution_times.append(exec.duration)
                    metrics.total_execution_time += exec.duration
                    metrics.execution_count += 1
            
            # 计算平均值
            if metrics.wait_times:
                metrics.avg_wait_time = sum(metrics.wait_times) / len(metrics.wait_times)
            
            if metrics.latencies:
                metrics.avg_latency = sum(metrics.latencies) / len(metrics.latencies)
                metrics.max_latency = max(metrics.latencies)
                metrics.latency_satisfaction_rate = 1.0 - (metrics.latency_violations / len(metrics.latencies))
            
            if metrics.execution_times:
                metrics.avg_execution_time = sum(metrics.execution_times) / len(metrics.execution_times)
            
            # 计算FPS
            if self.time_window > 0:
                metrics.achieved_fps = (metrics.instance_count * 1000.0) / self.time_window
                metrics.fps_achievement_rate = min(100.0, (metrics.achieved_fps / metrics.fps_requirement) * 100.0)
                metrics.fps_satisfaction = metrics.achieved_fps >= metrics.fps_requirement
    def _evaluate_resource_utilization(self):
        """评估资源利用率"""
        # 直接从 resource_queues 获取所有资源
        for res_id, queue in self.queue_manager.resource_queues.items():
            self.resource_metrics[res_id] = ResourceUtilizationMetrics(
                resource_id=res_id,
                resource_type=queue.resource_type,
                capacity=queue.bandwidth,
                total_time=self.time_window
            )
        
        # 分析每个资源的执行情况
        for execution in self.tracer.executions:
            res_id = execution.resource_id
            if res_id in self.resource_metrics:
                metrics = self.resource_metrics[res_id]
                
                # 计算执行时间
                duration = execution.end_time - execution.start_time
                metrics.busy_time += duration
                metrics.segment_executions += 1
                
                # 统计任务执行
                base_task_id = execution.task_id.split('#')[0] if '#' in execution.task_id else execution.task_id
                metrics.task_executions[base_task_id] += 1
        
        # 计算利用率
        for metrics in self.resource_metrics.values():
            metrics.idle_time = metrics.total_time - metrics.busy_time
            if metrics.total_time > 0:
                metrics.utilization_rate = (metrics.busy_time / metrics.total_time) * 100.0
    
    def _calculate_overall_metrics(self):
        """计算整体性能指标"""
        # 找到实际执行结束时间
        actual_end_time = 0.0
        if self.tracer.executions:
            actual_end_time = max(e.end_time for e in self.tracer.executions)
        
        # 计算空闲时间
        idle_time = max(0, self.time_window - actual_end_time)
        idle_time_ratio = (idle_time / self.time_window) * 100.0 if self.time_window > 0 else 0
        
        # FPS统计
        total_fps_req = sum(m.fps_requirement for m in self.task_metrics.values())
        achieved_fps = sum(m.achieved_fps for m in self.task_metrics.values())
        satisfied_tasks = sum(1 for m in self.task_metrics.values() if m.fps_satisfaction)
        fps_sat_rate = (satisfied_tasks / len(self.task_metrics)) * 100.0 if self.task_metrics else 0
        avg_fps_achievement = sum(m.fps_achievement_rate for m in self.task_metrics.values()) / len(self.task_metrics) if self.task_metrics else 0
        
        # 延迟统计
        all_wait_times = []
        all_latencies = []
        total_violations = 0
        total_instances = 0
        
        for m in self.task_metrics.values():
            all_wait_times.extend(m.wait_times)
            all_latencies.extend(m.latencies)
            total_violations += m.latency_violations
            total_instances += len(m.latencies)
        
        avg_wait = sum(all_wait_times) / len(all_wait_times) if all_wait_times else 0
        avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
        max_latency = max(all_latencies) if all_latencies else 0
        violation_rate = (total_violations / total_instances) * 100.0 if total_instances > 0 else 0
        
        # 资源利用率
        npu_utils = [m.utilization_rate for m in self.resource_metrics.values() 
                     if m.resource_type == ResourceType.NPU]
        dsp_utils = [m.utilization_rate for m in self.resource_metrics.values() 
                     if m.resource_type == ResourceType.DSP]
        
        avg_npu = sum(npu_utils) / len(npu_utils) if npu_utils else 0
        avg_dsp = sum(dsp_utils) / len(dsp_utils) if dsp_utils else 0
        all_utils = list(self.resource_metrics.values())
        overall_util = sum(m.utilization_rate for m in all_utils) / len(all_utils) if all_utils else 0
        
        # 资源负载均衡度（标准差越小越均衡）
        if len(all_utils) > 1:
            utils = [m.utilization_rate for m in all_utils]
            std_dev = np.std(utils)
            # 归一化到0-1，标准差越小分数越高
            balance_score = max(0, 1 - (std_dev / 50.0))  # 假设50%是最大可接受的标准差
        else:
            balance_score = 1.0
        
        # 任务完成统计
        total_tasks = len(self.task_metrics)
        completed_tasks = sum(1 for m in self.task_metrics.values() if m.instance_count > 0)
        total_segments = sum(m.execution_count for m in self.task_metrics.values())
        
        # 创建整体指标
        self.overall_metrics = OverallPerformanceMetrics(
            time_window=self.time_window,
            actual_execution_time=actual_end_time,
            idle_time=idle_time,
            idle_time_ratio=idle_time_ratio,
            total_fps_requirement=total_fps_req,
            achieved_total_fps=achieved_fps,
            fps_satisfaction_rate=fps_sat_rate,
            avg_fps_achievement_rate=avg_fps_achievement,
            avg_wait_time=avg_wait,
            avg_latency=avg_latency,
            max_latency=max_latency,
            latency_violation_rate=violation_rate,
            avg_npu_utilization=avg_npu,
            avg_dsp_utilization=avg_dsp,
            overall_resource_utilization=overall_util,
            resource_balance_score=balance_score,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            completion_rate=(completed_tasks / total_tasks) * 100.0 if total_tasks > 0 else 0,
            total_segments=total_segments,
            completed_segments=total_segments  # 假设所有开始的段都完成了
        )
    
    def print_summary_report(self):
        """打印详细的评估报告"""
        if not self.overall_metrics:
            print("No evaluation results available")
            return
        
        print("\n" + "="*80)
        print("📊 性能评估报告")
        print("="*80)
        
        # 1. 任务执行详情
        print("\n1️⃣ 任务执行详情:")
        print("-"*80)
        print(f"{'任务ID':<15} {'优先级':<8} {'FPS要求':<8} {'实际FPS':<8} {'达标':<6} "
              f"{'运行次数':<8} {'平均等待':<10} {'平均延迟':<10} {'延迟达标率':<10}")
        print("-"*80)
        
        # 按优先级排序
        sorted_tasks = sorted(self.task_metrics.values(), 
                            key=lambda m: (m.priority.value, m.task_id))
        
        for m in sorted_tasks:
            fps_ok = "✅" if m.fps_satisfaction else "❌"
            print(f"{m.task_id:<15} {m.priority.name:<8} {m.fps_requirement:<8.1f} "
                  f"{m.achieved_fps:<8.1f} {fps_ok:<6} {m.instance_count:<8} "
                  f"{m.avg_wait_time:<10.2f} {m.avg_latency:<10.2f} "
                  f"{m.latency_satisfaction_rate:<10.1%}")
        
        # 2. 资源利用率
        print("\n2️⃣ 资源利用率:")
        print("-"*80)
        print(f"{'资源ID':<15} {'类型':<8} {'利用率':<10} {'忙碌时间':<12} {'空闲时间':<12} {'执行段数':<10}")
        print("-"*80)
        
        # 按资源类型和ID排序
        sorted_resources = sorted(self.resource_metrics.values(),
                                key=lambda r: (r.resource_type.value, r.resource_id))
        
        for r in sorted_resources:
            print(f"{r.resource_id:<15} {r.resource_type.value:<8} {r.utilization_rate:<10.1f}% "
                  f"{r.busy_time:<12.1f}ms {r.idle_time:<12.1f}ms {r.segment_executions:<10}")
        
        # 3. 整体性能指标
        m = self.overall_metrics
        print("\n3️⃣ 整体性能指标:")
        print("-"*80)
        
        print(f"时间窗口: {m.time_window:.1f}ms")
        print(f"实际执行时间: {m.actual_execution_time:.1f}ms")
        print(f"🎯 空闲时间: {m.idle_time:.1f}ms ({m.idle_time_ratio:.1f}%)")
        
        print(f"\nFPS性能:")
        print(f"  - 总FPS要求: {m.total_fps_requirement:.1f}")
        print(f"  - 实际总FPS: {m.achieved_total_fps:.1f}")
        print(f"  - FPS满足率: {m.fps_satisfaction_rate:.1f}%")
        print(f"  - 平均FPS达成率: {m.avg_fps_achievement_rate:.1f}%")
        
        print(f"\n延迟性能:")
        print(f"  - 平均等待时间: {m.avg_wait_time:.2f}ms")
        print(f"  - 平均总延迟: {m.avg_latency:.2f}ms")
        print(f"  - 最大延迟: {m.max_latency:.2f}ms")
        print(f"  - 延迟违规率: {m.latency_violation_rate:.1f}%")
        
        print(f"\n资源利用:")
        print(f"  - NPU平均利用率: {m.avg_npu_utilization:.1f}%")
        print(f"  - DSP平均利用率: {m.avg_dsp_utilization:.1f}%")
        print(f"  - 整体利用率: {m.overall_resource_utilization:.1f}%")
        print(f"  - 负载均衡度: {m.resource_balance_score:.2f}")
        
        print(f"\n任务完成:")
        print(f"  - 任务完成率: {m.completion_rate:.1f}% ({m.completed_tasks}/{m.total_tasks})")
        print(f"  - 总执行段数: {m.total_segments}")
        
        print("\n" + "="*80)
    
    def export_json_report(self, filename: str):
        """导出JSON格式的评估报告"""
        report = {
            "time_window": self.time_window,
            "overall_metrics": self._serialize_overall_metrics(),
            "task_metrics": self._serialize_task_metrics(),
            "resource_metrics": self._serialize_resource_metrics()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    def _serialize_overall_metrics(self) -> Dict:
        """序列化整体指标"""
        if not self.overall_metrics:
            return {}
        
        m = self.overall_metrics
        return {
            "time_window": m.time_window,
            "actual_execution_time": m.actual_execution_time,
            "idle_time": m.idle_time,
            "idle_time_ratio": m.idle_time_ratio,
            "fps": {
                "total_requirement": m.total_fps_requirement,
                "achieved": m.achieved_total_fps,
                "satisfaction_rate": m.fps_satisfaction_rate,
                "avg_achievement_rate": m.avg_fps_achievement_rate
            },
            "latency": {
                "avg_wait_time": m.avg_wait_time,
                "avg_latency": m.avg_latency,
                "max_latency": m.max_latency,
                "violation_rate": m.latency_violation_rate
            },
            "resource_utilization": {
                "npu_avg": m.avg_npu_utilization,
                "dsp_avg": m.avg_dsp_utilization,
                "overall": m.overall_resource_utilization,
                "balance_score": m.resource_balance_score
            },
            "completion": {
                "total_tasks": m.total_tasks,
                "completed_tasks": m.completed_tasks,
                "completion_rate": m.completion_rate,
                "total_segments": m.total_segments
            }
        }
    
    def _serialize_task_metrics(self) -> List[Dict]:
        """序列化任务指标"""
        results = []
        for task_id, m in self.task_metrics.items():
            results.append({
                "task_id": m.task_id,
                "task_name": m.task_name,
                "priority": m.priority.name,
                "requirements": {
                    "fps": m.fps_requirement,
                    "latency": m.latency_requirement
                },
                "performance": {
                    "execution_count": m.execution_count,
                    "instance_count": m.instance_count,
                    "achieved_fps": m.achieved_fps,
                    "fps_satisfaction": m.fps_satisfaction,
                    "fps_achievement_rate": m.fps_achievement_rate,
                    "avg_wait_time": m.avg_wait_time,
                    "avg_latency": m.avg_latency,
                    "max_latency": m.max_latency,
                    "latency_violations": m.latency_violations,
                    "latency_satisfaction_rate": m.latency_satisfaction_rate
                }
            })
        return results
    
    def _serialize_resource_metrics(self) -> List[Dict]:
        """序列化资源指标"""
        results = []
        for res_id, m in self.resource_metrics.items():
            results.append({
                "resource_id": m.resource_id,
                "resource_type": m.resource_type.value,
                "capacity": m.capacity,
                "utilization": {
                    "busy_time": m.busy_time,
                    "idle_time": m.idle_time,
                    "total_time": m.total_time,
                    "utilization_rate": m.utilization_rate
                },
                "executions": {
                    "segment_count": m.segment_executions,
                    "task_breakdown": dict(m.task_executions)
                }
            })
        return results


# 导入numpy用于计算标准差
try:
    import numpy as np
except ImportError:
    # 如果没有numpy，提供简单的标准差计算
    def calculate_std(values):
        if not values:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    # 替换numpy.std
    class np:
        @staticmethod
        def std(values):
            return calculate_std(values)
