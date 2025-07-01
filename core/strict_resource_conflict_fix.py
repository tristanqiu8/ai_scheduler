#!/usr/bin/env python3
"""
严格的资源冲突修复
彻底解决多个任务同时使用同一资源的问题
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
from .enums import ResourceType, TaskPriority


def apply_strict_resource_conflict_fix(scheduler):
    """应用严格的资源冲突修复"""
    print("🔧 应用严格的资源冲突修复...")
    
    # 创建资源占用跟踪器
    scheduler._resource_occupancy = ResourceOccupancyTracker()
    
    # 保存原始方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def strict_conflict_free_schedule(time_window: float = 1000.0):
        """严格的无冲突调度"""
        
        # 重置状态
        for queue in scheduler.resource_queues.values():
            queue.available_time = 0.0
            queue.release_binding()
            if hasattr(queue, 'queues'):
                for p in TaskPriority:
                    queue.queues[p].clear()
        
        for task in scheduler.tasks.values():
            task.schedule_info = None
            task.last_execution_time = -float('inf')
            task.ready_time = 0
            task.current_segmentation = {}
            task.total_segmentation_overhead = 0.0
        
        scheduler.schedule_history.clear()
        scheduler.active_bindings.clear()
        if hasattr(scheduler, 'segmentation_decisions_history'):
            scheduler.segmentation_decisions_history.clear()
        
        # 重置资源占用跟踪
        scheduler._resource_occupancy.reset()
        
        # 跟踪任务执行次数
        from collections import defaultdict
        task_execution_count = defaultdict(int)
        current_time = 0.0
        
        while current_time < time_window:
            # 清理过期的绑定
            scheduler.cleanup_expired_bindings(current_time)
            
            # 获取所有就绪的任务
            ready_tasks = []
            
            for task in scheduler.tasks.values():
                # 检查最小间隔
                if task.last_execution_time + task.min_interval_ms > current_time:
                    continue
                
                # 检查依赖关系
                deps_satisfied = True
                max_dep_end_time = 0.0
                
                for dep_id in task.dependencies:
                    dep_task = scheduler.tasks.get(dep_id)
                    if dep_task:
                        if task_execution_count[dep_id] <= task_execution_count[task.task_id]:
                            deps_satisfied = False
                            break
                        if dep_task.schedule_info:
                            max_dep_end_time = max(max_dep_end_time, dep_task.schedule_info.end_time)
                
                if deps_satisfied:
                    task.ready_time = max(current_time, max_dep_end_time)
                    if task.ready_time <= current_time:
                        ready_tasks.append(task)
            
            # 按优先级和FIFO顺序排序就绪任务
            ready_tasks.sort(key=lambda t: (
                t.priority.value,  # 优先级（值越小优先级越高）
                getattr(t, '_fifo_order', 999),  # FIFO顺序
                t.task_id  # 任务ID作为最后的排序依据
            ))
            
            # 尝试调度任务
            scheduled_any = False
            
            for task in ready_tasks:
                # 查找可用资源
                assigned_resources = find_truly_available_resources(
                    scheduler, task, current_time
                )
                
                if assigned_resources:
                    # 创建调度
                    schedule_info = create_schedule_and_occupy_resources(
                        scheduler, task, assigned_resources, current_time
                    )
                    
                    if schedule_info:
                        # 记录调度
                        scheduler.schedule_history.append(schedule_info)
                        task.schedule_info = schedule_info
                        task.last_execution_time = schedule_info.start_time
                        task_execution_count[task.task_id] += 1
                        scheduled_any = True
                        
                        # 一次只调度一个任务，确保资源占用正确更新
                        break
            
            if not scheduled_any:
                # 没有任务可以调度，时间前进
                current_time += 0.1
            else:
                # 继续在当前时间检查
                pass
        
        return scheduler.schedule_history
    
    # 替换方法
    scheduler.priority_aware_schedule_with_segmentation = strict_conflict_free_schedule
    
    print("✅ 严格的资源冲突修复已应用")


class ResourceOccupancyTracker:
    """资源占用跟踪器"""
    
    def __init__(self):
        self.occupancy_timeline = defaultdict(list)  # {resource_id: [(start, end, task_id)]}
    
    def reset(self):
        """重置占用记录"""
        self.occupancy_timeline.clear()
    
    def is_resource_available(self, resource_id: str, start_time: float, end_time: float) -> bool:
        """检查资源在指定时间段是否可用"""
        for occ_start, occ_end, _ in self.occupancy_timeline.get(resource_id, []):
            # 检查时间段是否重叠
            if not (end_time <= occ_start or start_time >= occ_end):
                return False
        return True
    
    def occupy_resource(self, resource_id: str, start_time: float, end_time: float, task_id: str):
        """占用资源"""
        self.occupancy_timeline[resource_id].append((start_time, end_time, task_id))
        # 保持时间线排序
        self.occupancy_timeline[resource_id].sort(key=lambda x: x[0])
    
    def get_next_available_time(self, resource_id: str, after_time: float) -> float:
        """获取资源的下一个可用时间"""
        next_available = after_time
        for _, occ_end, _ in self.occupancy_timeline.get(resource_id, []):
            if occ_end > after_time:
                next_available = max(next_available, occ_end)
        return next_available


def find_truly_available_resources(scheduler, task, current_time) -> Optional[Dict[ResourceType, str]]:
    """查找真正可用的资源（考虑占用情况）"""
    
    assigned_resources = {}
    
    # 获取任务需要的资源类型
    required_types = set()
    for seg in task.segments:
        required_types.add(seg.resource_type)
    
    # 为每种资源类型找到可用的资源
    for res_type in required_types:
        found_resource = None
        
        # 遍历该类型的所有资源
        for resource in scheduler.resources[res_type]:
            resource_id = resource.unit_id
            
            # 计算任务在这个资源上的执行时间
            duration = 0
            for seg in task.segments:
                if seg.resource_type == res_type:
                    duration += seg.get_duration(resource.bandwidth)
            
            end_time = current_time + duration
            
            # 检查资源是否真正可用
            if scheduler._resource_occupancy.is_resource_available(
                resource_id, current_time, end_time
            ):
                found_resource = resource_id
                break
        
        if not found_resource:
            # 如果找不到可用资源，返回None
            return None
        
        assigned_resources[res_type] = found_resource
    
    return assigned_resources


def create_schedule_and_occupy_resources(scheduler, task, assigned_resources, current_time):
    """创建调度并占用资源"""
    
    # 获取子段
    sub_segments = task.get_sub_segments_for_scheduling()
    
    actual_start = current_time
    actual_end = actual_start
    sub_segment_schedule = []
    
    # 处理每个子段并占用资源
    for sub_seg in sub_segments:
        if sub_seg.resource_type in assigned_resources:
            resource_id = assigned_resources[sub_seg.resource_type]
            resource = next(r for r in scheduler.resources[sub_seg.resource_type] 
                          if r.unit_id == resource_id)
            
            sub_seg_start = actual_start + sub_seg.start_time
            sub_seg_duration = sub_seg.get_duration(resource.bandwidth)
            sub_seg_end = sub_seg_start + sub_seg_duration
            
            # 占用资源
            scheduler._resource_occupancy.occupy_resource(
                resource_id, sub_seg_start, sub_seg_end, task.task_id
            )
            
            # 记录子段调度
            sub_segment_schedule.append((sub_seg.sub_id, sub_seg_start, sub_seg_end))
            
            actual_end = max(actual_end, sub_seg_end)
    
    # 创建调度信息
    from .models import TaskScheduleInfo
    schedule_info = TaskScheduleInfo(
        task_id=task.task_id,
        start_time=actual_start,
        end_time=actual_end,
        assigned_resources=assigned_resources,
        actual_latency=actual_end - current_time,
        runtime_type=task.runtime_type,
        used_cuts=task.current_segmentation.copy() if hasattr(task, 'current_segmentation') else {},
        segmentation_overhead=getattr(task, 'total_segmentation_overhead', 0.0),
        sub_segment_schedule=sub_segment_schedule
    )
    
    return schedule_info


if __name__ == "__main__":
    print("严格的资源冲突修复模块")
    print("确保没有任何资源冲突")
