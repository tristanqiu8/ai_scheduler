#!/usr/bin/env python3
"""
带宽管理器 - 管理共享带宽池（修复版）
"""

from typing import Dict, Set, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import defaultdict
from NNScheduler.core.enums import ResourceType


@dataclass
class BandwidthAllocation:
    """带宽分配记录"""
    resource_id: str
    start_time: float
    end_time: float
    task_id: str
    allocated_bandwidth: float  # 新增：记录分配的带宽
    
    def is_active(self, current_time: float) -> bool:
        """检查分配是否在当前时间活跃"""
        return self.start_time <= current_time < self.end_time


class BandwidthManager:
    """共享带宽管理器 - NPU和DSP共享同一个带宽池"""
    
    def __init__(self, total_system_bandwidth: float = 120.0):
        # 系统总带宽（NPU和DSP共享）
        self.total_system_bandwidth = total_system_bandwidth
        self.min_bandwidth_per_unit = 1.0  # 每个单元的最小带宽保证
        
        # 活跃的带宽分配
        self.active_allocations: List[BandwidthAllocation] = []
        
        # 带宽历史记录（用于分析和可视化）
        self.bandwidth_history: Dict[str, List[Tuple[float, float, float]]] = defaultdict(list)
        # resource_id -> [(start_time, end_time, allocated_bandwidth)]
    
    def get_available_bandwidth(self, current_time: float,
                               exclude_resource: Optional[str] = None) -> float:
        """获取当前时刻的可用带宽（所有资源类型共享）
        
        这个方法用于预估，实际分配在 allocate_bandwidth 中进行
        """
        # 清理过期的分配
        self.cleanup_expired_allocations(current_time)
        
        # 计算当前所有活跃的资源数
        active_resources = self._get_all_active_resources(current_time)
        
        # 如果排除某个资源，从活跃列表中移除
        if exclude_resource and exclude_resource in active_resources:
            active_resources.discard(exclude_resource)
        
        # 计算活跃资源数（包括即将使用的资源）
        active_count = len(active_resources)
        if exclude_resource is None or exclude_resource not in active_resources:
            active_count += 1  # 加上请求的资源
        
        # 平均分配带宽
        if active_count > 0:
            per_unit = self.total_system_bandwidth / active_count
        else:
            per_unit = self.total_system_bandwidth
            
        return max(per_unit, self.min_bandwidth_per_unit)
    
    def allocate_bandwidth(self, resource_id: str, resource_type: ResourceType,
                          task_id: str, start_time: float, end_time: float) -> float:
        """为资源分配带宽
        
        Returns:
            分配的平均带宽值
        """
        # 清理过期的分配
        self.cleanup_expired_allocations(start_time)
        
        # 计算这个时间段的带宽分配
        bandwidth_over_time = self._calculate_bandwidth_over_time(
            resource_id, start_time, end_time
        )
        
        # 创建新的分配记录
        if bandwidth_over_time:
            # 计算平均带宽
            total_weighted_bandwidth = sum(
                (end - start) * bw for start, end, bw in bandwidth_over_time
            )
            total_duration = end_time - start_time
            avg_bandwidth = total_weighted_bandwidth / total_duration if total_duration > 0 else 0
            
            allocation = BandwidthAllocation(
                resource_id=resource_id,
                start_time=start_time,
                end_time=end_time,
                task_id=task_id,
                allocated_bandwidth=avg_bandwidth
            )
            self.active_allocations.append(allocation)
            
            # 记录到历史
            for time_start, time_end, bandwidth in bandwidth_over_time:
                self.bandwidth_history[resource_id].append((time_start, time_end, bandwidth))
            
            return avg_bandwidth
        
        return self.total_system_bandwidth  # 默认返回全部带宽
    
    def _calculate_bandwidth_over_time(self, resource_id: str,
                                      start_time: float, 
                                      end_time: float) -> List[Tuple[float, float, float]]:
        """计算一个时间段内的带宽变化
        
        Returns:
            List of (segment_start, segment_end, bandwidth)
        """
        # 收集所有相关的时间点
        time_points = {start_time, end_time}
        
        # 添加其他分配的开始和结束时间
        for alloc in self.active_allocations:
            # 只考虑有重叠的分配
            if alloc.end_time > start_time and alloc.start_time < end_time:
                if alloc.start_time >= start_time:
                    time_points.add(alloc.start_time)
                if alloc.end_time <= end_time:
                    time_points.add(alloc.end_time)
        
        # 按时间排序
        sorted_times = sorted(time_points)
        
        # 计算每个时间段的带宽
        bandwidth_segments = []
        for i in range(len(sorted_times) - 1):
            segment_start = sorted_times[i]
            segment_end = sorted_times[i + 1]
            segment_mid = (segment_start + segment_end) / 2
            
            # 计算这个时间段内的活跃资源数
            active_count = 1  # 包括当前资源
            for alloc in self.active_allocations:
                if (alloc.resource_id != resource_id and 
                    alloc.start_time <= segment_mid < alloc.end_time):
                    active_count += 1
            
            # 所有活跃资源平分总带宽
            bandwidth = self.total_system_bandwidth / active_count
            bandwidth = max(bandwidth, self.min_bandwidth_per_unit)
            
            bandwidth_segments.append((segment_start, segment_end, bandwidth))
        
        return bandwidth_segments
    
    def _get_all_active_resources(self, current_time: float) -> Set[str]:
        """获取某时刻所有活跃的资源（NPU和DSP）"""
        active = set()
        for alloc in self.active_allocations:
            if alloc.is_active(current_time):
                active.add(alloc.resource_id)
        return active
    
    def cleanup_expired_allocations(self, current_time: float):
        """清理过期的分配记录"""
        self.active_allocations = [
            alloc for alloc in self.active_allocations
            if alloc.end_time > current_time
        ]
    
    def get_bandwidth_timeline(self, resource_id: str, 
                             start_time: float, 
                             end_time: float) -> List[Tuple[float, float, float]]:
        """获取资源的带宽时间线"""
        timeline = []
        for time_start, time_end, bandwidth in self.bandwidth_history.get(resource_id, []):
            if time_end > start_time and time_start < end_time:
                timeline.append((
                    max(time_start, start_time),
                    min(time_end, end_time),
                    bandwidth
                ))
        return sorted(timeline)
    
    def get_system_status(self, current_time: float) -> Dict:
        """获取系统当前状态"""
        # 清理过期分配
        self.cleanup_expired_allocations(current_time)
        
        active_resources = self._get_all_active_resources(current_time)
        active_npus = [r for r in active_resources if "NPU" in r]
        active_dsps = [r for r in active_resources if "DSP" in r]
        
        total_active = len(active_resources)
        available_bandwidth_per_unit = (
            self.total_system_bandwidth / total_active 
            if total_active > 0 
            else self.total_system_bandwidth
        )
        
        return {
            'total_system_bandwidth': self.total_system_bandwidth,
            'active_resources': {
                'total': total_active,
                'npus': len(active_npus),
                'dsps': len(active_dsps),
                'list': sorted(active_resources)
            },
            'bandwidth_per_unit': available_bandwidth_per_unit,
            'allocations': len(self.active_allocations)
        }
    
    def reset(self):
        """重置带宽管理器"""
        self.active_allocations.clear()
        self.bandwidth_history.clear()
