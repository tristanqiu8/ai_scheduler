#!/usr/bin/env python3
"""
精简的神经网络任务类 - 移除 start_time，专注于任务定义
"""

from typing import List, Dict, Set, Optional, Tuple
from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.models import ResourceSegment, SubSegment


class NNTask:
    """神经网络任务类 - 纯粹的任务定义，不包含调度逻辑"""
    
    def __init__(self, 
                 task_id: str, 
                 name: str = "", 
                 priority: TaskPriority = TaskPriority.NORMAL,
                 runtime_type: RuntimeType = RuntimeType.ACPU_RUNTIME,
                 segmentation_strategy: SegmentationStrategy = SegmentationStrategy.NO_SEGMENTATION):
        self.task_id = task_id
        self.name = name or f"Task_{task_id}"
        self.priority = priority
        self.runtime_type = runtime_type
        self.segmentation_strategy = segmentation_strategy
        
        self.segments: List[ResourceSegment] = []
        self.dependencies: Set[str] = set()
        self.fps_requirement: float = 30.0
        self.latency_requirement: float = 100.0
        
        # 分段相关
        self.preset_cut_configurations: Dict[str, List[List[str]]] = {}
        self.selected_cut_config_index: Dict[str, int] = {}
    
    # ========== 任务定义方法 ==========
    
    def add_segment(self, 
                    resource_type: ResourceType,
                    duration_table: Dict[float, float],
                    segment_id: Optional[str] = None) -> ResourceSegment:
        """添加单个任务段
        
        Args:
            resource_type: 资源类型 (NPU/DSP)
            duration_table: 带宽到执行时间的映射 {bandwidth: duration_ms}
            segment_id: 段标识符
            
        Returns:
            创建的资源段
        """
        if segment_id is None:
            segment_id = f"{resource_type.value.lower()}_seg_{len(self.segments)}"
        
        segment = ResourceSegment(
            resource_type=resource_type,
            duration_table=duration_table,
            start_time=0,  # 固定为0，因为实际时序由调度器决定
            segment_id=segment_id
        )
        self.segments.append(segment)
        return segment
    
    def add_sequential_segments(self, 
                               segments: List[Tuple[ResourceType, Dict[float, float], str]]):
        """批量添加顺序执行的段
        
        Args:
            segments: [(资源类型, 时长表, 段ID)]
        """
        for resource_type, duration_table, segment_id in segments:
            self.add_segment(resource_type, duration_table, segment_id)
    
    def set_npu_only(self, duration_table: Dict[float, float], segment_id: str = "npu_main"):
        """设置为纯NPU任务"""
        self.segments = []
        self.add_segment(ResourceType.NPU, duration_table, segment_id)
    
    def set_dsp_only(self, duration_table: Dict[float, float], segment_id: str = "dsp_main"):
        """设置为纯DSP任务"""
        self.segments = []
        self.add_segment(ResourceType.DSP, duration_table, segment_id)
    
    # ========== 分段相关方法 ==========
    
    def add_cut_points_to_segment(self, 
                                 segment_id: str,
                                 cut_points: List[Tuple[str, Dict[float, float], float]]):
        """为指定段添加切分点
        
        Args:
            segment_id: 段ID
            cut_points: [(切分点ID, 切分前的时长表, 切分开销)]
        """
        segment = self.get_segment_by_id(segment_id)
        if segment:
            for op_id, before_duration_table, overhead_ms in cut_points:
                segment.add_cut_point(op_id, before_duration_table, overhead_ms)
        else:
            raise ValueError(f"Segment {segment_id} not found")
    
    def set_preset_cut_configurations(self, segment_id: str, configurations: List[List[str]]):
        """设置预定义的切分配置"""
        self.preset_cut_configurations[segment_id] = configurations
        self.selected_cut_config_index[segment_id] = 0  # 默认不切分
    
    def select_cut_configuration(self, segment_id: str, config_index: int):
        """选择特定的切分配置"""
        if segment_id in self.preset_cut_configurations:
            max_index = len(self.preset_cut_configurations[segment_id]) - 1
            if 0 <= config_index <= max_index:
                self.selected_cut_config_index[segment_id] = config_index
            else:
                raise ValueError(f"Config index {config_index} out of range")
        else:
            raise ValueError(f"No preset configurations for segment {segment_id}")
    
    def get_segment_by_id(self, segment_id: str) -> Optional[ResourceSegment]:
        """根据ID获取段"""
        return next((seg for seg in self.segments if seg.segment_id == segment_id), None)
    
    def apply_segmentation(self) -> List[SubSegment]:
        """应用当前的分段配置，返回所有子段"""
        all_sub_segments = []
        
        for segment in self.segments:
            # 获取该段的切分配置
            cuts = []
            if self.segmentation_strategy == SegmentationStrategy.NO_SEGMENTATION:
                cuts = []
            elif self.segmentation_strategy == SegmentationStrategy.FORCED_SEGMENTATION:
                cuts = segment.get_available_cuts()
            elif self.segmentation_strategy == SegmentationStrategy.CUSTOM_SEGMENTATION:
                if segment.segment_id in self.preset_cut_configurations:
                    config_idx = self.selected_cut_config_index.get(segment.segment_id, 0)
                    cuts = self.preset_cut_configurations[segment.segment_id][config_idx]
            
            # 应用切分
            sub_segments = segment.apply_segmentation(cuts)
            all_sub_segments.extend(sub_segments)
        
        return all_sub_segments
    
    # ========== 任务属性设置 ==========
    
    def set_performance_requirements(self, fps: float, latency: float):
        """设置性能需求"""
        self.fps_requirement = fps
        self.latency_requirement = latency
    
    def add_dependency(self, task_id: str):
        """添加任务依赖"""
        self.dependencies.add(task_id)
    
    def add_dependencies(self, task_ids: List[str]):
        """批量添加依赖"""
        self.dependencies.update(task_ids)
    
    # ========== 信息获取方法 ==========
    
    def estimate_duration(self, bandwidth_map: Dict[ResourceType, float]) -> float:
        """估算任务在给定带宽下的执行时间
        
        Args:
            bandwidth_map: {资源类型: 带宽}
            
        Returns:
            估算的总执行时间（毫秒）
        """
        total_duration = 0.0
        
        # 如果任务被分段，使用子段计算
        sub_segments = self.apply_segmentation()
        if sub_segments:
            for sub_seg in sub_segments:
                bw = bandwidth_map.get(sub_seg.resource_type, 40.0)  # 默认40
                total_duration += sub_seg.get_duration(bw)
        else:
            # 使用原始段计算
            for segment in self.segments:
                bw = bandwidth_map.get(segment.resource_type, 40.0)
                total_duration += segment.get_duration(bw)
        
        return total_duration
    
    def get_resource_requirements(self) -> Set[ResourceType]:
        """获取任务需要的资源类型"""
        return {seg.resource_type for seg in self.segments}
    
    def get_segment_count(self) -> int:
        """获取段数（考虑分段）"""
        sub_segments = self.apply_segmentation()
        return len(sub_segments) if sub_segments else len(self.segments)
    
    @property
    def min_interval_ms(self) -> float:
        """最小调度间隔（基于FPS需求）"""
        return 1000.0 / self.fps_requirement if self.fps_requirement > 0 else float('inf')
    
    @property
    def uses_dsp(self) -> bool:
        """是否使用DSP资源"""
        return ResourceType.DSP in self.get_resource_requirements()
    
    @property
    def uses_npu(self) -> bool:
        """是否使用NPU资源"""
        return ResourceType.NPU in self.get_resource_requirements()
    
    @property
    def is_mixed_resource(self) -> bool:
        """是否是混合资源任务"""
        return len(self.get_resource_requirements()) > 1
    
    def __repr__(self):
        return (f"NNTask(id={self.task_id}, name={self.name}, "
                f"priority={self.priority.name}, "
                f"segments={len(self.segments)}, "
                f"fps={self.fps_requirement})")


# ========== 便捷的工厂方法 ==========

def create_npu_task(task_id: str, name: str, duration_table: Dict[float, float], 
                   **kwargs) -> NNTask:
    """创建纯NPU任务"""
    task = NNTask(task_id, name, **kwargs)
    task.set_npu_only(duration_table)
    return task


def create_dsp_task(task_id: str, name: str, duration_table: Dict[float, float],
                   **kwargs) -> NNTask:
    """创建纯DSP任务"""
    task = NNTask(task_id, name, **kwargs)
    task.set_dsp_only(duration_table)
    return task


def create_mixed_task(task_id: str, name: str, 
                     segments: List[Tuple[ResourceType, Dict[float, float], str]],
                     **kwargs) -> NNTask:
    """创建混合资源任务"""
    task = NNTask(task_id, name, **kwargs)
    task.add_sequential_segments(segments)
    return task
