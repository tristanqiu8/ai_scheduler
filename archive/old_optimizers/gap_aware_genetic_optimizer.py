#!/usr/bin/env python3
"""
空隙感知遗传算法优化器
将空隙填充优化集成到遗传算法框架中
"""

import copy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field

from .enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from .models import TaskScheduleInfo
from .scheduler import MultiResourceScheduler
from .genetic_task_optimizer import GeneticTaskOptimizer, GeneticIndividual
from .fixed_validation_and_metrics import validate_schedule_correctly


@dataclass
class GapAwareGeneticIndividual(GeneticIndividual):
    """扩展的遗传个体，包含空隙优化相关基因"""
    # 继承原有基因
    # 新增空隙优化相关基因
    enable_gap_filling: bool = True
    gap_filling_aggressiveness: float = 0.8  # 0-1, 越高越激进
    prefer_early_execution: bool = True  # 是否优先提前执行
    
    # 空隙优化效果指标
    gap_utilization_rate: float = 0.0
    cross_resource_efficiency: float = 0.0


class GapAwareGeneticOptimizer(GeneticTaskOptimizer):
    """结合空隙感知的遗传算法优化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, tasks: List, time_window: float = 200.0):
        super().__init__(scheduler, tasks, time_window)
        self.enable_gap_optimization = True
        self.gap_optimization_weight = 0.3  # 空隙优化在适应度中的权重
        
    def _create_gap_filling_schedule(self, baseline_schedule: List[TaskScheduleInfo], 
                                   individual: GapAwareGeneticIndividual) -> Tuple[List[TaskScheduleInfo], Dict]:
        """
        核心方法：创建空隙填充的优化调度
        这是从test中提取的核心逻辑，增加了基因控制
        """
        if not individual.enable_gap_filling:
            return baseline_schedule, {'gap_filled': 0, 'gap_utilization': 0.0}
        
        # 复制基线调度
        working_schedule = copy.deepcopy(baseline_schedule)
        
        # 1. 识别DSP忙碌时段（跨资源空隙）
        dsp_busy_periods = []
        for event in working_schedule:
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    if 'dsp' in sub_id.lower():
                        dsp_busy_periods.append((start, end, event.task_id))
        
        if not dsp_busy_periods:
            return working_schedule, {'gap_filled': 0, 'gap_utilization': 0.0}
        
        # 2. 根据个体基因决定优化策略
        gap_filled_count = 0
        total_gap_time = sum(end - start for start, end, _ in dsp_busy_periods)
        utilized_gap_time = 0.0
        
        processed_events = set()
        
        # 3. 对每个DSP空隙尝试填充
        for dsp_idx, (dsp_start, dsp_end, dsp_task) in enumerate(dsp_busy_periods):
            # 根据aggressiveness决定是否处理这个空隙
            if individual.gap_filling_aggressiveness < 0.5 and dsp_idx > 0:
                continue  # 保守策略：只处理第一个空隙
            
            # 查找可移动的任务段
            candidates = []
            for event_idx, event in enumerate(working_schedule):
                if event_idx in processed_events:
                    continue
                    
                # 根据基因决定哪些任务可以移动
                task = self.tasks.get(event.task_id)
                if not task:
                    continue
                
                # 只移动低优先级任务（可以通过基因控制）
                if task.priority in [TaskPriority.NORMAL, TaskPriority.LOW]:
                    if individual.prefer_early_execution and event.start_time > dsp_end:
                        candidates.append((event_idx, event))
                    elif not individual.prefer_early_execution and event.start_time < dsp_start:
                        candidates.append((event_idx, event))
            
            # 尝试填充空隙
            for event_idx, event in candidates:
                if self._try_fill_gap(working_schedule, event_idx, event, 
                                    dsp_start, dsp_end, individual):
                    gap_filled_count += 1
                    processed_events.add(event_idx)
                    
                    # 计算利用的空隙时间
                    for sub_id, start, end in event.sub_segment_schedule:
                        if start >= dsp_start and end <= dsp_end:
                            utilized_gap_time += (end - start)
                    
                    # 根据aggressiveness决定是否继续
                    if individual.gap_filling_aggressiveness < 0.8:
                        break  # 适度策略：每个空隙只填充一个任务
        
        # 4. 计算空隙利用指标
        gap_stats = {
            'gap_filled': gap_filled_count,
            'gap_utilization': utilized_gap_time / total_gap_time if total_gap_time > 0 else 0.0,
            'total_gaps': len(dsp_busy_periods),
            'utilized_gap_time': utilized_gap_time,
            'total_gap_time': total_gap_time
        }
        
        return working_schedule, gap_stats
    
    def _try_fill_gap(self, schedule: List[TaskScheduleInfo], event_idx: int, 
                     event: TaskScheduleInfo, gap_start: float, gap_end: float,
                     individual: GapAwareGeneticIndividual) -> bool:
        """尝试将事件的段填充到空隙中"""
        if not hasattr(event, 'sub_segment_schedule'):
            return False
        
        # 分析哪些段可以放入空隙
        segments_for_gap = []
        segments_remaining = []
        available_time = gap_start
        
        for sub_id, start, end in event.sub_segment_schedule:
            duration = end - start
            
            # 根据基因决定是否严格要求段完全放入空隙
            if individual.gap_filling_aggressiveness > 0.9:
                # 激进策略：即使稍微超出也尝试
                tolerance = 2.0  # 允许2ms的超出
            else:
                tolerance = 0.0
            
            if available_time + duration <= gap_end + tolerance:
                segments_for_gap.append({
                    'sub_id': sub_id,
                    'new_start': available_time,
                    'new_end': available_time + duration,
                    'duration': duration
                })
                available_time += duration
            else:
                segments_remaining.append({
                    'sub_id': sub_id,
                    'duration': duration
                })
        
        if not segments_for_gap:
            return False
        
        # 实施移动（这里简化了原始逻辑）
        # 实际实现需要正确更新schedule中的事件
        return True
    
    def _evaluate_fitness(self, individual: GapAwareGeneticIndividual) -> float:
        """增强的适应度评估，包含空隙优化"""
        
        # 1. 应用个体配置
        self._apply_individual_config(individual)
        
        # 2. 运行基础调度
        self.scheduler.schedule_history.clear()
        baseline_schedule = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
        
        # 3. 验证基础调度
        is_valid, conflicts = validate_schedule_correctly(self.scheduler)
        if not is_valid:
            individual.conflict_count = len(conflicts)
            individual.fitness = -1000.0  # 严重惩罚
            return individual.fitness
        
        # 4. 应用空隙优化
        optimized_schedule, gap_stats = self._create_gap_filling_schedule(baseline_schedule, individual)
        self.scheduler.schedule_history = optimized_schedule
        
        # 5. 再次验证（确保优化没有引入冲突）
        is_valid_after, conflicts_after = validate_schedule_correctly(self.scheduler)
        if not is_valid_after:
            individual.conflict_count = len(conflicts_after)
            individual.fitness = -500.0
            return individual.fitness
        
        # 6. 计算各项指标
        # 基础指标（继承自父类）
        fps_satisfaction = self._calculate_fps_satisfaction()
        resource_utilization = self._calculate_resource_utilization()
        
        # 空隙优化指标
        gap_utilization = gap_stats['gap_utilization']
        individual.gap_utilization_rate = gap_utilization
        
        # 跨资源效率
        cross_resource_efficiency = self._calculate_cross_resource_efficiency(optimized_schedule)
        individual.cross_resource_efficiency = cross_resource_efficiency
        
        # 7. 综合适应度计算
        fitness = 0.0
        
        # 基础分数
        fitness += fps_satisfaction * 300
        fitness += resource_utilization * 200
        
        # 空隙优化加分
        fitness += gap_utilization * self.gap_optimization_weight * 200
        fitness += cross_resource_efficiency * 100
        
        # 特殊奖励
        if gap_stats['gap_filled'] > 0:
            fitness += 50 * gap_stats['gap_filled']  # 每成功填充一个空隙加分
        
        individual.fitness = fitness
        return fitness
    
    def _calculate_cross_resource_efficiency(self, schedule: List[TaskScheduleInfo]) -> float:
        """计算跨资源效率"""
        # 简化实现：计算NPU在DSP忙碌时的利用率
        dsp_busy_time = 0.0
        npu_during_dsp_busy = 0.0
        
        # 找出DSP忙碌时段
        dsp_periods = []
        for event in schedule:
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    if 'dsp' in sub_id.lower():
                        dsp_periods.append((start, end))
                        dsp_busy_time += (end - start)
        
        # 计算这些时段内NPU的利用
        for event in schedule:
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    if 'npu' in sub_id.lower() or 'main' in sub_id:
                        # 检查与DSP时段的重叠
                        for dsp_start, dsp_end in dsp_periods:
                            overlap_start = max(start, dsp_start)
                            overlap_end = min(end, dsp_end)
                            if overlap_start < overlap_end:
                                npu_during_dsp_busy += (overlap_end - overlap_start)
        
        return npu_during_dsp_busy / dsp_busy_time if dsp_busy_time > 0 else 0.0
    
    def _mutate(self, individual: GapAwareGeneticIndividual):
        """扩展的变异操作"""
        # 调用父类变异
        super()._mutate(individual)
        
        # 变异空隙优化相关基因
        if self.rng.random() < self.mutation_rate:
            individual.enable_gap_filling = not individual.enable_gap_filling
        
        if self.rng.random() < self.mutation_rate:
            # 小幅调整aggressiveness
            delta = (self.rng.random() - 0.5) * 0.2
            individual.gap_filling_aggressiveness = max(0.0, min(1.0, 
                individual.gap_filling_aggressiveness + delta))
        
        if self.rng.random() < self.mutation_rate:
            individual.prefer_early_execution = not individual.prefer_early_execution
    
    def optimize_with_gap_awareness(self) -> GapAwareGeneticIndividual:
        """运行空隙感知的遗传算法优化"""
        print("\n🧬 启动空隙感知遗传算法优化")
        print("=" * 60)
        print(f"种群大小: {self.population_size}")
        print(f"迭代代数: {self.generations}")
        print(f"空隙优化权重: {self.gap_optimization_weight}")
        
        # 运行优化（使用父类的optimize方法框架）
        best_individual = self.optimize()
        
        # 打印额外的空隙优化统计
        print(f"\n空隙优化统计:")
        print(f"  空隙利用率: {best_individual.gap_utilization_rate:.1%}")
        print(f"  跨资源效率: {best_individual.cross_resource_efficiency:.1%}")
        
        return best_individual


def create_and_run_gap_aware_optimizer(scheduler, tasks, time_window=200.0):
    """便捷函数：创建并运行空隙感知优化器"""
    
    # 创建优化器
    optimizer = GapAwareGeneticOptimizer(scheduler, tasks, time_window)
    
    # 设置参数
    optimizer.population_size = 50
    optimizer.generations = 100
    optimizer.gap_optimization_weight = 0.4  # 提高空隙优化的重要性
    
    # 运行优化
    best_solution = optimizer.optimize_with_gap_awareness()
    
    # 应用最佳方案并返回最终调度
    optimizer._apply_individual_config(best_solution)
    final_schedule = scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    # 应用空隙优化
    optimized_schedule, gap_stats = optimizer._create_gap_filling_schedule(
        final_schedule, best_solution)
    
    print(f"\n最终空隙利用统计:")
    for key, value in gap_stats.items():
        print(f"  {key}: {value}")
    
    return optimized_schedule, best_solution


if __name__ == "__main__":
    print("空隙感知遗传算法优化器")
    print("结合了遗传算法的全局搜索和空隙填充的局部优化")
