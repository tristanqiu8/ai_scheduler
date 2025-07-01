#!/usr/bin/env python3
"""
激进的遗传算法优化器 - 专注于最大化空闲时间
目标：通过优化任务配置，使紧凑化后的空闲时间最大化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Optional
from collections import defaultdict
import copy
import random
import numpy as np

# 导入必要的类和枚举
from .enums import TaskPriority, RuntimeType, SegmentationStrategy, ResourceType
from .genetic_task_optimizer import GeneticTaskOptimizer
from .models import TaskScheduleInfo
from dataclasses import dataclass, field

# 扩展GeneticIndividual以支持idle_time
@dataclass
class GeneticIndividual:
    """遗传算法个体（扩展版）"""
    # 基因编码
    task_priorities: Dict[str, TaskPriority] = field(default_factory=dict)
    task_runtime_types: Dict[str, RuntimeType] = field(default_factory=dict)
    task_segmentation_strategies: Dict[str, SegmentationStrategy] = field(default_factory=dict)
    task_segmentation_configs: Dict[str, int] = field(default_factory=dict)  # 分段配置索引
    resource_assignments: Dict[str, Dict[ResourceType, str]] = field(default_factory=dict)  # 资源分配
    
    # 适应度相关
    fitness: float = 0.0
    fps_satisfaction_rate: float = 0.0
    conflict_count: int = 0
    resource_utilization: float = 0.0
    avg_latency: float = 0.0
    idle_time: float = 0.0  # 新增：空闲时间
    
    def __hash__(self):
        """使个体可哈希"""
        return hash(str(self.task_priorities) + str(self.task_runtime_types))
from .fixed_validation_and_metrics import validate_schedule_correctly


class AggressiveIdleOptimizer(GeneticTaskOptimizer):
    """激进的空闲时间优化器"""
    
    def __init__(self, scheduler, tasks, time_window=200.0):
        super().__init__(scheduler, tasks, time_window)
        # 更激进的参数
        self.population_size = 100  # 更大的种群
        self.generations = 200      # 更多代数
        self.elite_size = 5         # 减少精英保留
        self.mutation_rate = 0.4    # 大幅提高变异率
        self.crossover_rate = 0.9   # 提高交叉率
        
        # 新增参数
        self.aggressive_mutation_rate = 0.6  # 激进变异率
        self.chaos_injection_rate = 0.1      # 混沌注入率
        self.fps_tolerance = 0.85            # FPS容忍度（85%）
        
        # 基线性能
        self.baseline_performance = None
        
    def set_baseline_performance(self, baseline_stats, baseline_conflicts):
        """设置基线性能指标"""
        self.baseline_performance = {
            'fps_rates': {tid: info['fps_rate'] 
                         for tid, info in baseline_stats['task_fps'].items()},
            'avg_fps': baseline_stats['total_fps_rate'] / len(self.tasks),
            'conflicts': baseline_conflicts,
            'task_counts': {tid: info['count'] 
                           for tid, info in baseline_stats['task_fps'].items()}
        }
        
    def _calculate_separate_utilization(self):
        """分别计算NPU和DSP的利用率"""
        npu_busy_time = 0.0
        dsp_busy_time = 0.0
        npu_count = 0
        dsp_count = 0
        
        for res_type, resources in self.scheduler.resources.items():
            if isinstance(resources, dict):
                resource_items = resources.items()
            elif isinstance(resources, list):
                resource_items = [(f"{res_type.value}_{i}", res) for i, res in enumerate(resources)]
            else:
                continue
            
            for res_id, resource in resource_items:
                busy_time = 0.0
                last_end = 0.0
                
                for event in sorted(self.scheduler.schedule_history, key=lambda x: x.start_time):
                    if event.assigned_resources.get(res_type) == res_id:
                        if event.start_time >= last_end:
                            busy_time += event.end_time - event.start_time
                            last_end = event.end_time
                
                if res_type.value == "NPU":
                    npu_busy_time += busy_time
                    npu_count += 1
                elif res_type.value == "DSP":
                    dsp_busy_time += busy_time
                    dsp_count += 1
        
        npu_util = (npu_busy_time / (self.time_window * npu_count)) if npu_count > 0 else 0
        dsp_util = (dsp_busy_time / (self.time_window * dsp_count)) if dsp_count > 0 else 0
        
        return npu_util, dsp_util
        
    def _evaluate_fitness_for_idle(self, individual: GeneticIndividual) -> float:
        """针对空闲时间优化的适应度函数"""
        # 应用配置
        self._apply_individual_config(individual)
        
        # 清空调度历史
        self.scheduler.schedule_history.clear()
        
        try:
            # 运行调度
            results = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
            
            # 验证
            is_valid, conflicts = validate_schedule_correctly(self.scheduler)
            individual.conflict_count = len(conflicts)
            
            # 运行紧凑化算法估算空闲时间
            idle_time = self._estimate_idle_time()
            individual.idle_time = idle_time
            
            # 计算FPS
            task_counts = defaultdict(int)
            for event in self.scheduler.schedule_history:
                task_counts[event.task_id] += 1
            
            # FPS分析
            total_fps_satisfaction = 0.0
            critical_fps_violation = False
            
            for task in self.tasks:
                count = task_counts[task.task_id]
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                
                if expected > 0:
                    fps_rate = count / expected
                    total_fps_satisfaction += fps_rate
                    
                    # 关键任务的FPS检查
                    if task.priority == TaskPriority.CRITICAL and fps_rate < self.fps_tolerance:
                        critical_fps_violation = True
            
            individual.fps_satisfaction_rate = total_fps_satisfaction / len(self.tasks)
            
            # 计算资源利用率
            npu_util, dsp_util = self._calculate_separate_utilization()
            individual.resource_utilization = (npu_util + dsp_util) / 2
            
            # 新的适应度计算 - 专注于空闲时间
            fitness = 0.0
            
            # 1. 空闲时间是最重要的指标（权重最高）
            fitness += idle_time * 10.0  # 每ms空闲时间10分
            
            # 2. 基本的冲突惩罚
            if individual.conflict_count > 0:
                fitness -= individual.conflict_count * 100
            
            # 3. FPS要求（放宽标准）
            if individual.fps_satisfaction_rate >= self.fps_tolerance:
                fitness += 200  # 满足基本要求即可
            else:
                # 低于容忍度的惩罚
                fps_penalty = (self.fps_tolerance - individual.fps_satisfaction_rate) * 500
                fitness -= fps_penalty
            
            # 4. 关键任务惩罚
            if critical_fps_violation:
                fitness -= 300
            
            # 5. 资源利用率奖励（鼓励高效利用）
            if individual.resource_utilization > 0.8:
                fitness += 100
            
            # 6. 任务优先级合理性
            priority_bonus = 0
            for task in self.tasks:
                # 低优先级任务降级奖励
                if task.fps_requirement <= 10 and individual.task_priorities.get(task.task_id) == TaskPriority.LOW:
                    priority_bonus += 20
                # 高FPS任务保持高优先级
                elif task.fps_requirement >= 25 and individual.task_priorities.get(task.task_id) in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                    priority_bonus += 10
            fitness += priority_bonus
            
        except Exception as e:
            print(f"评估失败: {e}")
            fitness = -10000.0
            individual.idle_time = 0
            
        individual.fitness = fitness
        return fitness
    
    def _estimate_idle_time(self) -> float:
        """估算紧凑化后的空闲时间"""
        if not self.scheduler.schedule_history:
            return self.time_window
        
        # 简单估算：找到第一个时间窗口内的最后一个事件
        first_window_events = [e for e in self.scheduler.schedule_history 
                              if e.start_time < self.time_window]
        
        if not first_window_events:
            return self.time_window
        
        # 按结束时间排序
        last_end = max(e.end_time for e in first_window_events)
        
        # 计算总的资源占用时间（考虑并行）
        resource_timelines = defaultdict(list)
        for event in first_window_events:
            for res_type, res_id in event.assigned_resources.items():
                resource_timelines[res_id].append((event.start_time, event.end_time))
        
        # 合并重叠时间段
        max_resource_end = 0
        for res_id, timeline in resource_timelines.items():
            if not timeline:
                continue
            
            # 排序并合并
            timeline.sort()
            merged_end = 0
            current_start, current_end = timeline[0]
            
            for start, end in timeline[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    merged_end = max(merged_end, current_end)
                    current_start, current_end = start, end
            
            merged_end = max(merged_end, current_end)
            max_resource_end = max(max_resource_end, merged_end)
        
        # 估算紧凑化后的空闲时间
        estimated_idle = self.time_window - max_resource_end
        return max(0, estimated_idle)
    
    def _create_random_aggressive_individual(self) -> GeneticIndividual:
        """创建更激进的随机个体"""
        individual = GeneticIndividual()
        
        for task in self.tasks:
            task_id = task.task_id
            
            # 更激进的优先级分配
            if task.fps_requirement <= 5:
                # 低FPS任务大概率降级
                individual.task_priorities[task_id] = random.choice([
                    TaskPriority.LOW, TaskPriority.LOW, TaskPriority.NORMAL
                ])
            elif task.fps_requirement >= 25:
                # 高FPS任务倾向高优先级
                individual.task_priorities[task_id] = random.choice([
                    TaskPriority.HIGH, TaskPriority.CRITICAL, TaskPriority.NORMAL
                ])
            else:
                # 中等任务随机
                individual.task_priorities[task_id] = random.choice(self.priority_options)
            
            # 运行时类型 - 更多变化
            if random.random() < 0.3:  # 30%概率违反常规
                individual.task_runtime_types[task_id] = random.choice(self.runtime_options)
            else:
                # 70%概率合理选择
                if task.uses_dsp:
                    individual.task_runtime_types[task_id] = RuntimeType.DSP_RUNTIME
                else:
                    individual.task_runtime_types[task_id] = RuntimeType.ACPU_RUNTIME
            
            # 分段策略 - 更激进
            if task_id in ["T2", "T3"]:  # YOLO任务
                individual.task_segmentation_strategies[task_id] = random.choice([
                    SegmentationStrategy.ADAPTIVE_SEGMENTATION,
                    SegmentationStrategy.FORCED_SEGMENTATION,  # 强制分段
                    SegmentationStrategy.CUSTOM_SEGMENTATION
                ])
            else:
                individual.task_segmentation_strategies[task_id] = random.choice([
                    SegmentationStrategy.NO_SEGMENTATION,
                    SegmentationStrategy.ADAPTIVE_SEGMENTATION
                ])
            
            # 分段配置
            individual.task_segmentation_configs[task_id] = random.randint(0, 4)
            
        return individual
    
    def _aggressive_mutate(self, individual: GeneticIndividual):
        """激进的变异策略"""
        for task in self.tasks:
            task_id = task.task_id
            
            # 优先级激进变异
            if random.random() < self.aggressive_mutation_rate:
                # 完全随机
                individual.task_priorities[task_id] = random.choice(self.priority_options)
            
            # 运行时类型变异
            if random.random() < self.mutation_rate:
                individual.task_runtime_types[task_id] = random.choice(self.runtime_options)
            
            # 分段策略变异
            if random.random() < self.mutation_rate:
                individual.task_segmentation_strategies[task_id] = random.choice(list(SegmentationStrategy))
            
            # 混沌注入 - 偶尔完全打乱一个任务的配置
            if random.random() < self.chaos_injection_rate:
                individual.task_priorities[task_id] = random.choice(self.priority_options)
                individual.task_runtime_types[task_id] = random.choice(self.runtime_options)
                individual.task_segmentation_strategies[task_id] = random.choice(list(SegmentationStrategy))
                individual.task_segmentation_configs[task_id] = random.randint(0, 4)
    
    def optimize_for_idle_time(self):
        """针对空闲时间的优化"""
        print("\n🚀 启动激进空闲时间优化")
        print("=" * 60)
        print(f"种群大小: {self.population_size}")
        print(f"迭代代数: {self.generations}")
        print(f"变异率: {self.mutation_rate} (激进: {self.aggressive_mutation_rate})")
        print(f"FPS容忍度: {self.fps_tolerance * 100}%")
        print(f"优化目标: 最大化紧凑化后的空闲时间")
        
        # 初始化种群
        population = []
        
        # 1. 添加原始配置
        original = copy.deepcopy(self.original_config)
        self._evaluate_fitness = self._evaluate_fitness_for_idle
        self._evaluate_fitness(original)
        population.append(original)
        print(f"\n原始配置空闲时间: {original.idle_time:.1f}ms")
        
        # 2. 添加多样化的个体
        while len(population) < self.population_size:
            if random.random() < 0.7:  # 70%激进个体
                individual = self._create_random_aggressive_individual()
            else:  # 30%智能个体
                individual = self._create_intelligent_individual()
            self._evaluate_fitness(individual)
            population.append(individual)
        
        # 排序
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0]
        
        print(f"\n初始最佳:")
        print(f"  适应度: {self.best_individual.fitness:.2f}")
        print(f"  空闲时间: {self.best_individual.idle_time:.1f}ms")
        print(f"  FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        
        # 进化过程
        best_idle_time = self.best_individual.idle_time
        stagnation_counter = 0
        
        for generation in range(self.generations):
            # 精英保留
            new_population = population[:self.elite_size]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 多样化选择策略
                if random.random() < 0.7:
                    # 标准交叉变异
                    parent1 = self._tournament_selection(population, tournament_size=3)
                    parent2 = self._tournament_selection(population, tournament_size=3)
                    
                    child1, child2 = self._crossover(parent1, parent2)
                    self._aggressive_mutate(child1)
                    self._aggressive_mutate(child2)
                    
                    new_population.extend([child1, child2])
                else:
                    # 创建全新的激进个体
                    new_individual = self._create_random_aggressive_individual()
                    new_population.append(new_individual)
            
            # 评估新个体
            for ind in new_population[self.elite_size:]:
                self._evaluate_fitness(ind)
            
            # 更新种群
            population = new_population[:self.population_size]
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 检查改进
            current_best = population[0]
            if current_best.idle_time > best_idle_time:
                best_idle_time = current_best.idle_time
                self.best_individual = current_best
                stagnation_counter = 0
                print(f"\n✨ 第{generation}代发现更好解: 空闲时间={best_idle_time:.1f}ms")
            else:
                stagnation_counter += 1
            
            # 定期报告
            if generation % 20 == 0:
                avg_idle = sum(ind.idle_time for ind in population[:10]) / 10
                print(f"\n第{generation}代:")
                print(f"  最佳空闲时间: {self.best_individual.idle_time:.1f}ms")
                print(f"  平均空闲时间(top10): {avg_idle:.1f}ms")
                print(f"  最佳FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
                print(f"  停滞计数: {stagnation_counter}")
            
            # 停滞处理
            if stagnation_counter > 30:
                print(f"\n💉 注入新血液（停滞{stagnation_counter}代）")
                # 替换部分种群
                for i in range(self.population_size // 3, self.population_size):
                    population[i] = self._create_random_aggressive_individual()
                    self._evaluate_fitness(population[i])
                stagnation_counter = 0
            
            # 提前停止条件
            if best_idle_time > self.time_window * 0.3:  # 30%空闲已经很好
                print(f"\n🎯 达到优秀解（空闲时间>{self.time_window * 0.3:.1f}ms），提前停止")
                break
        
        # 应用最佳配置
        self._apply_individual_config(self.best_individual)
        
        print(f"\n🏁 优化完成!")
        print(f"最终最佳空闲时间: {self.best_individual.idle_time:.1f}ms ({self.best_individual.idle_time/self.time_window*100:.1f}%)")
        
        return self.best_individual
    
    def print_idle_optimization_report(self):
        """打印空闲时间优化报告"""
        print("\n" + "=" * 60)
        print("🎯 空闲时间优化报告")
        print("=" * 60)
        
        if not self.best_individual:
            print("❌ 未找到优化解")
            return
        
        print(f"\n📊 最佳个体性能:")
        print(f"  空闲时间: {self.best_individual.idle_time:.1f}ms ({self.best_individual.idle_time/self.time_window*100:.1f}%)")
        print(f"  适应度: {self.best_individual.fitness:.2f}")
        print(f"  FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        print(f"  资源冲突: {self.best_individual.conflict_count}")
        print(f"  资源利用率: {self.best_individual.resource_utilization:.1%}")
        
        print("\n📋 任务配置变化:")
        print("-" * 80)
        print(f"{'任务':<8} {'名称':<15} {'原优先级':<12} {'新优先级':<12} {'运行时':<15} {'分段策略':<20}")
        print("-" * 80)
        
        for task in self.tasks:
            task_id = task.task_id
            orig_priority = self.original_config.task_priorities[task_id]
            new_priority = self.best_individual.task_priorities[task_id]
            new_runtime = self.best_individual.task_runtime_types[task_id]
            new_seg = self.best_individual.task_segmentation_strategies[task_id]
            
            priority_change = ""
            if orig_priority != new_priority:
                if orig_priority.value > new_priority.value:
                    priority_change = "↑"  # 升级
                else:
                    priority_change = "↓"  # 降级
            
            print(f"{task_id:<8} {task.name:<15} {orig_priority.name:<12} "
                  f"{new_priority.name}{priority_change:<11} {new_runtime.value:<15} {new_seg.value:<20}")
        
        print("\n💡 优化策略分析:")
        # 分析优化策略
        priority_changes = defaultdict(int)
        for task in self.tasks:
            orig = self.original_config.task_priorities[task.task_id]
            new = self.best_individual.task_priorities[task.task_id]
            if orig != new:
                if orig.value > new.value:
                    priority_changes['upgrades'] += 1
                else:
                    priority_changes['downgrades'] += 1
        
        print(f"  - 优先级提升: {priority_changes['upgrades']} 个任务")
        print(f"  - 优先级降低: {priority_changes['downgrades']} 个任务")
        
        # 分析低FPS任务
        low_fps_low_priority = 0
        for task in self.tasks:
            if task.fps_requirement <= 10 and self.best_individual.task_priorities[task.task_id] == TaskPriority.LOW:
                low_fps_low_priority += 1
        print(f"  - 低FPS任务降级: {low_fps_low_priority} 个")
