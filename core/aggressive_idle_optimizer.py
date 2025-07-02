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
        self.fps_tolerance = 0.90            # FPS容忍度（修改为95%）
        self.low_fps_tolerance = 0.85        # 低FPS任务的容忍度
        
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
        """针对空闲时间优化的适应度函数 - 更激进版本"""
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
            low_fps_tasks_satisfied = 0
            high_fps_tasks_satisfied = 0
            
            for task in self.tasks:
                count = task_counts[task.task_id]
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                
                if expected > 0:
                    fps_rate = count / expected
                    total_fps_satisfaction += fps_rate
                    
                    # 检查低FPS任务（使用低容忍度）
                    if task.fps_requirement <= 10:
                        if fps_rate >= self.low_fps_tolerance:
                            low_fps_tasks_satisfied += 1
                    
                    # 检查高FPS任务（使用正常容忍度）
                    if task.fps_requirement >= 25:
                        if fps_rate >= self.fps_tolerance:
                            high_fps_tasks_satisfied += 1
                        elif task.priority == TaskPriority.CRITICAL and fps_rate < self.fps_tolerance * 0.9:
                            critical_fps_violation = True
            
            individual.fps_satisfaction_rate = total_fps_satisfaction / len(self.tasks)
            
            # 新的激进适应度计算
            fitness = 0.0
            
            # 1. 空闲时间是绝对主导因素（权重大幅提高）
            fitness += idle_time * 50.0  # 从10提高到50
            
            # 2. 对比基线的空闲时间改进
            if self.baseline_performance and 'baseline_idle' in self.baseline_performance:
                idle_improvement = idle_time - self.baseline_performance['baseline_idle']
                if idle_improvement > 0:
                    fitness += idle_improvement * 100  # 每ms改进100分
                else:
                    fitness += idle_improvement * 20   # 退化的惩罚较轻
            
            # 3. 冲突惩罚（降低权重）
            if individual.conflict_count > 0:
                fitness -= individual.conflict_count * 50  # 从100降到50
            
            # 4. FPS要求（使用配置的容忍度）
            if individual.fps_satisfaction_rate >= self.fps_tolerance:
                fitness += 300  # 满足FPS要求的奖励
            else:
                # 根据差距计算惩罚
                fps_gap = self.fps_tolerance - individual.fps_satisfaction_rate
                fps_penalty = fps_gap * 1000  # 加大惩罚力度
                fitness -= fps_penalty
            
            # 5. 激励牺牲低优先级任务
            for task in self.tasks:
                task_id = task.task_id
                count = task_counts.get(task_id, 0)
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                
                # 如果低FPS任务执行次数减少但仍满足低容忍度，奖励
                if task.fps_requirement <= 10 and expected > 0:
                    fps_rate = count / expected
                    if fps_rate >= self.low_fps_tolerance and fps_rate < 1.0:
                        fitness += 20
                
                # 如果低优先级任务被降级，奖励
                if individual.task_priorities.get(task_id) == TaskPriority.LOW:
                    if task.fps_requirement <= 10:
                        fitness += 30
            
            # 6. 不要过度惩罚关键任务违规
            if critical_fps_violation:
                fitness -= 200  # 适度惩罚
                
        except Exception as e:
            print(f"评估失败: {e}")
            fitness = -10000.0
            individual.idle_time = 0
            
        individual.fitness = fitness
        return fitness
    
    def _estimate_idle_time(self) -> float:
        """实际运行紧凑化来测量空闲时间"""
        if not self.scheduler.schedule_history:
            return self.time_window
        
        # 导入紧凑化器
        try:
            from .debug_compactor import DebugCompactor
        except ImportError:
            # 如果无法导入，使用简单估算
            first_window_events = [e for e in self.scheduler.schedule_history 
                                  if e.start_time < self.time_window]
            if not first_window_events:
                return self.time_window
            
            # 计算实际占用时间
            total_busy = 0
            for event in first_window_events:
                total_busy += (event.end_time - event.start_time)
            
            # 粗略估算：假设可以压缩掉30%的空隙
            return self.time_window - total_busy * 0.7
        
        # 使用实际的紧凑化器
        import copy
        original_history = copy.deepcopy(self.scheduler.schedule_history)
        
        compactor = DebugCompactor(self.scheduler, self.time_window)
        try:
            _, idle_time = compactor.simple_compact()
            # 恢复原始历史
            self.scheduler.schedule_history = original_history
            return idle_time
        except:
            # 如果紧凑化失败，返回保守估计
            self.scheduler.schedule_history = original_history
            return 0.0
    
    def _create_extreme_individual(self) -> GeneticIndividual:
        """创建极端的个体 - 最大化空闲时间"""
        individual = GeneticIndividual()
        
        for task in self.tasks:
            task_id = task.task_id
            
            # 极端策略1：所有低FPS任务都降为最低优先级
            if task.fps_requirement <= 10:
                individual.task_priorities[task_id] = TaskPriority.LOW
            # 极端策略2：只有最高FPS的任务保持高优先级
            elif task.fps_requirement >= 50:
                individual.task_priorities[task_id] = TaskPriority.CRITICAL
            else:
                # 其他任务随机低优先级
                individual.task_priorities[task_id] = random.choice([
                    TaskPriority.LOW, TaskPriority.NORMAL
                ])
            
            # 极端的运行时分配
            if random.random() < 0.5:
                # 50%概率使用"错误"的运行时
                if task.uses_dsp:
                    individual.task_runtime_types[task_id] = RuntimeType.ACPU_RUNTIME
                else:
                    individual.task_runtime_types[task_id] = RuntimeType.DSP_RUNTIME
            else:
                individual.task_runtime_types[task_id] = random.choice(self.runtime_options)
            
            # 激进的分段策略
            if task_id in ["T2", "T3"]:
                # YOLO任务强制分段
                individual.task_segmentation_strategies[task_id] = SegmentationStrategy.FORCED_SEGMENTATION
            else:
                # 随机极端策略
                individual.task_segmentation_strategies[task_id] = random.choice([
                    SegmentationStrategy.NO_SEGMENTATION,
                    SegmentationStrategy.FORCED_SEGMENTATION
                ])
            
            individual.task_segmentation_configs[task_id] = random.randint(0, 4)
            
        return individual
    
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
        """针对空闲时间的优化 - 更激进版本"""
        print("\n🚀 启动激进空闲时间优化")
        print("=" * 60)
        print(f"种群大小: {self.population_size}")
        print(f"迭代代数: {self.generations}")
        print(f"变异率: {self.mutation_rate} (激进: {self.aggressive_mutation_rate})")
        print(f"FPS容忍度: {self.fps_tolerance * 100}%")
        print(f"优化目标: 最大化紧凑化后的空闲时间")
        
        # 初始化种群
        population = []
        
        # 保存所有满足FPS要求的个体
        self.fps_compliant_individuals = []
        
        # 1. 添加原始配置并记录基线空闲时间
        original = copy.deepcopy(self.original_config)
        self._evaluate_fitness = self._evaluate_fitness_for_idle
        self._evaluate_fitness(original)
        population.append(original)
        
        # 记录基线空闲时间用于比较
        if self.baseline_performance:
            self.baseline_performance['baseline_idle'] = original.idle_time
        print(f"\n原始配置空闲时间: {original.idle_time:.1f}ms")
        
        # 检查原始配置是否满足FPS要求
        if original.fps_satisfaction_rate >= self.fps_tolerance:
            self.fps_compliant_individuals.append(copy.deepcopy(original))
            print(f"  ✓ 原始配置满足FPS要求 ({original.fps_satisfaction_rate:.1%})")
        
        # 2. 添加极端个体（专门为最大化空闲时间设计）
        print("创建极端个体...")
        for i in range(20):  # 20%极端个体
            extreme = self._create_extreme_individual()
            self._evaluate_fitness(extreme)
            population.append(extreme)
            if extreme.idle_time > original.idle_time:
                print(f"  极端个体{i}: 空闲时间={extreme.idle_time:.1f}ms")
            # 检查是否满足FPS要求
            if extreme.fps_satisfaction_rate >= self.fps_tolerance:
                self.fps_compliant_individuals.append(copy.deepcopy(extreme))
        
        # 3. 添加激进个体
        while len(population) < self.population_size * 0.8:
            individual = self._create_random_aggressive_individual()
            self._evaluate_fitness(individual)
            population.append(individual)
            # 检查是否满足FPS要求
            if individual.fps_satisfaction_rate >= self.fps_tolerance:
                self.fps_compliant_individuals.append(copy.deepcopy(individual))
        
        # 4. 添加一些智能个体
        while len(population) < self.population_size:
            individual = self._create_intelligent_individual()
            # 但是要修改使其更激进
            for task in self.tasks:
                if task.fps_requirement <= 10 and random.random() < 0.7:
                    individual.task_priorities[task.task_id] = TaskPriority.LOW
            self._evaluate_fitness(individual)
            population.append(individual)
            # 检查是否满足FPS要求
            if individual.fps_satisfaction_rate >= self.fps_tolerance:
                self.fps_compliant_individuals.append(copy.deepcopy(individual))
        
        # 排序
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0]
        
        # 找出满足FPS要求的最佳个体
        if self.fps_compliant_individuals:
            self.best_fps_compliant = max(self.fps_compliant_individuals, key=lambda x: x.idle_time)
            print(f"\n初始最佳（满足FPS）:")
            print(f"  空闲时间: {self.best_fps_compliant.idle_time:.1f}ms")
            print(f"  FPS满足率: {self.best_fps_compliant.fps_satisfaction_rate:.1%}")
        else:
            self.best_fps_compliant = None
            print("\n⚠️ 警告：初始种群中没有满足FPS要求的个体")
        
        print(f"\n初始最佳（总体）:")
        print(f"  适应度: {self.best_individual.fitness:.2f}")
        print(f"  空闲时间: {self.best_individual.idle_time:.1f}ms")
        print(f"  FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        
        # 进化过程
        best_idle_time = self.best_individual.idle_time
        best_compliant_idle_time = self.best_fps_compliant.idle_time if self.best_fps_compliant else 0
        stagnation_counter = 0
        
        for generation in range(self.generations):
            # 精英保留（但更少）
            new_population = population[:self.elite_size]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                strategy = random.random()
                
                if strategy < 0.3:  # 30% 极端个体
                    new_individual = self._create_extreme_individual()
                elif strategy < 0.6:  # 30% 交叉变异
                    parent1 = self._tournament_selection(population[:20], tournament_size=2)
                    parent2 = self._tournament_selection(population[:20], tournament_size=2)
                    
                    child1, child2 = self._crossover(parent1, parent2)
                    self._aggressive_mutate(child1)
                    self._aggressive_mutate(child2)
                    
                    new_population.extend([child1, child2])
                    continue
                else:  # 40% 新的激进个体
                    new_individual = self._create_random_aggressive_individual()
                
                new_population.append(new_individual)
            
            # 评估新个体
            for ind in new_population[self.elite_size:]:
                self._evaluate_fitness(ind)
                # 检查是否满足FPS要求
                if ind.fps_satisfaction_rate >= self.fps_tolerance:
                    self.fps_compliant_individuals.append(copy.deepcopy(ind))
            
            # 更新种群
            population = new_population[:self.population_size]
            population.sort(key=lambda x: x.idle_time, reverse=True)  # 按空闲时间排序！
            
            # 检查改进
            current_best = population[0]
            if current_best.idle_time > best_idle_time:
                best_idle_time = current_best.idle_time
                self.best_individual = current_best
                stagnation_counter = 0
                print(f"\n✨ 第{generation}代发现更好解: 空闲时间={best_idle_time:.1f}ms (FPS={current_best.fps_satisfaction_rate:.1%})")
            else:
                stagnation_counter += 1
            
            # 检查满足FPS要求的最佳个体
            if self.fps_compliant_individuals:
                current_best_compliant = max(self.fps_compliant_individuals, key=lambda x: x.idle_time)
                if current_best_compliant.idle_time > best_compliant_idle_time:
                    best_compliant_idle_time = current_best_compliant.idle_time
                    self.best_fps_compliant = current_best_compliant
                    print(f"  ✅ 满足FPS要求的新最佳: 空闲时间={best_compliant_idle_time:.1f}ms")
            
            # 定期报告
            if generation % 10 == 0:
                top_idle = [ind.idle_time for ind in population[:10]]
                avg_idle = sum(top_idle) / len(top_idle)
                max_idle = max(top_idle)
                compliant_count = len([ind for ind in population if ind.fps_satisfaction_rate >= self.fps_tolerance])
                
                print(f"\n第{generation}代:")
                print(f"  最佳空闲时间: {self.best_individual.idle_time:.1f}ms")
                print(f"  Top10平均: {avg_idle:.1f}ms, 最大: {max_idle:.1f}ms")
                print(f"  最佳FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
                print(f"  满足FPS要求的个体数: {compliant_count}/{self.population_size}")
                if self.best_fps_compliant:
                    print(f"  满足FPS的最佳空闲时间: {self.best_fps_compliant.idle_time:.1f}ms")
                print(f"  停滞计数: {stagnation_counter}")
            
            # 停滞处理 - 更激进
            if stagnation_counter > 20:  # 更快注入新血
                print(f"\n💉 激进注入新血液（停滞{stagnation_counter}代）")
                # 保留最好的几个，其余全部替换为极端个体
                for i in range(3, self.population_size):
                    if i % 2 == 0:
                        population[i] = self._create_extreme_individual()
                    else:
                        population[i] = self._create_random_aggressive_individual()
                    self._evaluate_fitness(population[i])
                    # 检查是否满足FPS要求
                    if population[i].fps_satisfaction_rate >= self.fps_tolerance:
                        self.fps_compliant_individuals.append(copy.deepcopy(population[i]))
                stagnation_counter = 0
            
            # 提前停止条件
            if self.best_fps_compliant and self.best_fps_compliant.idle_time > self.time_window * 0.4:  # 40%空闲
                print(f"\n🎯 达到优秀解（满足FPS的空闲时间>{self.time_window * 0.4:.1f}ms），提前停止")
                break
        
        # 选择最终结果：优先选择满足FPS要求的最佳个体
        if self.best_fps_compliant:
            print(f"\n✅ 找到满足FPS要求的最佳解")
            self.best_individual = self.best_fps_compliant
        else:
            print(f"\n⚠️ 警告：没有找到满足FPS要求（{self.fps_tolerance*100}%）的解，返回最佳空闲时间解")
        
        # 应用最佳配置
        self._apply_individual_config(self.best_individual)
        
        print(f"\n🏁 优化完成!")
        print(f"最终最佳空闲时间: {self.best_individual.idle_time:.1f}ms ({self.best_individual.idle_time/self.time_window*100:.1f}%)")
        print(f"最终FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        print(f"是否满足FPS要求: {'✅ 是' if self.best_individual.fps_satisfaction_rate >= self.fps_tolerance else '❌ 否'}")
        
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
