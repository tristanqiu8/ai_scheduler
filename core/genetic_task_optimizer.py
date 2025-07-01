#!/usr/bin/env python3
"""
遗传算法任务优化器
基于real_task的任务定义，使用遗传算法进行智能优化
参考test_simple_optimization.py和dragon4_with_smart_gap.py的import顺序
"""

import sys
import os
import random
import copy
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# 核心导入 (按照test文件的顺序)
from .enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from .scheduler import MultiResourceScheduler
from .task import NNTask
from .models import SubSegment

# 修复导入
from .modular_scheduler_fixes import apply_basic_fixes
from .fixed_validation_and_metrics import validate_schedule_correctly

# 场景导入 - 需要使用绝对路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scenario.real_task import create_real_tasks

# 可视化导入
from viz.elegant_visualization import ElegantSchedulerVisualizer


@dataclass
class GeneticIndividual:
    """遗传算法个体"""
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
    
    def __hash__(self):
        """使个体可哈希"""
        return hash(str(self.task_priorities) + str(self.task_runtime_types))


class GeneticTaskOptimizer:
    """基于遗传算法的任务优化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, tasks: List[NNTask], 
                 time_window: float = 200.0):
        self.scheduler = scheduler
        self.tasks = tasks
        self.time_window = time_window
        
        # 遗传算法参数
        self.population_size = 50
        self.elite_size = 10
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8
        self.generations = 100
        
        # 搜索空间定义
        self.priority_options = list(TaskPriority)
        self.runtime_options = list(RuntimeType)
        self.segmentation_options = list(SegmentationStrategy)
        
        # 缓存原始配置
        self.original_config = self._save_original_config()
        
        # 最佳个体追踪
        self.best_individual: Optional[GeneticIndividual] = None
        self.generation_history: List[Dict] = []
        
    def _save_original_config(self) -> GeneticIndividual:
        """保存原始配置"""
        individual = GeneticIndividual()
        
        for task in self.tasks:
            individual.task_priorities[task.task_id] = task.priority
            individual.task_runtime_types[task.task_id] = task.runtime_type
            individual.task_segmentation_strategies[task.task_id] = task.segmentation_strategy
            individual.task_segmentation_configs[task.task_id] = 0
            
        return individual
    
    def _create_random_individual(self) -> GeneticIndividual:
        """创建随机个体"""
        individual = GeneticIndividual()
        
        for task in self.tasks:
            # 随机优先级（考虑任务特性）
            if task.task_id == "T1":  # MOTR关键任务
                individual.task_priorities[task.task_id] = random.choice([
                    TaskPriority.CRITICAL, TaskPriority.HIGH
                ])
            else:
                individual.task_priorities[task.task_id] = random.choice(self.priority_options)
            
            # 随机运行时类型
            individual.task_runtime_types[task.task_id] = random.choice(self.runtime_options)
            
            # 随机分段策略
            if task.task_id in ["T2", "T3", "T5"]:  # 适合分段的任务
                individual.task_segmentation_strategies[task.task_id] = random.choice([
                    SegmentationStrategy.ADAPTIVE_SEGMENTATION,
                    SegmentationStrategy.CUSTOM_SEGMENTATION,
                    SegmentationStrategy.NO_SEGMENTATION
                ])
            else:
                individual.task_segmentation_strategies[task.task_id] = random.choice([
                    SegmentationStrategy.NO_SEGMENTATION,
                    SegmentationStrategy.ADAPTIVE_SEGMENTATION
                ])
            
            # 随机分段配置
            individual.task_segmentation_configs[task.task_id] = random.randint(0, 3)
            
            # 随机资源分配
            individual.resource_assignments[task.task_id] = self._generate_resource_assignment(task)
            
        return individual
    
    def _create_intelligent_individual(self) -> GeneticIndividual:
        """创建智能初始个体（基于启发式规则）"""
        individual = GeneticIndividual()
        
        for task in self.tasks:
            task_id = task.task_id
            
            # 基于任务特性的智能优先级分配
            if "MOTR" in task.name or task_id == "T1":
                # 关键任务保持高优先级
                individual.task_priorities[task_id] = TaskPriority.CRITICAL
            elif task.fps_requirement >= 20:
                # 高FPS要求的任务给予较高优先级
                individual.task_priorities[task_id] = TaskPriority.HIGH
            elif task.fps_requirement <= 5:
                # 低FPS要求的任务可以降低优先级
                individual.task_priorities[task_id] = TaskPriority.LOW
            else:
                individual.task_priorities[task_id] = TaskPriority.NORMAL
            
            # 基于资源使用的运行时类型选择
            if task.uses_dsp and task.uses_npu:
                # 混合资源任务倾向于DSP_Runtime
                individual.task_runtime_types[task_id] = RuntimeType.DSP_RUNTIME
            elif task.uses_dsp:
                individual.task_runtime_types[task_id] = RuntimeType.DSP_RUNTIME
            else:
                individual.task_runtime_types[task_id] = RuntimeType.ACPU_RUNTIME
            
            # 基于任务复杂度的分段策略
            total_duration = sum(seg.get_duration(40.0) for seg in task.segments)
            if total_duration > 20.0 and len(task.segments) > 2:
                # 复杂任务考虑分段
                individual.task_segmentation_strategies[task_id] = SegmentationStrategy.ADAPTIVE_SEGMENTATION
            else:
                individual.task_segmentation_strategies[task_id] = SegmentationStrategy.NO_SEGMENTATION
            
            # 智能资源分配
            individual.resource_assignments[task_id] = self._intelligent_resource_assignment(task)
            individual.task_segmentation_configs[task_id] = 0
            
        return individual
    
    def _intelligent_resource_assignment(self, task: NNTask) -> Dict[ResourceType, str]:
        """智能资源分配"""
        assignment = {}
        
        # 获取资源列表
        npu_resources = self._get_resource_list(ResourceType.NPU)
        dsp_resources = self._get_resource_list(ResourceType.DSP)
        
        # NPU分配策略：高优先级任务分配到高性能NPU
        if npu_resources and task.uses_npu:
            if task.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                # 优先分配到NPU_0（通常性能更好）
                assignment[ResourceType.NPU] = npu_resources[0] if npu_resources else None
            else:
                # 低优先级任务可以分配到任意NPU
                assignment[ResourceType.NPU] = random.choice(npu_resources)
        
        # DSP分配策略
        if dsp_resources and task.uses_dsp:
            assignment[ResourceType.DSP] = random.choice(dsp_resources)
            
        return assignment
    
    def _get_resource_list(self, res_type: ResourceType) -> List[str]:
        """获取资源列表"""
        resources = self.scheduler.resources.get(res_type, {})
        if isinstance(resources, dict):
            return list(resources.keys())
        elif isinstance(resources, list):
            return [f"{res_type.value}_{i}" for i in range(len(resources))]
        return []
        """创建随机个体"""
        individual = GeneticIndividual()
        
        for task in self.tasks:
            # 随机优先级（考虑任务特性）
            if task.task_id == "T1":  # MOTR关键任务
                individual.task_priorities[task.task_id] = random.choice([
                    TaskPriority.CRITICAL, TaskPriority.HIGH
                ])
            else:
                individual.task_priorities[task.task_id] = random.choice(self.priority_options)
            
            # 随机运行时类型
            individual.task_runtime_types[task.task_id] = random.choice(self.runtime_options)
            
            # 随机分段策略
            if task.task_id in ["T2", "T3", "T5"]:  # 适合分段的任务
                individual.task_segmentation_strategies[task.task_id] = random.choice([
                    SegmentationStrategy.ADAPTIVE_SEGMENTATION,
                    SegmentationStrategy.CUSTOM_SEGMENTATION,
                    SegmentationStrategy.NO_SEGMENTATION
                ])
            else:
                individual.task_segmentation_strategies[task.task_id] = random.choice([
                    SegmentationStrategy.NO_SEGMENTATION,
                    SegmentationStrategy.ADAPTIVE_SEGMENTATION
                ])
            
            # 随机分段配置
            individual.task_segmentation_configs[task.task_id] = random.randint(0, 3)
            
            # 随机资源分配
            individual.resource_assignments[task.task_id] = self._generate_resource_assignment(task)
            
        return individual
    
    def _generate_resource_assignment(self, task: NNTask) -> Dict[ResourceType, str]:
        """生成资源分配方案"""
        assignment = {}
        
        # NPU分配
        npu_resources = self.scheduler.resources.get(ResourceType.NPU, {})
        if isinstance(npu_resources, dict):
            npu_units = list(npu_resources.keys())
        elif isinstance(npu_resources, list):
            npu_units = [f"NPU_{i}" for i in range(len(npu_resources))]
        else:
            npu_units = []
            
        if npu_units and task.uses_npu:
            assignment[ResourceType.NPU] = random.choice(npu_units)
        
        # DSP分配
        dsp_resources = self.scheduler.resources.get(ResourceType.DSP, {})
        if isinstance(dsp_resources, dict):
            dsp_units = list(dsp_resources.keys())
        elif isinstance(dsp_resources, list):
            dsp_units = [f"DSP_{i}" for i in range(len(dsp_resources))]
        else:
            dsp_units = []
            
        if dsp_units and task.uses_dsp:
            assignment[ResourceType.DSP] = random.choice(dsp_units)
            
        return assignment
    
    def _apply_individual_config(self, individual: GeneticIndividual):
        """应用个体配置到调度器"""
        for task in self.tasks:
            task_id = task.task_id
            
            # 应用优先级
            task.priority = individual.task_priorities.get(task_id, task.priority)
            
            # 应用运行时类型
            task.runtime_type = individual.task_runtime_types.get(task_id, task.runtime_type)
            
            # 应用分段策略
            task.segmentation_strategy = individual.task_segmentation_strategies.get(
                task_id, task.segmentation_strategy
            )
            
            # 应用资源分配建议（这个需要在调度时考虑）
            if task_id in individual.resource_assignments:
                task._preferred_resources = individual.resource_assignments[task_id]
    
    def _evaluate_fitness(self, individual: GeneticIndividual) -> float:
        """评估个体适应度"""
        # 应用配置
        self._apply_individual_config(individual)
        
        # 清空调度历史
        self.scheduler.schedule_history.clear()
        
        # 运行调度
        try:
            results = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
            
            # 验证调度结果
            is_valid, conflicts = validate_schedule_correctly(self.scheduler)
            individual.conflict_count = len(conflicts)
            
            # 计算FPS满足率
            task_counts = defaultdict(int)
            for event in self.scheduler.schedule_history:
                task_counts[event.task_id] += 1
            
            satisfied_tasks = 0
            total_fps_rate = 0.0
            total_latency = 0.0
            
            for task in self.tasks:
                count = task_counts[task.task_id]
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                
                if expected > 0:
                    fps_rate = min(1.0, count / expected)
                    total_fps_rate += fps_rate
                    
                    if fps_rate >= 0.95:
                        satisfied_tasks += 1
                
                # 计算平均延迟
                if task.schedule_info:
                    total_latency += task.schedule_info.actual_latency
            
            individual.fps_satisfaction_rate = total_fps_rate / len(self.tasks)
            individual.avg_latency = total_latency / len(self.tasks) if self.tasks else 0
            
            # 计算资源利用率
            resource_utilization = self._calculate_resource_utilization()
            individual.resource_utilization = resource_utilization
            
            # 计算适应度（多目标优化）
            fitness = 0.0
            
            # 1. 无冲突是最重要的
            if individual.conflict_count == 0:
                fitness += 1000.0
            else:
                fitness -= individual.conflict_count * 100.0
            
            # 2. FPS满足率
            fitness += individual.fps_satisfaction_rate * 500.0
            
            # 3. 资源利用率
            fitness += resource_utilization * 200.0
            
            # 4. 低延迟奖励
            if individual.avg_latency < 50:
                fitness += 100.0
            
            # 5. 关键任务优先级正确性
            if individual.task_priorities.get("T1") == TaskPriority.CRITICAL:
                fitness += 50.0
                
            # 6. 合理的运行时类型选择
            runtime_score = 0
            for task in self.tasks:
                if task.uses_dsp and individual.task_runtime_types.get(task.task_id) == RuntimeType.DSP_RUNTIME:
                    runtime_score += 10
                elif not task.uses_dsp and individual.task_runtime_types.get(task.task_id) == RuntimeType.ACPU_RUNTIME:
                    runtime_score += 10
            fitness += runtime_score
            
            # 7. 分段策略合理性
            segmentation_score = 0
            for task in self.tasks:
                total_duration = sum(seg.get_duration(40.0) for seg in task.segments)
                if total_duration > 20.0 and individual.task_segmentation_strategies.get(task.task_id) != SegmentationStrategy.NO_SEGMENTATION:
                    segmentation_score += 5
                elif total_duration <= 20.0 and individual.task_segmentation_strategies.get(task.task_id) == SegmentationStrategy.NO_SEGMENTATION:
                    segmentation_score += 5
            fitness += segmentation_score
                
        except Exception as e:
            print(f"评估失败: {e}")
            fitness = -1000.0
            
        individual.fitness = fitness
        return fitness
    
    def _calculate_resource_utilization(self) -> float:
        """计算资源利用率"""
        total_util = 0.0
        resource_count = 0
        
        for res_type, resources in self.scheduler.resources.items():
            # 处理resources可能是列表或字典的情况
            if isinstance(resources, dict):
                resource_items = resources.items()
            elif isinstance(resources, list):
                resource_items = [(f"{res_type.value}_{i}", res) for i, res in enumerate(resources)]
            else:
                continue
                
            for res_id, resource in resource_items:
                busy_time = 0.0
                last_end = 0.0
                
                # 计算资源忙碌时间
                for event in sorted(self.scheduler.schedule_history, key=lambda x: x.start_time):
                    if event.assigned_resources.get(res_type) == res_id:
                        if event.start_time >= last_end:
                            busy_time += event.end_time - event.start_time
                            last_end = event.end_time
                
                utilization = busy_time / self.time_window if self.time_window > 0 else 0
                total_util += utilization
                resource_count += 1
        
        return total_util / resource_count if resource_count > 0 else 0
    
    def _tournament_selection(self, population: List[GeneticIndividual], 
                            tournament_size: int = 3) -> GeneticIndividual:
        """锦标赛选择"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: GeneticIndividual, 
                   parent2: GeneticIndividual) -> Tuple[GeneticIndividual, GeneticIndividual]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
        
        child1 = GeneticIndividual()
        child2 = GeneticIndividual()
        
        # 对每个任务进行交叉
        for task in self.tasks:
            task_id = task.task_id
            
            if random.random() < 0.5:
                # 子代1继承父代1的基因
                child1.task_priorities[task_id] = parent1.task_priorities.get(task_id)
                child1.task_runtime_types[task_id] = parent1.task_runtime_types.get(task_id)
                child1.task_segmentation_strategies[task_id] = parent1.task_segmentation_strategies.get(task_id)
                child1.task_segmentation_configs[task_id] = parent1.task_segmentation_configs.get(task_id)
                child1.resource_assignments[task_id] = parent1.resource_assignments.get(task_id, {})
                
                # 子代2继承父代2的基因
                child2.task_priorities[task_id] = parent2.task_priorities.get(task_id)
                child2.task_runtime_types[task_id] = parent2.task_runtime_types.get(task_id)
                child2.task_segmentation_strategies[task_id] = parent2.task_segmentation_strategies.get(task_id)
                child2.task_segmentation_configs[task_id] = parent2.task_segmentation_configs.get(task_id)
                child2.resource_assignments[task_id] = parent2.resource_assignments.get(task_id, {})
            else:
                # 反向继承
                child1.task_priorities[task_id] = parent2.task_priorities.get(task_id)
                child1.task_runtime_types[task_id] = parent2.task_runtime_types.get(task_id)
                child1.task_segmentation_strategies[task_id] = parent2.task_segmentation_strategies.get(task_id)
                child1.task_segmentation_configs[task_id] = parent2.task_segmentation_configs.get(task_id)
                child1.resource_assignments[task_id] = parent2.resource_assignments.get(task_id, {})
                
                child2.task_priorities[task_id] = parent1.task_priorities.get(task_id)
                child2.task_runtime_types[task_id] = parent1.task_runtime_types.get(task_id)
                child2.task_segmentation_strategies[task_id] = parent1.task_segmentation_strategies.get(task_id)
                child2.task_segmentation_configs[task_id] = parent1.task_segmentation_configs.get(task_id)
                child2.resource_assignments[task_id] = parent1.resource_assignments.get(task_id, {})
        
        return child1, child2
    
    def _mutate(self, individual: GeneticIndividual):
        """变异操作"""
        for task in self.tasks:
            task_id = task.task_id
            
            # 优先级变异
            if random.random() < self.mutation_rate:
                if task_id == "T1":  # 保持关键任务的高优先级
                    individual.task_priorities[task_id] = random.choice([
                        TaskPriority.CRITICAL, TaskPriority.HIGH
                    ])
                else:
                    individual.task_priorities[task_id] = random.choice(self.priority_options)
            
            # 运行时类型变异
            if random.random() < self.mutation_rate:
                individual.task_runtime_types[task_id] = random.choice(self.runtime_options)
            
            # 分段策略变异
            if random.random() < self.mutation_rate:
                if task_id in ["T2", "T3", "T5"]:
                    individual.task_segmentation_strategies[task_id] = random.choice([
                        SegmentationStrategy.ADAPTIVE_SEGMENTATION,
                        SegmentationStrategy.CUSTOM_SEGMENTATION,
                        SegmentationStrategy.NO_SEGMENTATION
                    ])
                else:
                    individual.task_segmentation_strategies[task_id] = random.choice([
                        SegmentationStrategy.NO_SEGMENTATION,
                        SegmentationStrategy.ADAPTIVE_SEGMENTATION
                    ])
            
            # 资源分配变异
            if random.random() < self.mutation_rate:
                individual.resource_assignments[task_id] = self._generate_resource_assignment(
                    next(t for t in self.tasks if t.task_id == task_id)
                )
    
    def optimize(self) -> GeneticIndividual:
        """运行遗传算法优化"""
        print("\n🧬 启动遗传算法优化")
        print("=" * 60)
        print(f"种群大小: {self.population_size}")
        print(f"精英个体: {self.elite_size}")
        print(f"变异率: {self.mutation_rate}")
        print(f"交叉率: {self.crossover_rate}")
        print(f"迭代代数: {self.generations}")
        
        # 初始化种群
        population = []
        
        # 1. 添加原始配置
        original = copy.deepcopy(self.original_config)
        self._evaluate_fitness(original)
        population.append(original)
        
        # 2. 添加几个智能个体（基于启发式规则）
        for _ in range(min(5, self.population_size // 4)):
            intelligent = self._create_intelligent_individual()
            self._evaluate_fitness(intelligent)
            population.append(intelligent)
        
        # 3. 其余为随机个体
        while len(population) < self.population_size:
            individual = self._create_random_individual()
            self._evaluate_fitness(individual)
            population.append(individual)
        
        # 排序种群
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0]
        
        print(f"\n初始种群最佳适应度: {self.best_individual.fitness:.2f}")
        print(f"  - FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        print(f"  - 资源冲突: {self.best_individual.conflict_count}")
        
        # 进化过程
        no_improvement_count = 0
        best_fitness_history = []
        
        for generation in range(self.generations):
            # 精英保留
            new_population = population[:self.elite_size]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # 交叉
                child1, child2 = self._crossover(parent1, parent2)
                
                # 变异
                self._mutate(child1)
                self._mutate(child2)
                
                # 评估适应度
                self._evaluate_fitness(child1)
                self._evaluate_fitness(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            population = new_population[:self.population_size]
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 更新最佳个体
            if population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # 记录进化历史
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            self.generation_history.append({
                'generation': generation,
                'best_fitness': self.best_individual.fitness,
                'avg_fitness': avg_fitness,
                'best_fps_rate': self.best_individual.fps_satisfaction_rate,
                'best_conflicts': self.best_individual.conflict_count
            })
            
            best_fitness_history.append(self.best_individual.fitness)
            
            # 定期报告进度
            if generation % 10 == 0:
                print(f"\n第 {generation} 代:")
                print(f"  最佳适应度: {self.best_individual.fitness:.2f}")
                print(f"  平均适应度: {avg_fitness:.2f}")
                print(f"  FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
                print(f"  资源冲突: {self.best_individual.conflict_count}")
                
            # 早停条件
            # 1. 找到理想解
            if self.best_individual.conflict_count == 0 and \
               self.best_individual.fps_satisfaction_rate >= 0.99:
                print(f"\n✅ 找到理想解，提前停止进化（第{generation}代）")
                break
                
            # 2. 长时间无改进
            if no_improvement_count >= 20:
                print(f"\n⚠️ 连续{no_improvement_count}代无改进，提前停止")
                break
                
            # 3. 适应度收敛
            if len(best_fitness_history) >= 10:
                recent_fitness = best_fitness_history[-10:]
                fitness_variance = np.var(recent_fitness)
                if fitness_variance < 0.01:
                    print(f"\n⚠️ 适应度已收敛（方差={fitness_variance:.4f}），提前停止")
                    break
        
        # 应用最佳配置
        self._apply_individual_config(self.best_individual)
        
        return self.best_individual
    
    def print_optimization_report(self):
        """打印优化报告"""
        print("\n" + "=" * 60)
        print("🎯 遗传算法优化报告")
        print("=" * 60)
        
        if not self.best_individual:
            print("❌ 未找到优化解")
            return
        
        print(f"\n最佳个体适应度: {self.best_individual.fitness:.2f}")
        print(f"FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        print(f"资源冲突数: {self.best_individual.conflict_count}")
        print(f"资源利用率: {self.best_individual.resource_utilization:.1%}")
        print(f"平均延迟: {self.best_individual.avg_latency:.1f}ms")
        
        print("\n📊 任务配置优化结果:")
        print("-" * 60)
        print(f"{'任务ID':<8} {'优先级变化':<20} {'运行时变化':<20} {'分段策略':<20}")
        print("-" * 60)
        
        for task in self.tasks:
            task_id = task.task_id
            
            # 优先级变化
            orig_priority = self.original_config.task_priorities[task_id]
            new_priority = self.best_individual.task_priorities[task_id]
            priority_change = f"{orig_priority.name} → {new_priority.name}" if orig_priority != new_priority else "不变"
            
            # 运行时变化
            orig_runtime = self.original_config.task_runtime_types[task_id]
            new_runtime = self.best_individual.task_runtime_types[task_id]
            runtime_change = f"{orig_runtime.value} → {new_runtime.value}" if orig_runtime != new_runtime else "不变"
            
            # 分段策略
            seg_strategy = self.best_individual.task_segmentation_strategies[task_id].name
            
            print(f"{task_id:<8} {priority_change:<20} {runtime_change:<20} {seg_strategy:<20}")
        
        # 进化曲线
        if self.generation_history:
            print(f"\n📈 进化过程摘要:")
            print(f"  初始适应度: {self.generation_history[0]['best_fitness']:.2f}")
            print(f"  最终适应度: {self.generation_history[-1]['best_fitness']:.2f}")
            print(f"  改进幅度: {(self.generation_history[-1]['best_fitness'] - self.generation_history[0]['best_fitness']):.2f}")
            print(f"  收敛代数: {len(self.generation_history)}")


def run_genetic_optimization(scheduler: MultiResourceScheduler, tasks: List[NNTask], 
                           time_window: float = 200.0) -> GeneticIndividual:
    """运行遗传算法优化的便捷函数"""
    
    optimizer = GeneticTaskOptimizer(scheduler, tasks, time_window)
    best_individual = optimizer.optimize()
    optimizer.print_optimization_report()
    
    return best_individual


if __name__ == "__main__":
    """测试遗传算法优化器"""
    
    print("🧬 遗传算法任务优化器测试")
    print("=" * 80)
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=120.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    # 应用基础修复
    fix_manager = apply_basic_fixes(scheduler)
    
    # 创建任务
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    # 运行遗传算法优化
    best_solution = run_genetic_optimization(scheduler, tasks, time_window=200.0)
    
    # 生成可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt(save_filename="genetic_optimized_schedule.png")
        print("\n✅ 优化结果可视化已保存到 genetic_optimized_schedule.png")
    except Exception as e:
        print(f"\n⚠️ 可视化生成失败: {e}")
