#!/usr/bin/env python3
"""
遗传算法优化任务优先级配置
使用遗传算法搜索最优的任务优先级配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enhanced_launcher import EnhancedTaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority, SegmentationStrategy
from core.evaluator import PerformanceEvaluator
from scenario.hybrid_task import create_real_tasks
import numpy as np
import random
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from collections import defaultdict
import concurrent.futures
from copy import deepcopy


@dataclass
class Individual:
    """个体（一个优先级配置方案）"""
    genes: Dict[str, int]  # task_id -> priority_level (0-3)
    fitness: float = 0.0
    fps_satisfaction_rate: float = 0.0
    latency_satisfaction_rate: float = 0.0
    avg_latency: float = float('inf')
    
    def to_priority_config(self) -> Dict[str, TaskPriority]:
        """转换为优先级配置"""
        priority_map = [TaskPriority.LOW, TaskPriority.NORMAL, 
                       TaskPriority.HIGH, TaskPriority.CRITICAL]
        return {task_id: priority_map[level] for task_id, level in self.genes.items()}


class GeneticPriorityOptimizer:
    """遗传算法优先级优化器"""
    
    def __init__(self, tasks, time_window=1000.0, segment_mode=True):
        self.tasks = tasks
        self.time_window = time_window
        self.segment_mode = segment_mode
        
        # 任务ID列表
        self.task_ids = [task.task_id for task in tasks]
        
        # 分析任务特征
        self.task_features = self._analyze_task_features()
        
        # 遗传算法参数
        self.population_size = 20
        self.elite_size = 4
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        
        # 最佳个体历史
        self.best_individuals = []
        
    def _analyze_task_features(self) -> Dict[str, dict]:
        """分析任务特征"""
        features = {}
        
        # 计算被依赖次数
        dependency_count = defaultdict(int)
        for task in self.tasks:
            for dep in task.dependencies:
                dependency_count[dep] += 1
        
        # 计算最大FPS用于归一化
        max_fps = max(task.fps_requirement for task in self.tasks)
        
        for task in self.tasks:
            # 估算执行时间
            bandwidth_map = {ResourceType.NPU: 40.0, ResourceType.DSP: 40.0}
            estimated_duration = task.estimate_duration(bandwidth_map)
            
            features[task.task_id] = {
                'name': task.name,
                'fps_requirement': task.fps_requirement,
                'fps_normalized': task.fps_requirement / max_fps if max_fps > 0 else 0,
                'latency_requirement': task.latency_requirement,
                'latency_strictness': estimated_duration / task.latency_requirement 
                                    if task.latency_requirement > 0 else 0,
                'dependency_count': dependency_count[task.task_id],
                'has_dependencies': len(task.dependencies) > 0,
                'is_mixed': task.uses_npu and task.uses_dsp,
                'num_segments': len(task.segments)
            }
        
        return features
    
    def create_individual(self, guided=True) -> Individual:
        """创建个体"""
        genes = {}
        
        if guided and random.random() < 0.7:  # 70%概率使用启发式
            # 基于任务特征的启发式初始化
            for task_id in self.task_ids:
                features = self.task_features[task_id]
                
                # 计算推荐优先级
                score = 0.0
                score += features['dependency_count'] * 0.3
                score += features['fps_normalized'] * 0.2
                score += features['latency_strictness'] * 0.3
                score += (0.2 if features['is_mixed'] else 0.0)
                score += (0.1 if features['num_segments'] > 5 else 0.0)
                
                # 映射到优先级等级
                if score > 0.7:
                    genes[task_id] = 3  # CRITICAL
                elif score > 0.5:
                    genes[task_id] = 2  # HIGH
                elif score > 0.3:
                    genes[task_id] = 1  # NORMAL
                else:
                    genes[task_id] = 0  # LOW
                
                # 添加随机扰动
                if random.random() < 0.2:
                    genes[task_id] = max(0, min(3, genes[task_id] + random.randint(-1, 1)))
        else:
            # 完全随机初始化
            for task_id in self.task_ids:
                genes[task_id] = random.randint(0, 3)
        
        return Individual(genes)
    
    def evaluate_individual(self, individual: Individual) -> None:
        """评估个体适应度"""
        # 转换为优先级配置
        priority_config = individual.to_priority_config()
        
        # 应用配置到任务
        task_copy = deepcopy(self.tasks)
        for task in task_copy:
            task.priority = priority_config[task.task_id]
        
        # 创建调度环境
        queue_manager = ResourceQueueManager()
        queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
        queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
        
        tracer = ScheduleTracer(queue_manager)
        
        if self.segment_mode:
            launcher = EnhancedTaskLauncher(queue_manager, tracer)
        else:
            launcher = TaskLauncher(queue_manager, tracer)
        
        # 注册任务
        for task in task_copy:
            launcher.register_task(task)
        
        # 执行调度
        plan = launcher.create_launch_plan(self.time_window, "balanced")
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, self.time_window, segment_mode=self.segment_mode)
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(self.time_window, plan.events)
        
        # 计算满足率
        fps_satisfied = 0
        latency_satisfied = 0
        total_latency_score = 0
        
        for task_id, task_metrics in evaluator.task_metrics.items():
            if task_metrics.fps_satisfaction:
                fps_satisfied += 1
            if task_metrics.latency_satisfaction_rate > 0.9:
                latency_satisfied += 1
            
            # 计算延迟分数（越接近要求越好）
            if task_metrics.latency_requirement > 0:
                latency_ratio = task_metrics.avg_latency / task_metrics.latency_requirement
                latency_score = 1.0 / (1.0 + max(0, latency_ratio - 1.0))
                total_latency_score += latency_score
        
        num_tasks = len(evaluator.task_metrics)
        fps_rate = fps_satisfied / num_tasks if num_tasks > 0 else 0
        latency_rate = latency_satisfied / num_tasks if num_tasks > 0 else 0
        avg_latency_score = total_latency_score / num_tasks if num_tasks > 0 else 0
        
        # 计算综合适应度
        # 优先满足FPS和延迟要求，同时考虑资源利用率
        fitness = (
            fps_rate * 0.35 +                    # FPS满足率
            latency_rate * 0.35 +                 # 延迟满足率
            avg_latency_score * 0.2 +             # 平均延迟得分
            (metrics.avg_npu_utilization / 100) * 0.05 +  # NPU利用率
            (metrics.avg_dsp_utilization / 100) * 0.05    # DSP利用率
        )
        
        # 更新个体信息
        individual.fitness = fitness
        individual.fps_satisfaction_rate = fps_rate
        individual.latency_satisfaction_rate = latency_rate
        individual.avg_latency = metrics.avg_latency
    
    def tournament_selection(self, population: List[Individual], 
                           tournament_size: int = 3) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # 单点交叉
        crossover_point = random.randint(1, len(self.task_ids) - 1)
        
        child1_genes = {}
        child2_genes = {}
        
        for i, task_id in enumerate(self.task_ids):
            if i < crossover_point:
                child1_genes[task_id] = parent1.genes[task_id]
                child2_genes[task_id] = parent2.genes[task_id]
            else:
                child1_genes[task_id] = parent2.genes[task_id]
                child2_genes[task_id] = parent1.genes[task_id]
        
        return Individual(child1_genes), Individual(child2_genes)
    
    def mutate(self, individual: Individual) -> Individual:
        """变异操作"""
        mutated_genes = individual.genes.copy()
        
        for task_id in self.task_ids:
            if random.random() < self.mutation_rate:
                # 随机改变优先级
                current_level = mutated_genes[task_id]
                if random.random() < 0.5:
                    # 小幅调整
                    new_level = max(0, min(3, current_level + random.choice([-1, 1])))
                else:
                    # 随机重置
                    new_level = random.randint(0, 3)
                mutated_genes[task_id] = new_level
        
        return Individual(mutated_genes)
    
    def evolve(self, generations: int = 50, target_fitness: float = 0.95,
               early_stop_generations: int = 10):
        """执行遗传算法进化"""
        print(f"\n🧬 开始遗传算法优化")
        print(f"  种群大小: {self.population_size}")
        print(f"  最大代数: {generations}")
        print(f"  目标适应度: {target_fitness}")
        
        # 初始化种群
        print("\n[ANALYSIS] 初始化种群...")
        population = [self.create_individual(guided=True) for _ in range(self.population_size)]
        
        # 评估初始种群
        print("评估初始种群...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            executor.map(self.evaluate_individual, population)
        
        # 排序种群
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_individual = population[0]
        self.best_individuals.append(deepcopy(best_individual))
        
        print(f"\n初始最佳适应度: {best_individual.fitness:.3f}")
        
        # 进化循环
        no_improvement_count = 0
        
        for generation in range(generations):
            print(f"\n📈 第 {generation + 1} 代:")
            
            # 创建新种群
            new_population = []
            
            # 精英保留
            elite = population[:self.elite_size]
            new_population.extend([deepcopy(ind) for ind in elite])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                # 交叉
                child1, child2 = self.crossover(parent1, parent2)
                
                # 变异
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 限制种群大小
            new_population = new_population[:self.population_size]
            
            # 评估新个体（排除精英）
            new_individuals = new_population[self.elite_size:]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                executor.map(self.evaluate_individual, new_individuals)
            
            # 更新种群
            population = new_population
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 记录最佳个体
            current_best = population[0]
            if current_best.fitness > best_individual.fitness:
                best_individual = deepcopy(current_best)
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            self.best_individuals.append(deepcopy(best_individual))
            
            # 打印进度
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            print(f"  最佳适应度: {best_individual.fitness:.3f}")
            print(f"  平均适应度: {avg_fitness:.3f}")
            print(f"  FPS满足率: {best_individual.fps_satisfaction_rate:.1%}")
            print(f"  延迟满足率: {best_individual.latency_satisfaction_rate:.1%}")
            
            # 检查终止条件
            if best_individual.fitness >= target_fitness:
                print(f"\n🎉 达到目标适应度！")
                break
            
            if no_improvement_count >= early_stop_generations:
                print(f"\n⏹️ {early_stop_generations}代没有改进，提前停止")
                break
        
        print(f"\n✅ 进化完成！最佳适应度: {best_individual.fitness:.3f}")
        
        return best_individual
    
    def print_results(self, best_individual: Individual):
        """打印优化结果"""
        print("\n" + "=" * 100)
        print("🏆 遗传算法优化结果")
        print("=" * 100)
        
        priority_config = best_individual.to_priority_config()
        
        print(f"\n适应度: {best_individual.fitness:.3f}")
        print(f"FPS满足率: {best_individual.fps_satisfaction_rate:.1%}")
        print(f"延迟满足率: {best_individual.latency_satisfaction_rate:.1%}")
        print(f"平均延迟: {best_individual.avg_latency:.1f}ms")
        
        print("\n优先级配置:")
        print("-" * 100)
        print(f"{'任务ID':<10} {'任务名':<15} {'优先级':<10} {'被依赖':<8} "
              f"{'FPS要求':<10} {'延迟要求':<12} {'延迟严格度':<12}")
        print("-" * 100)
        
        for task_id, priority in sorted(priority_config.items()):
            features = self.task_features[task_id]
            print(f"{task_id:<10} {features['name']:<15} {priority.name:<10} "
                  f"{features['dependency_count']:<8} {features['fps_requirement']:<10} "
                  f"{features['latency_requirement']:<12.1f} "
                  f"{features['latency_strictness']:<12.2f}")
        
        # 打印进化历史
        print(f"\n进化历史（最佳适应度）:")
        print("-" * 60)
        print(f"{'代数':<6} {'适应度':<10} {'FPS满足率':<12} {'延迟满足率':<12}")
        print("-" * 60)
        
        # 显示关键代数
        key_generations = [0, len(self.best_individuals)//4, len(self.best_individuals)//2, 
                          3*len(self.best_individuals)//4, len(self.best_individuals)-1]
        
        for i in key_generations:
            if i < len(self.best_individuals):
                ind = self.best_individuals[i]
                print(f"{i+1:<6} {ind.fitness:<10.3f} "
                      f"{ind.fps_satisfaction_rate:<12.1%} "
                      f"{ind.latency_satisfaction_rate:<12.1%}")
        
        # 保存结果
        self.save_results(best_individual)
    
    def save_results(self, best_individual: Individual):
        """保存优化结果"""
        priority_config = best_individual.to_priority_config()
        
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'algorithm': 'genetic',
            'fitness': best_individual.fitness,
            'fps_satisfaction_rate': best_individual.fps_satisfaction_rate,
            'latency_satisfaction_rate': best_individual.latency_satisfaction_rate,
            'avg_latency': best_individual.avg_latency,
            'priority_config': {k: v.name for k, v in priority_config.items()},
            'evolution_history': [
                {
                    'generation': i + 1,
                    'fitness': ind.fitness,
                    'fps_rate': ind.fps_satisfaction_rate,
                    'latency_rate': ind.latency_satisfaction_rate
                }
                for i, ind in enumerate(self.best_individuals)
            ]
        }
        
        filename = f"genetic_optimized_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 优化结果已保存到: {filename}")


def compare_with_baseline():
    """与基准配置对比"""
    print("\n\n" + "=" * 100)
    print("[ANALYSIS] 与原始配置对比")
    print("=" * 100)
    
    tasks = create_real_tasks()
    
    # 保存原始优先级
    original_priorities = {task.task_id: task.priority for task in tasks}
    
    # 评估原始配置
    print("\n评估原始配置...")
    optimizer = GeneticPriorityOptimizer(tasks, segment_mode=True)
    original_individual = Individual({task.task_id: 
                                    [TaskPriority.LOW, TaskPriority.NORMAL, 
                                     TaskPriority.HIGH, TaskPriority.CRITICAL].index(task.priority)
                                    for task in tasks})
    optimizer.evaluate_individual(original_individual)
    
    print(f"\n原始配置性能:")
    print(f"  适应度: {original_individual.fitness:.3f}")
    print(f"  FPS满足率: {original_individual.fps_satisfaction_rate:.1%}")
    print(f"  延迟满足率: {original_individual.latency_satisfaction_rate:.1%}")
    
    # 运行优化
    print("\n开始优化...")
    best_individual = optimizer.evolve(generations=50, target_fitness=0.95)
    
    # 对比结果
    print(f"\n\n优化后配置性能:")
    print(f"  适应度: {best_individual.fitness:.3f} "
          f"(提升 {(best_individual.fitness - original_individual.fitness) / original_individual.fitness * 100:.1f}%)")
    print(f"  FPS满足率: {best_individual.fps_satisfaction_rate:.1%} "
          f"(原始: {original_individual.fps_satisfaction_rate:.1%})")
    print(f"  延迟满足率: {best_individual.latency_satisfaction_rate:.1%} "
          f"(原始: {original_individual.latency_satisfaction_rate:.1%})")
    
    optimizer.print_results(best_individual)


def main():
    """主函数"""
    print("=" * 100)
    print("遗传算法任务优先级优化")
    print("=" * 100)
    
    # 运行优化并与基准对比
    compare_with_baseline()


if __name__ == "__main__":
    main()