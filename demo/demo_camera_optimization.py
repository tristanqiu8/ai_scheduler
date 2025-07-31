#!/usr/bin/env python3
"""
相机任务场景的遗传算法优化
针对camera_task.py中的真实场景，使用高带宽(120)配置进行优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enhanced_launcher import EnhancedTaskLauncher
from core.executor import ScheduleExecutor
from core.enums import ResourceType, TaskPriority
from core.evaluator import PerformanceEvaluator
from scenario.camera_task import create_real_tasks
from viz.schedule_visualizer import ScheduleVisualizer
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
    genes: Dict[str, int]  # task_id -> priority_level (0-2: LOW, NORMAL, HIGH)
    fitness: float = 0.0
    fps_satisfaction_rate: float = 0.0
    latency_satisfaction_rate: float = 0.0
    avg_latency: float = float('inf')
    npu_utilization: float = 0.0
    dsp_utilization: float = 0.0
    tracer: object = None  # ScheduleTracer for analysis
    tasks: List = None     # Task list for analysis
    
    def to_priority_config(self) -> Dict[str, TaskPriority]:
        """转换为优先级配置"""
        # 相机场景只使用三个优先级等级
        priority_map = [TaskPriority.LOW, TaskPriority.NORMAL, TaskPriority.HIGH]
        return {task_id: priority_map[level] for task_id, level in self.genes.items()}


class CameraTaskOptimizer:
    """相机任务优化器"""
    
    def __init__(self, time_window=200.0, segment_mode=True, bandwidth=120.0):
        """
        初始化优化器
        Args:
            time_window: 仿真时间窗口（ms）
            segment_mode: 是否使用段级调度
            bandwidth: NPU/DSP带宽配置
        """
        self.time_window = time_window
        self.segment_mode = segment_mode
        self.bandwidth = bandwidth
        
        # 创建任务
        self.tasks = create_real_tasks()
        self.task_ids = [task.task_id for task in self.tasks]
        
        # 分析任务特征
        self.task_features = self._analyze_task_features()
        
        # 遗传算法参数（针对相机场景调整）
        self.population_size = 30  # 增大种群以探索更多组合
        self.elite_size = 6
        self.mutation_rate = 0.15  # 稍高的变异率
        self.crossover_rate = 0.85
        
        # 最佳个体历史
        self.best_individuals = []
        
        print(f"\n[INIT] 相机任务优化器初始化完成")
        print(f"  任务数量: {len(self.tasks)}")
        print(f"  时间窗口: {time_window}ms")
        print(f"  带宽配置: {bandwidth}")
        print(f"  调度模式: {'段级' if segment_mode else '传统'}")
        
    def _analyze_task_features(self) -> Dict[str, dict]:
        """分析任务特征"""
        features = {}
        
        # 计算被依赖次数
        dependency_count = defaultdict(int)
        for task in self.tasks:
            for dep in task.dependencies:
                dependency_count[dep] += 1
        
        # 分析帧率分布
        fps_values = [task.fps_requirement for task in self.tasks]
        max_fps = max(fps_values)
        min_fps = min(fps_values)
        
        print(f"\n[ANALYSIS] 任务特征分析:")
        print(f"  FPS范围: {min_fps} - {max_fps}")
        print(f"  主频率: 15 FPS, 高频率: 30 FPS")
        
        for task in self.tasks:
            # 估算在高带宽下的执行时间
            bandwidth_map = {ResourceType.NPU: self.bandwidth, ResourceType.DSP: self.bandwidth}
            estimated_duration = task.estimate_duration(bandwidth_map)
            
            # 计算紧急程度分数
            urgency_score = 0.0
            
            # FPS要求越高，紧急程度越高
            fps_norm = task.fps_requirement / max_fps
            urgency_score += fps_norm * 0.3
            
            # 被依赖越多，紧急程度越高
            dep_norm = min(1.0, dependency_count[task.task_id] / 3.0)
            urgency_score += dep_norm * 0.4
            
            # 延迟要求越严格，紧急程度越高
            if task.latency_requirement > 0:
                latency_tightness = estimated_duration / task.latency_requirement
                urgency_score += min(1.0, latency_tightness) * 0.3
            
            features[task.task_id] = {
                'name': task.name,
                'fps_requirement': task.fps_requirement,
                'fps_normalized': fps_norm,
                'latency_requirement': task.latency_requirement,
                'estimated_duration': estimated_duration,
                'dependency_count': dependency_count[task.task_id],
                'has_dependencies': len(task.dependencies) > 0,
                'is_depended': dependency_count[task.task_id] > 0,
                'urgency_score': urgency_score,
                'is_high_fps': task.fps_requirement > 15,  # 高于主频率
                'num_segments': len(task.segments)
            }
        
        return features
    
    def create_individual(self, guided=True) -> Individual:
        """创建个体"""
        genes = {}
        
        if guided and random.random() < 0.8:  # 80%概率使用启发式
            # 基于紧急程度的启发式初始化
            for task_id in self.task_ids:
                features = self.task_features[task_id]
                urgency = features['urgency_score']
                
                # 根据紧急程度分配优先级
                if urgency > 0.7 or features['is_depended']:
                    genes[task_id] = 2  # HIGH
                elif urgency > 0.4 or features['is_high_fps']:
                    genes[task_id] = 1  # NORMAL
                else:
                    genes[task_id] = 0  # LOW
                
                # 特殊处理：被T2依赖的任务应该有较高优先级
                if task_id in ['T1', 'T3', 'T5']:
                    genes[task_id] = max(1, genes[task_id])  # 至少NORMAL
                
                # 添加随机扰动
                if random.random() < 0.2:
                    genes[task_id] = max(0, min(2, genes[task_id] + random.randint(-1, 1)))
        else:
            # 随机初始化
            for task_id in self.task_ids:
                genes[task_id] = random.randint(0, 2)
        
        return Individual(genes)
    
    def evaluate_individual(self, individual: Individual) -> None:
        """评估个体适应度"""
        # 转换为优先级配置
        priority_config = individual.to_priority_config()
        
        # 应用配置到任务副本
        task_copy = deepcopy(self.tasks)
        for task in task_copy:
            task.priority = priority_config[task.task_id]
        
        # 创建高带宽调度环境
        queue_manager = ResourceQueueManager()
        queue_manager.add_resource("NPU_0", ResourceType.NPU, self.bandwidth)
        queue_manager.add_resource("DSP_0", ResourceType.DSP, self.bandwidth)
        
        tracer = ScheduleTracer(queue_manager)
        
        # 使用增强型发射器以支持段级调度
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
        
        # 保存tracer和tasks以便后续分析
        individual.tracer = tracer
        individual.tasks = task_copy
        
        # 计算任务级满足率
        fps_satisfied = 0
        latency_satisfied = 0
        weighted_fps_score = 0.0
        critical_tasks_satisfied = 0
        critical_task_count = 0
        
        for task_id, task_metrics in evaluator.task_metrics.items():
            features = self.task_features[task_id]
            
            # FPS满足情况
            if task_metrics.fps_satisfaction:
                fps_satisfied += 1
                # 高FPS任务权重更高
                if features['is_high_fps']:
                    weighted_fps_score += 1.5
                else:
                    weighted_fps_score += 1.0
            
            # 延迟满足情况
            if task_metrics.latency_satisfaction_rate > 0.9:
                latency_satisfied += 1
            
            # 关键任务（被依赖的任务）满足情况
            if features['is_depended']:
                critical_task_count += 1
                if task_metrics.fps_satisfaction and task_metrics.latency_satisfaction_rate > 0.85:
                    critical_tasks_satisfied += 1
        
        num_tasks = len(evaluator.task_metrics)
        fps_rate = fps_satisfied / num_tasks if num_tasks > 0 else 0
        latency_rate = latency_satisfied / num_tasks if num_tasks > 0 else 0
        weighted_fps_rate = weighted_fps_score / (num_tasks * 1.2) if num_tasks > 0 else 0
        critical_satisfaction = critical_tasks_satisfied / critical_task_count if critical_task_count > 0 else 1.0
        
        # 计算综合适应度（针对相机场景调整权重）
        fitness = (
            weighted_fps_rate * 0.35 +           # 加权FPS满足率
            latency_rate * 0.25 +                # 延迟满足率
            critical_satisfaction * 0.25 +        # 关键任务满足率
            (metrics.avg_npu_utilization / 100) * 0.10 +  # NPU利用率
            (metrics.avg_dsp_utilization / 100) * 0.05    # DSP利用率
        )
        
        # 如果所有任务都不能满足FPS，严重惩罚
        if fps_satisfied == 0:
            fitness *= 0.1
        
        # 更新个体信息
        individual.fitness = fitness
        individual.fps_satisfaction_rate = fps_rate
        individual.latency_satisfaction_rate = latency_rate
        individual.avg_latency = metrics.avg_latency
        individual.npu_utilization = metrics.avg_npu_utilization
        individual.dsp_utilization = metrics.avg_dsp_utilization
    
    def tournament_selection(self, population: List[Individual], 
                           tournament_size: int = 4) -> Individual:
        """锦标赛选择"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        # 均匀交叉（更适合优先级配置）
        child1_genes = {}
        child2_genes = {}
        
        for task_id in self.task_ids:
            if random.random() < 0.5:
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
                current_level = mutated_genes[task_id]
                # 倾向于小幅调整
                if random.random() < 0.7:
                    # 上下调整一级
                    new_level = max(0, min(2, current_level + random.choice([-1, 1])))
                else:
                    # 随机重置
                    new_level = random.randint(0, 2)
                mutated_genes[task_id] = new_level
        
        return Individual(mutated_genes)
    
    def evolve(self, generations: int = 30, target_fitness: float = 0.9,
               early_stop_generations: int = 8):
        """执行遗传算法进化"""
        print(f"\n[EVOLUTION] 开始相机任务优化")
        print(f"  种群大小: {self.population_size}")
        print(f"  最大代数: {generations}")
        print(f"  目标适应度: {target_fitness}")
        
        # 初始化种群
        print("\n[INIT] 初始化种群...")
        population = [self.create_individual(guided=True) for _ in range(self.population_size)]
        
        # 评估初始种群
        print("评估初始种群...")
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            executor.map(self.evaluate_individual, population)
        
        # 排序种群
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_individual = population[0]
        self.best_individuals.append(deepcopy(best_individual))
        
        print(f"\n初始最佳适应度: {best_individual.fitness:.3f}")
        print(f"初始FPS满足率: {best_individual.fps_satisfaction_rate:.1%}")
        
        # 进化循环
        no_improvement_count = 0
        
        for generation in range(generations):
            gen_start = time.time()
            print(f"\n[GEN {generation + 1}] 第 {generation + 1} 代:")
            
            # 创建新种群
            new_population = []
            
            # 精英保留
            elite = population[:self.elite_size]
            new_population.extend([deepcopy(ind) for ind in elite])
            
            # 生成新个体
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population)
                parent2 = self.tournament_selection(population)
                
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.extend([child1, child2])
            
            # 限制种群大小
            new_population = new_population[:self.population_size]
            
            # 评估新个体
            new_individuals = new_population[self.elite_size:]
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                executor.map(self.evaluate_individual, new_individuals)
            
            # 更新种群
            population = new_population
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 记录最佳个体
            current_best = population[0]
            if current_best.fitness > best_individual.fitness:
                best_individual = deepcopy(current_best)
                no_improvement_count = 0
                print(f"  [IMPROVED] 发现更优解！")
            else:
                no_improvement_count += 1
            
            self.best_individuals.append(deepcopy(best_individual))
            
            # 打印进度
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            print(f"  最佳适应度: {best_individual.fitness:.3f}")
            print(f"  平均适应度: {avg_fitness:.3f}")
            print(f"  FPS满足率: {best_individual.fps_satisfaction_rate:.1%}")
            print(f"  延迟满足率: {best_individual.latency_satisfaction_rate:.1%}")
            print(f"  资源利用率: NPU {best_individual.npu_utilization:.1f}%, DSP {best_individual.dsp_utilization:.1f}%")
            print(f"  代耗时: {time.time() - gen_start:.1f}s")
            
            # 检查终止条件
            if best_individual.fitness >= target_fitness:
                print(f"\n[SUCCESS] 达到目标适应度！")
                break
            
            if no_improvement_count >= early_stop_generations:
                print(f"\n[EARLY_STOP] {early_stop_generations}代没有改进，提前停止")
                break
        
        total_time = time.time() - start_time
        print(f"\n[COMPLETED] 优化完成！")
        print(f"  最终适应度: {best_individual.fitness:.3f}")
        print(f"  总耗时: {total_time:.1f}s")
        
        return best_individual
    
    def analyze_timing_accuracy(self, individual: Individual):
        """分析理论vs实际耗时精度"""
        if not individual.tracer or not individual.tasks:
            return {}
        
        print("\n" + "=" * 100)
        print("[TIMING_ANALYSIS] 理论vs实际耗时对比分析")
        print("=" * 100)
        
        bandwidth_map = {ResourceType.NPU: self.bandwidth, ResourceType.DSP: self.bandwidth}
        timing_analysis = {}
        
        print(f"\n{'任务ID':<8} {'理论耗时(ms)':<12} {'实际耗时(ms)':<12} {'误差(ms)':<10} {'误差率':<10} {'状态':<8}")
        print("-" * 80)
        
        total_theoretical = 0.0
        total_actual = 0.0
        task_count = 0
        
        # 分析每个任务的理论vs实际耗时
        for task in individual.tasks:
            # 计算理论执行时间
            theoretical_time = task.estimate_duration(bandwidth_map)
            
            # 从tracer中获取实际执行时间
            actual_executions = [exec for exec in individual.tracer.executions 
                               if exec.task_id == task.task_id]
            
            if actual_executions:
                # 计算平均实际执行时间
                actual_time = sum(exec.duration for exec in actual_executions) / len(actual_executions)
                
                # 计算误差
                error = actual_time - theoretical_time
                error_rate = (error / theoretical_time) * 100 if theoretical_time > 0 else 0
                
                # 状态判断
                if abs(error_rate) < 1.0:
                    status = "[EXACT]"
                elif abs(error_rate) < 5.0:
                    status = "[GOOD]"
                elif abs(error_rate) < 10.0:
                    status = "[OK]"
                else:
                    status = "[WARN]"
                
                print(f"{task.task_id:<8} {theoretical_time:<12.2f} {actual_time:<12.2f} "
                      f"{error:<10.2f} {error_rate:<10.1f}% {status:<8}")
                
                timing_analysis[task.task_id] = {
                    'theoretical': theoretical_time,
                    'actual': actual_time,
                    'error': error,
                    'error_rate': error_rate,
                    'executions': len(actual_executions)
                }
                
                total_theoretical += theoretical_time
                total_actual += actual_time
                task_count += 1
            else:
                print(f"{task.task_id:<8} {theoretical_time:<12.2f} {'N/A':<12} "
                      f"{'N/A':<10} {'N/A':<10} [NO_EXEC]")
        
        # 总体统计
        if task_count > 0:
            avg_error = (total_actual - total_theoretical) / task_count
            avg_error_rate = (avg_error / (total_theoretical / task_count)) * 100
            
            print("-" * 80)
            print(f"{'总体':<8} {total_theoretical/task_count:<12.2f} {total_actual/task_count:<12.2f} "
                  f"{avg_error:<10.2f} {avg_error_rate:<10.1f}% [AVG]")
            
            timing_analysis['summary'] = {
                'avg_theoretical': total_theoretical / task_count,
                'avg_actual': total_actual / task_count,
                'avg_error': avg_error,
                'avg_error_rate': avg_error_rate,
                'task_count': task_count
            }
        
        # 资源利用率分析
        print("\n[RESOURCE_ANALYSIS] NPU/DSP实际使用统计:")
        print("-" * 60)
        
        npu_executions = [exec for exec in individual.tracer.executions 
                         if exec.resource_id == "NPU_0"]
        dsp_executions = [exec for exec in individual.tracer.executions 
                         if exec.resource_id == "DSP_0"]
        
        npu_total_time = sum(exec.duration for exec in npu_executions)
        dsp_total_time = sum(exec.duration for exec in dsp_executions)
        
        print(f"NPU实际执行时间: {npu_total_time:.2f}ms (执行次数: {len(npu_executions)})")
        print(f"DSP实际执行时间: {dsp_total_time:.2f}ms (执行次数: {len(dsp_executions)})")
        print(f"NPU利用率: {individual.npu_utilization:.1f}%")
        print(f"DSP利用率: {individual.dsp_utilization:.1f}%")
        
        return timing_analysis
    
    def export_chrome_tracing(self, individual: Individual, filename: str = None):
        """导出Chrome Tracing格式文件"""
        if not individual.tracer:
            print("[WARNING] 无tracer数据，跳过Chrome Tracing导出")
            return
        
        if not filename:
            filename = f"camera_schedule_trace_{time.strftime('%Y%m%d_%H%M%S')}.json"
        
        visualizer = ScheduleVisualizer(individual.tracer)
        visualizer.export_chrome_tracing(filename)
        print(f"\n[CHROME_TRACING] 调度轨迹已导出到: {filename}")
        print(f"  可以在Chrome浏览器中打开 chrome://tracing 并加载此文件进行可视化分析")
        
        return filename

    def print_results(self, best_individual: Individual):
        """打印优化结果"""
        print("\n" + "=" * 100)
        print("[RESULTS] 相机任务优化结果")
        print("=" * 100)
        
        priority_config = best_individual.to_priority_config()
        
        print(f"\n性能指标:")
        print(f"  适应度: {best_individual.fitness:.3f}")
        print(f"  FPS满足率: {best_individual.fps_satisfaction_rate:.1%}")
        print(f"  延迟满足率: {best_individual.latency_satisfaction_rate:.1%}")
        print(f"  平均延迟: {best_individual.avg_latency:.1f}ms")
        print(f"  NPU利用率: {best_individual.npu_utilization:.1f}%")
        print(f"  DSP利用率: {best_individual.dsp_utilization:.1f}%")
        
        print("\n优化后的优先级配置:")
        print("-" * 100)
        print(f"{'任务ID':<8} {'任务名':<20} {'FPS':<6} {'优先级':<10} "
              f"{'被依赖':<8} {'紧急度':<8}")
        print("-" * 100)
        
        # 按优先级排序显示
        sorted_tasks = sorted(priority_config.items(), 
                            key=lambda x: (priority_config[x[0]].value, x[0]), 
                            reverse=True)
        
        for task_id, priority in sorted_tasks:
            features = self.task_features[task_id]
            print(f"{task_id:<8} {features['name']:<20} {features['fps_requirement']:<6} "
                  f"{priority.name:<10} {features['dependency_count']:<8} "
                  f"{features['urgency_score']:<8.2f}")
        
        # 打印依赖关系提示
        print("\n依赖关系分析:")
        print("-" * 60)
        print("T2 (FaceEhnsLite) 依赖: T1, T3, T5")
        t2_deps_priority = [priority_config[dep].name for dep in ['T1', 'T3', 'T5']]
        print(f"  依赖任务优先级: {', '.join(t2_deps_priority)}")
        
        # 分析理论vs实际耗时
        timing_analysis = self.analyze_timing_accuracy(best_individual)
        
        # 导出Chrome Tracing文件
        trace_filename = self.export_chrome_tracing(best_individual)
        
        # 保存结果
        self.save_results(best_individual, timing_analysis, trace_filename)
    
    def save_results(self, best_individual: Individual, timing_analysis: Dict = None, trace_filename: str = None):
        """保存优化结果"""
        priority_config = best_individual.to_priority_config()
        
        output = {
            'scenario': 'camera_task',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'configuration': {
                'bandwidth': self.bandwidth,
                'time_window': self.time_window,
                'segment_mode': self.segment_mode
            },
            'performance': {
                'fitness': best_individual.fitness,
                'fps_satisfaction_rate': best_individual.fps_satisfaction_rate,
                'latency_satisfaction_rate': best_individual.latency_satisfaction_rate,
                'avg_latency': best_individual.avg_latency,
                'npu_utilization': best_individual.npu_utilization,
                'dsp_utilization': best_individual.dsp_utilization
            },
            'priority_config': {k: v.name for k, v in priority_config.items()},
            'task_features': self.task_features
        }
        
        # 添加时序分析结果
        if timing_analysis:
            output['timing_analysis'] = timing_analysis
            
        # 添加trace文件信息
        if trace_filename:
            output['chrome_tracing_file'] = trace_filename
        
        filename = f"camera_optimized_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SAVED] 优化结果已保存到: {filename}")


def test_different_bandwidths():
    """测试不同带宽下的优化效果"""
    print("\n" + "=" * 100)
    print("[BANDWIDTH_TEST] 不同带宽配置下的优化测试")
    print("=" * 100)
    
    bandwidths = [60, 80, 100, 120]
    results = []
    
    for bw in bandwidths:
        print(f"\n\n[TEST] 测试带宽: {bw}")
        print("-" * 60)
        
        optimizer = CameraTaskOptimizer(time_window=200.0, segment_mode=True, bandwidth=bw)
        best = optimizer.evolve(generations=20, target_fitness=0.95)
        
        results.append({
            'bandwidth': bw,
            'fitness': best.fitness,
            'fps_rate': best.fps_satisfaction_rate,
            'latency_rate': best.latency_satisfaction_rate
        })
    
    # 打印对比结果
    print("\n\n[COMPARISON] 带宽对比结果:")
    print("-" * 80)
    print(f"{'带宽':<10} {'适应度':<10} {'FPS满足率':<12} {'延迟满足率':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['bandwidth']:<10} {r['fitness']:<10.3f} "
              f"{r['fps_rate']:<12.1%} {r['latency_rate']:<12.1%}")


def main():
    """主函数"""
    print("=" * 100)
    print("[CAMERA] 相机任务场景优化")
    print("=" * 100)
    
    # 创建优化器（使用120带宽）
    optimizer = CameraTaskOptimizer(time_window=200.0, segment_mode=True, bandwidth=120.0)
    
    # 运行优化
    best_individual = optimizer.evolve(generations=30, target_fitness=0.95)
    
    # 打印结果
    optimizer.print_results(best_individual)
    
    # 可选：测试不同带宽
    # test_different_bandwidths()


if __name__ == "__main__":
    main()