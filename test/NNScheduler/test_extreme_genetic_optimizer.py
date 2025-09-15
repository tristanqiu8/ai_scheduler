#!/usr/bin/env python3
"""
极致遗传算法优化器 - 专注于最大化200ms窗口末尾的连续空闲时间
核心策略：尽早完成所有任务，留出最大的末尾空闲时间
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time

from NNScheduler.core import (
    ResourceType, TaskPriority,
    ResourceQueueManager, ScheduleTracer,
    TaskLauncher, ScheduleExecutor,
    PerformanceEvaluator, LaunchPlan,
    NNTask
)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


@dataclass
class ExtremeGene:
    """极致优化基因 - 表示任务的发射时机"""
    task_id: str
    launch_delay: float  # 相对于最早可能时间的延迟
    priority_boost: float  # 优先级提升值
    
    def __hash__(self):
        return hash((self.task_id, self.launch_delay, self.priority_boost))


@dataclass
class ExtremeIndividual:
    """极致优化个体 - 完整的发射策略"""
    genes: Dict[str, ExtremeGene] = field(default_factory=dict)
    fitness: float = -float('inf')
    tail_idle_time: float = 0.0  # 末尾连续空闲时间
    completion_time: float = 200.0  # 所有任务完成时间
    fps_satisfaction: float = 0.0
    dependency_violations: int = 0
    
    def __lt__(self, other):
        return self.fitness < other.fitness


class ExtremeGeneticOptimizer:
    """极致遗传算法优化器"""
    
    def __init__(self,
                 launcher: TaskLauncher,
                 queue_manager: ResourceQueueManager,
                 time_window: float = 200.0):
        self.launcher = launcher
        self.queue_manager = queue_manager
        self.time_window = time_window
        
        # 遗传算法参数 - 更激进的配置
        self.population_size = 100  # 大种群
        self.elite_size = 10  # 精英保留
        self.mutation_rate = 0.4  # 更高变异率
        self.crossover_rate = 0.9  # 更高交叉率
        self.max_generations = 100  # 更多代数
        self.tournament_size = 5
        
        # 任务依赖分析
        self.task_dependencies = self._analyze_dependencies()
        self.earliest_start_times = self._calculate_earliest_starts()
        
        # 优化历史
        self.best_individual = None
        self.generation_history = []
        
    def _analyze_dependencies(self) -> Dict[str, Set[str]]:
        """分析任务依赖关系"""
        dependencies = defaultdict(set)
        for task in self.launcher.tasks.values():
            if hasattr(task, 'dependencies') and task.dependencies:
                dependencies[task.task_id] = set(task.dependencies)
        return dict(dependencies)
    
    def _calculate_earliest_starts(self) -> Dict[str, float]:
        """计算每个任务的最早可能启动时间（考虑依赖）"""
        earliest = {}
        tasks = list(self.launcher.tasks.values())
        
        # 拓扑排序计算最早时间
        visited = set()
        
        def dfs(task_id: str) -> float:
            if task_id in visited:
                return earliest.get(task_id, 0.0)
            
            visited.add(task_id)
            task = self.launcher.tasks[task_id]
            
            # 如果没有依赖，可以立即启动
            if task_id not in self.task_dependencies or not self.task_dependencies[task_id]:
                earliest[task_id] = 0.0
                return 0.0
            
            # 计算依赖任务的完成时间
            max_dep_completion = 0.0
            for dep_id in self.task_dependencies[task_id]:
                if dep_id in self.launcher.tasks:
                    dep_task = self.launcher.tasks[dep_id]
                    dep_start = dfs(dep_id)
                    # 估算任务执行时间（使用默认带宽）
                    bandwidth_map = {}
                    for seg in dep_task.segments:
                        # 使用中等带宽估算
                        bandwidth_map[seg.resource_type] = 60.0
                    dep_duration = dep_task.estimate_duration(bandwidth_map)
                    max_dep_completion = max(max_dep_completion, dep_start + dep_duration)
            
            earliest[task_id] = max_dep_completion
            return max_dep_completion
        
        # 计算所有任务
        for task in tasks:
            dfs(task.task_id)
        
        return earliest
    
    def _create_random_individual(self) -> ExtremeIndividual:
        """创建随机个体 - 更激进的策略"""
        individual = ExtremeIndividual()
        
        for task_id in self.launcher.tasks:
            task = self.launcher.tasks[task_id]
            
            # 根据任务优先级设置不同的延迟策略
            if task.priority == TaskPriority.CRITICAL:
                # CRITICAL任务几乎不延迟
                launch_delay = random.uniform(0, 2)
            elif task.priority == TaskPriority.HIGH:
                # HIGH任务小延迟
                launch_delay = random.uniform(0, 5)
            elif task.priority == TaskPriority.NORMAL:
                # NORMAL任务可以有中等延迟
                launch_delay = random.uniform(0, 20)
            else:  # LOW
                # LOW任务可以大幅延迟
                launch_delay = random.uniform(0, 50)
            
            gene = ExtremeGene(
                task_id=task_id,
                launch_delay=launch_delay,
                priority_boost=random.uniform(0, 2)
            )
            individual.genes[task_id] = gene
        
        return individual
    
    def _evaluate_individual(self, individual: ExtremeIndividual) -> None:
        """评估个体 - 核心优化目标：最大化末尾空闲时间 + 最小化NPU气泡"""
        # 创建独立的执行环境
        eval_queue_manager = ResourceQueueManager()
        for res_id, queue in self.queue_manager.resource_queues.items():
            eval_queue_manager.add_resource(res_id, queue.resource_type, queue.bandwidth)
        
        eval_tracer = ScheduleTracer(eval_queue_manager)
        
        # 创建发射计划
        plan = self._create_launch_plan(individual)
        
        # 执行计划
        executor = ScheduleExecutor(eval_queue_manager, eval_tracer, self.launcher.tasks)
        stats = executor.execute_plan(plan, self.time_window)
        
        # 评估性能
        evaluator = PerformanceEvaluator(eval_tracer, self.launcher.tasks, eval_queue_manager)
        metrics = evaluator.evaluate(self.time_window, plan.events)
        
        # 计算末尾连续空闲时间
        tail_idle_time = self._calculate_tail_idle_time(eval_tracer)
        
        # 计算所有任务的实际完成时间
        completion_time = self._calculate_completion_time(eval_tracer)
        
        # 检查依赖违反
        dependency_violations = self._check_dependency_violations(eval_tracer)
        
        # 计算NPU气泡时间（新增）
        npu_bubble_time = self._calculate_npu_bubble_time(eval_tracer)
        
        # 更新个体属性
        individual.tail_idle_time = tail_idle_time
        individual.completion_time = completion_time
        individual.fps_satisfaction = metrics.fps_satisfaction_rate
        individual.dependency_violations = dependency_violations
        
        # 计算适应度 - 极致优化版本（修改）
        if dependency_violations is None:
            dependency_violations = 0
            
        if dependency_violations > 0:
            # 有依赖违反，严重惩罚
            individual.fitness = -1000 * dependency_violations
        elif metrics.fps_satisfaction_rate < 95:
            # FPS不满足，惩罚
            individual.fitness = -100 * (95 - metrics.fps_satisfaction_rate)
        else:
            # 主要目标：最大化末尾空闲时间 + 最小化NPU气泡
            individual.fitness = (
                tail_idle_time * 20 +  # 末尾空闲时间权重大幅提高
                (200 - completion_time) * 5 +  # 早完成奖励提高
                metrics.fps_satisfaction_rate * 0.1 -  # FPS满足率
                npu_bubble_time * 10  # NPU气泡时间惩罚加重
            )
    
    def _create_launch_plan(self, individual: ExtremeIndividual) -> LaunchPlan:
        """根据个体基因创建发射计划"""
        plan = LaunchPlan()
        
        for task_id, gene in individual.genes.items():
            task = self.launcher.tasks[task_id]
            
            # 计算实际发射时间
            earliest_start = self.earliest_start_times.get(task_id, 0.0)
            launch_time = earliest_start + gene.launch_delay
            
            # 确保不超过时间窗口
            launch_time = min(launch_time, self.time_window - 10)
            
            # 根据FPS计算实例
            period = 1000.0 / task.fps_requirement
            instance_id = 0
            
            current_time = launch_time
            while current_time < self.time_window:
                plan.add_launch(task_id, current_time, instance_id)
                instance_id += 1
                current_time += period
        
        plan.sort_events()
        return plan
    
    def _calculate_tail_idle_time(self, tracer: ScheduleTracer) -> float:
        """计算末尾连续空闲时间"""
        # 找到最后一个任务完成时间
        last_completion = 0.0
        
        for execution in tracer.executions:
            last_completion = max(last_completion, execution.end_time)
        
        # 末尾空闲时间
        return max(0, self.time_window - last_completion)
    
    def _calculate_completion_time(self, tracer: ScheduleTracer) -> float:
        """计算所有任务完成时间"""
        max_completion = 0.0
        
        for execution in tracer.executions:
            max_completion = max(max_completion, execution.end_time)
        
        return max_completion
    
    def _check_dependency_violations(self, tracer: ScheduleTracer) -> int:
        """检查依赖违反次数"""
        violations = 0
        
        # 记录每个任务的启动时间
        task_start_times = defaultdict(lambda: float('inf'))
        
        for execution in tracer.executions:
            task_id = execution.task_id.split('_')[0]  # 去掉实例编号
            task_start_times[task_id] = min(task_start_times[task_id], execution.start_time)
        
        # 检查依赖
        for task_id, deps in self.task_dependencies.items():
            if task_id in task_start_times:
                task_start = task_start_times[task_id]
                for dep_id in deps:
                    if dep_id in task_start_times:
                        # 简化检查：依赖任务应该先启动
                        if task_start_times[dep_id] > task_start:
                            violations += 1
        
    def _calculate_npu_bubble_time(self, tracer: ScheduleTracer) -> float:
        """计算NPU的气泡时间（空闲间隙）"""
        npu_executions = []
        
        # 收集所有NPU上的执行
        for execution in tracer.executions:
            if "NPU" in execution.resource_id:
                npu_executions.append((execution.start_time, execution.end_time))
        
        if not npu_executions:
            return 0.0
        
        # 按开始时间排序
        npu_executions.sort(key=lambda x: x[0])
        
        # 合并重叠的执行段
        merged = []
        for start, end in npu_executions:
            if merged and start <= merged[-1][1]:
                # 重叠，合并
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                # 不重叠，添加新段
                merged.append((start, end))
        
        # 计算气泡时间
        bubble_time = 0.0
        for i in range(1, len(merged)):
            gap = merged[i][0] - merged[i-1][1]
            if gap > 0.1:  # 忽略极小的间隙
                bubble_time += gap
        
        return bubble_time
    
    def _crossover(self, parent1: ExtremeIndividual, parent2: ExtremeIndividual) -> ExtremeIndividual:
        """交叉操作 - 混合策略"""
        child = ExtremeIndividual()
        
        for task_id in self.launcher.tasks:
            if random.random() < 0.5:
                # 继承parent1的基因
                child.genes[task_id] = ExtremeGene(
                    task_id=task_id,
                    launch_delay=parent1.genes[task_id].launch_delay,
                    priority_boost=parent1.genes[task_id].priority_boost
                )
            else:
                # 继承parent2的基因
                child.genes[task_id] = ExtremeGene(
                    task_id=task_id,
                    launch_delay=parent2.genes[task_id].launch_delay,
                    priority_boost=parent2.genes[task_id].priority_boost
                )
        
        return child
    
    def _mutate(self, individual: ExtremeIndividual) -> None:
        """变异操作 - 激进变异"""
        for task_id in self.launcher.tasks:
            if random.random() < self.mutation_rate:
                gene = individual.genes[task_id]
                
                # 变异类型
                mutation_type = random.choice(['delay', 'priority', 'both'])
                
                if mutation_type in ['delay', 'both']:
                    # 延迟变异 - 偏向减少延迟
                    if random.random() < 0.7:  # 70%概率减少延迟
                        gene.launch_delay *= random.uniform(0.5, 0.9)
                    else:
                        gene.launch_delay *= random.uniform(1.1, 1.5)
                    
                    # 限制范围
                    gene.launch_delay = max(0, min(30, gene.launch_delay))
                
                if mutation_type in ['priority', 'both']:
                    # 优先级变异
                    gene.priority_boost += random.uniform(-0.5, 0.5)
                    gene.priority_boost = max(0, min(3, gene.priority_boost))
    
    def optimize(self) -> ExtremeIndividual:
        """运行极致遗传算法优化"""
        print("\n🧬 极致遗传算法优化器启动")
        print(f"  目标: 最大化200ms窗口末尾的连续空闲时间")
        print(f"  策略: 尽早完成所有任务，满足依赖和FPS要求")
        print(f"  种群: {self.population_size}, 代数: {self.max_generations}")
        
        start_time = time.time()
        
        # 初始化种群
        population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            self._evaluate_individual(individual)
            population.append(individual)
        
        # 排序找出最佳
        population.sort(reverse=True)
        self.best_individual = population[0]
        
        print(f"\n初始最佳: 末尾空闲={self.best_individual.tail_idle_time:.1f}ms, "
              f"完成时间={self.best_individual.completion_time:.1f}ms")
        
        # 进化循环
        for generation in range(self.max_generations):
            # 精英保留
            new_population = population[:self.elite_size]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 锦标赛选择
                tournament = random.sample(population[:self.population_size//2], self.tournament_size)
                parent1 = max(tournament)
                
                tournament = random.sample(population[:self.population_size//2], self.tournament_size)
                parent2 = max(tournament)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._create_random_individual()
                
                # 变异
                self._mutate(child)
                
                # 评估
                self._evaluate_individual(child)
                new_population.append(child)
            
            # 更新种群
            population = new_population
            population.sort(reverse=True)
            
            # 更新最佳个体
            if population[0].fitness > self.best_individual.fitness:
                self.best_individual = population[0]
                print(f"\n代 {generation+1}: 新最佳! "
                      f"末尾空闲={self.best_individual.tail_idle_time:.1f}ms, "
                      f"完成={self.best_individual.completion_time:.1f}ms, "
                      f"适应度={self.best_individual.fitness:.2f}")
            
            # 记录历史
            self.generation_history.append({
                'generation': generation,
                'best_fitness': population[0].fitness,
                'avg_fitness': np.mean([ind.fitness for ind in population[:20]]),
                'tail_idle_time': population[0].tail_idle_time
            })
            
            # 收敛检查
            if len(self.generation_history) > 10:
                recent_fitness = [h['best_fitness'] for h in self.generation_history[-10:]]
                if max(recent_fitness) - min(recent_fitness) < 0.1:
                    print(f"\n收敛于代 {generation+1}")
                    break
        
        elapsed_time = time.time() - start_time
        
        print(f"\n[COMPLETE] 优化完成!")
        print(f"  用时: {elapsed_time:.2f}秒")
        print(f"  最佳个体:")
        print(f"    末尾空闲时间: {self.best_individual.tail_idle_time:.1f}ms")
        print(f"    任务完成时间: {self.best_individual.completion_time:.1f}ms")
        print(f"    FPS满足率: {self.best_individual.fps_satisfaction:.1f}%")
        print(f"    依赖违反: {self.best_individual.dependency_violations}")
        
        return self.best_individual
    
    def apply_best_strategy(self) -> LaunchPlan:
        """应用最佳策略生成发射计划"""
        if not self.best_individual:
            return None
        
        return self._create_launch_plan(self.best_individual)
    
    def print_optimization_report(self):
        """打印优化报告"""
        if not self.best_individual:
            print("未找到优化结果")
            return
        
        print("\n" + "="*80)
        print("极致优化报告")
        print("="*80)
        
        print(f"\n1. 最佳策略基因:")
        sorted_genes = sorted(self.best_individual.genes.items(), 
                            key=lambda x: x[1].launch_delay)
        
        for task_id, gene in sorted_genes[:5]:
            earliest = self.earliest_start_times.get(task_id, 0.0)
            actual_start = earliest + gene.launch_delay
            print(f"  {task_id}: 最早={earliest:.1f}ms, 延迟={gene.launch_delay:.1f}ms, "
                  f"实际={actual_start:.1f}ms, 优先级提升={gene.priority_boost:.1f}")
        
        print(f"\n2. 性能指标:")
        print(f"  末尾连续空闲: {self.best_individual.tail_idle_time:.1f}ms "
              f"({self.best_individual.tail_idle_time/self.time_window*100:.1f}%)")
        print(f"  所有任务完成: {self.best_individual.completion_time:.1f}ms")
        print(f"  提前完成时间: {self.time_window - self.best_individual.completion_time:.1f}ms")
        
        print(f"\n3. 进化历史:")
        if len(self.generation_history) > 0:
            print(f"  初始末尾空闲: {self.generation_history[0]['tail_idle_time']:.1f}ms")
            print(f"  最终末尾空闲: {self.generation_history[-1]['tail_idle_time']:.1f}ms")
            print(f"  改进: +{self.generation_history[-1]['tail_idle_time'] - self.generation_history[0]['tail_idle_time']:.1f}ms")


def test_extreme_optimizer():
    """测试极致遗传算法优化器"""
    print("[DEMO] 测试极致遗传算法优化器")
    print("="*80)
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)  # 单个NPU，40带宽
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)  # 单个DSP，40带宽
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 加载真实任务
    from NNScheduler.scenario.real_task import create_real_tasks
    tasks = create_real_tasks()
    
    for task in tasks:
        launcher.register_task(task)
    
    print(f"\n已加载 {len(tasks)} 个任务")
    
    # 创建极致优化器
    optimizer = ExtremeGeneticOptimizer(launcher, queue_manager, time_window=200.0)
    
    # 运行优化
    best_individual = optimizer.optimize()
    
    # 打印详细报告
    optimizer.print_optimization_report()
    
    # 生成并执行最优发射计划
    best_plan = optimizer.apply_best_strategy()
    
    print(f"\n最优发射计划包含 {len(best_plan.events)} 个事件")
    
    # 执行最优计划并生成可视化
    print("\n" + "="*80)
    print("执行最优计划并生成可视化")
    print("="*80)
    
    # 创建新的执行环境
    exec_queue_manager = ResourceQueueManager()
    exec_queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    exec_queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    exec_tracer = ScheduleTracer(exec_queue_manager)
    executor = ScheduleExecutor(exec_queue_manager, exec_tracer, optimizer.launcher.tasks)
    
    # 执行计划
    stats = executor.execute_plan(best_plan, 200.0)
    
    print(f"\n执行统计:")
    print(f"  完成实例: {stats['completed_instances']}/{stats['total_instances']}")
    print(f"  执行段数: {stats['total_segments_executed']}")
    
    # 生成可视化
    from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
    visualizer = ScheduleVisualizer(exec_tracer)
    
    # 1. 生成甘特图
    print("\n生成甘特图...")
    visualizer.print_gantt_chart(width=100)
    
    # 2. 生成Chrome Tracing JSON
    trace_filename = "extreme_optimized_trace.json"
    visualizer.export_chrome_tracing(trace_filename)
    print(f"\n✅ Chrome Tracing文件已生成: {trace_filename}")
    print("   (可以在Chrome浏览器中打开 chrome://tracing 并加载此文件)")
    
    # 3. 生成matplotlib图表
    try:
        from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
        visualizer = ScheduleVisualizer(exec_tracer)
        
        # 创建图形
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        # 上图：资源利用时间线
        # 获取所有资源和时间线
        all_resources = sorted(['NPU_0', 'DSP_0'])  # 只有两个资源
        timeline = exec_tracer.get_timeline()
        
        # 设置Y轴
        y_positions = {res: i for i, res in enumerate(all_resources)}
        ax1.set_yticks(range(len(all_resources)))
        ax1.set_yticklabels(all_resources)
        
        # 颜色映射
        priority_colors = {
            TaskPriority.CRITICAL: '#FF4444',
            TaskPriority.HIGH: '#FF8844', 
            TaskPriority.NORMAL: '#4488FF',
            TaskPriority.LOW: '#888888'
        }
        
        # 绘制任务块
        for resource_id, executions in timeline.items():
            y_pos = y_positions.get(resource_id, 0)
            
            for exec in executions:
                color = priority_colors.get(exec.priority, '#4488FF')
                
                # 创建矩形
                rect = Rectangle(
                    (exec.start_time, y_pos - 0.3),
                    exec.duration,
                    0.6,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1
                )
                ax1.add_patch(rect)
                
                # 添加任务标签 - 智能处理
                if exec.duration > 5:  # 足够宽的任务
                    # 解析任务ID
                    parts = exec.task_id.split('#')
                    if len(parts) == 2:
                        task_name = parts[0]
                        instance = parts[1].split('_')[0]
                        label = f"{task_name}#{instance}"
                    else:
                        label = exec.task_id
                    
                    ax1.text(
                        exec.start_time + exec.duration / 2,
                        y_pos,
                        label,
                        ha='center',
                        va='center',
                        fontsize=7 if exec.duration > 10 else 6,
                        color='white' if exec.priority == TaskPriority.CRITICAL else 'black',
                        weight='bold' if exec.priority == TaskPriority.CRITICAL else 'normal'
                    )
        
        # 设置图表属性
        ax1.set_xlabel('时间 (ms)', fontsize=12)
        ax1.set_ylabel('资源', fontsize=12)
        ax1.set_title('任务执行时间线', fontsize=14, weight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        ax1.set_xlim(0, 200)
        ax1.set_ylim(-0.5, len(all_resources) - 0.5)
        
        # 添加图例
        legend_elements = [
            patches.Patch(color=color, label=priority.name)
            for priority, color in priority_colors.items()
        ]
        ax1.legend(handles=legend_elements, loc='upper right')
        
        # 下图：空闲时间可视化
        ax2.set_xlim(0, 200)
        ax2.set_ylim(0, 1)
        
        # 标记末尾空闲时间
        if best_individual.tail_idle_time > 0:
            idle_start = 200 - best_individual.tail_idle_time
            ax2.axvspan(0, idle_start, alpha=0.3, color='lightcoral', label='工作时间')
            ax2.axvspan(idle_start, 200, alpha=0.3, color='lightgreen', label='空闲时间')
            ax2.text(idle_start + best_individual.tail_idle_time/2, 0.5,
                    f'{best_individual.tail_idle_time:.1f}ms\n空闲时间', 
                    ha='center', va='center', fontsize=16, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        ax2.set_xlabel('时间 (ms)', fontsize=12)
        ax2.set_title('末尾空闲时间分布', fontsize=14, weight='bold')
        ax2.legend()
        ax2.set_yticks([])
        
        plt.tight_layout()
        plt.savefig("extreme_optimization_result.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print("\n✅ 可视化图表已生成:")
        print("   - extreme_optimization_result.png (优化结果)")
        
    except ImportError:
        print("\n[WARNING] matplotlib未安装，跳过图表生成")
    except Exception as e:
        print(f"\n[WARNING] 图表生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. 对比基线
    print("\n" + "="*80)
    print("对比基线策略")
    print("="*80)
    
    # 执行基线策略
    baseline_plan = launcher.create_launch_plan(200.0, "eager")
    
    baseline_queue_manager = ResourceQueueManager()
    baseline_queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    baseline_queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    baseline_tracer = ScheduleTracer(baseline_queue_manager)
    baseline_executor = ScheduleExecutor(baseline_queue_manager, baseline_tracer, launcher.tasks)
    
    baseline_stats = baseline_executor.execute_plan(baseline_plan, 200.0)
    
    # 计算基线的末尾空闲时间
    baseline_completion = 0.0
    for execution in baseline_tracer.executions:
        baseline_completion = max(baseline_completion, execution.end_time)
    baseline_idle = 200.0 - baseline_completion
    
    print(f"\n[ANALYSIS] 优化效果对比:")
    print(f"{'指标':<20} {'基线':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 65)
    print(f"{'末尾空闲时间':<20} {baseline_idle:.1f}ms{'':<10} "
          f"{best_individual.tail_idle_time:.1f}ms{'':<10} "
          f"+{best_individual.tail_idle_time - baseline_idle:.1f}ms")
    print(f"{'完成时间':<20} {baseline_completion:.1f}ms{'':<10} "
          f"{best_individual.completion_time:.1f}ms{'':<10} "
          f"{baseline_completion - best_individual.completion_time:.1f}ms提前")
    print(f"{'空闲时间占比':<20} {baseline_idle/200*100:.1f}%{'':<10} "
          f"{best_individual.tail_idle_time/200*100:.1f}%{'':<10} "
          f"+{(best_individual.tail_idle_time - baseline_idle)/200*100:.1f}%")
    
    return optimizer, best_plan


if __name__ == "__main__":
    test_extreme_optimizer()
