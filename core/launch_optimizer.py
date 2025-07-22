#!/usr/bin/env python3
"""
发射策略优化器 - 优化任务发射时机以最大化空闲时间
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random

from .enums import TaskPriority, ResourceType
from .task import NNTask
from .launcher import TaskLauncher, LaunchPlan, LaunchEvent
from .executor import ScheduleExecutor
from .evaluator import PerformanceEvaluator, OverallPerformanceMetrics
from .resource_queue import ResourceQueueManager
from .schedule_tracer import ScheduleTracer


@dataclass
class OptimizationConfig:
    """优化器配置"""
    # 优化目标权重
    idle_time_weight: float = 0.6      # 空闲时间权重
    fps_satisfaction_weight: float = 0.3  # FPS满足率权重
    resource_balance_weight: float = 0.1  # 资源均衡权重
    
    # FPS容忍度
    fps_tolerance: float = 0.95        # 可接受的FPS满足率(95%)
    critical_fps_tolerance: float = 1.0  # CRITICAL任务必须100%满足
    
    # 优化参数
    max_iterations: int = 100          # 最大迭代次数
    population_size: int = 50          # 种群大小(遗传算法)
    mutation_rate: float = 0.2         # 变异率
    
    # 策略偏好
    prefer_early_launch: bool = False   # 是否偏好早发射
    batch_similar_tasks: bool = True    # 是否批量处理相似任务


@dataclass 
class LaunchStrategy:
    """发射策略"""
    strategy_type: str  # "eager", "lazy", "balanced", "custom"
    delay_factors: Dict[str, float] = field(default_factory=dict)  # 任务延迟因子
    priority_boosts: Dict[str, float] = field(default_factory=dict)  # 优先级提升
    
    def apply_to_plan(self, plan: LaunchPlan) -> LaunchPlan:
        """将策略应用到发射计划"""
        modified_plan = LaunchPlan()
        
        for event in plan.events:
            new_event = LaunchEvent(
                time=event.time,
                task_id=event.task_id,
                instance_id=event.instance_id
            )
            
            # 应用延迟因子
            if event.task_id in self.delay_factors:
                delay = self.delay_factors[event.task_id]
                new_event.time += delay
            
            # 应用优先级提升
            if event.task_id in self.priority_boosts:
                # 这里简化处理，实际可能需要更复杂的优先级调整
                pass
            
            modified_plan.events.append(new_event)
        
        modified_plan.sort_events()
        return modified_plan


class LaunchOptimizer:
    """发射策略优化器"""
    
    def __init__(self, 
                 launcher: TaskLauncher,
                 queue_manager: ResourceQueueManager,
                 config: Optional[OptimizationConfig] = None):
        self.launcher = launcher
        self.queue_manager = queue_manager
        self.config = config or OptimizationConfig()
        
        # 优化结果缓存
        self.best_strategy: Optional[LaunchStrategy] = None
        self.best_metrics: Optional[OverallPerformanceMetrics] = None
        self.optimization_history: List[Tuple[LaunchStrategy, OverallPerformanceMetrics]] = []
        
    def optimize(self, time_window: float, base_strategy: str = "eager") -> LaunchStrategy:
        """
        优化发射策略
        
        Args:
            time_window: 时间窗口
            base_strategy: 基础策略类型
            
        Returns:
            优化后的发射策略
        """
        print(f"\n[OPTIMIZE] 开始优化发射策略 (目标: 最大化空闲时间)")
        print(f"  时间窗口: {time_window}ms")
        print(f"  基础策略: {base_strategy}")
        
        # 评估基线性能
        baseline_metrics = self._evaluate_strategy(
            LaunchStrategy(strategy_type=base_strategy),
            time_window
        )
        
        print(f"\n[ANALYSIS] 基线性能:")
        print(f"  空闲时间: {baseline_metrics.idle_time:.1f}ms ({baseline_metrics.idle_time_ratio:.1f}%)")
        print(f"  FPS满足率: {baseline_metrics.fps_satisfaction_rate:.1f}%")
        
        # 根据配置选择优化算法
        if self.config.population_size > 1:
            # 使用遗传算法
            best_strategy = self._genetic_optimize(time_window, baseline_metrics)
        else:
            # 使用爬山算法
            best_strategy = self._hill_climbing_optimize(time_window, baseline_metrics)
        
        self.best_strategy = best_strategy
        
        # 评估最优策略
        if self.best_metrics:
            print(f"\n[COMPLETE] 优化结果:")
            print(f"  空闲时间: {self.best_metrics.idle_time:.1f}ms "
                  f"({self.best_metrics.idle_time_ratio:.1f}%)")
            print(f"  FPS满足率: {self.best_metrics.fps_satisfaction_rate:.1f}%")
            
            improvement = self.best_metrics.idle_time - baseline_metrics.idle_time
            print(f"  改进: +{improvement:.1f}ms空闲时间")
        
        return best_strategy
    
    def _evaluate_strategy(self, strategy: LaunchStrategy, time_window: float) -> OverallPerformanceMetrics:
        """评估发射策略的性能"""
        # 为每次评估创建独立的资源管理器，避免状态污染
        eval_queue_manager = ResourceQueueManager()
        
        # 复制原始资源配置
        for res_id, queue in self.queue_manager.resource_queues.items():
            eval_queue_manager.add_resource(res_id, queue.resource_type, queue.bandwidth)
        
        # 创建新的tracer
        eval_tracer = ScheduleTracer(eval_queue_manager)
        
        # 生成基础发射计划
        base_plan = self.launcher.create_launch_plan(time_window, strategy.strategy_type)
        
        # 应用策略修改
        modified_plan = strategy.apply_to_plan(base_plan)
        
        # 执行调度
        executor = ScheduleExecutor(eval_queue_manager, eval_tracer, self.launcher.tasks)
        executor.execute_plan(modified_plan, time_window)
        
        # 评估性能
        evaluator = PerformanceEvaluator(eval_tracer, self.launcher.tasks, eval_queue_manager)
        metrics = evaluator.evaluate(time_window, modified_plan.events)
        
        return metrics
    
    def _calculate_fitness(self, metrics: OverallPerformanceMetrics) -> float:
        """计算适应度分数"""
        # 检查FPS约束
        if metrics.fps_satisfaction_rate < self.config.fps_tolerance * 100:
            # FPS不满足要求，给予惩罚
            fps_penalty = (self.config.fps_tolerance * 100 - metrics.fps_satisfaction_rate) * 10
        else:
            fps_penalty = 0
        
        # 计算加权分数
        fitness = (
            self.config.idle_time_weight * metrics.idle_time_ratio +
            self.config.fps_satisfaction_weight * metrics.fps_satisfaction_rate +
            self.config.resource_balance_weight * metrics.resource_balance_score * 100
        ) - fps_penalty
        
        return fitness
    
    def _genetic_optimize(self, time_window: float, 
                         baseline_metrics: OverallPerformanceMetrics) -> LaunchStrategy:
        """使用遗传算法优化"""
        print("\n[GENETIC] 使用遗传算法优化...")
        
        # 初始化种群
        population = self._initialize_population()
        
        best_fitness = -float('inf')
        best_strategy = None
        
        for generation in range(self.config.max_iterations):
            # 评估种群
            fitness_scores = []
            for strategy in population:
                metrics = self._evaluate_strategy(strategy, time_window)
                fitness = self._calculate_fitness(metrics)
                fitness_scores.append((fitness, strategy, metrics))
                
                # 记录历史
                self.optimization_history.append((strategy, metrics))
                
                # 更新最佳
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_strategy = strategy
                    self.best_metrics = metrics
            
            # 排序
            fitness_scores.sort(key=lambda x: x[0], reverse=True)
            
            # 选择
            elite_size = self.config.population_size // 4
            new_population = [fs[1] for fs in fitness_scores[:elite_size]]
            
            # 交叉和变异
            while len(new_population) < self.config.population_size:
                parent1 = self._tournament_select(fitness_scores)
                parent2 = self._tournament_select(fitness_scores)
                child = self._crossover(parent1, parent2)
                
                if random.random() < self.config.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            population = new_population
            
            # 进度报告
            if generation % 10 == 0:
                print(f"  第{generation}代: 最佳适应度={best_fitness:.2f}, "
                      f"空闲时间={self.best_metrics.idle_time:.1f}ms")
        
        return best_strategy
    
    def _hill_climbing_optimize(self, time_window: float,
                               baseline_metrics: OverallPerformanceMetrics) -> LaunchStrategy:
        """使用爬山算法优化"""
        print("\n⛰️ 使用爬山算法优化...")
        
        # 从基础策略开始
        current_strategy = LaunchStrategy(strategy_type="eager")
        current_metrics = baseline_metrics
        current_fitness = self._calculate_fitness(current_metrics)
        
        improvement_found = True
        iteration = 0
        
        while improvement_found and iteration < self.config.max_iterations:
            improvement_found = False
            
            # 尝试各种邻域移动
            neighbors = self._generate_neighbors(current_strategy)
            
            for neighbor in neighbors:
                metrics = self._evaluate_strategy(neighbor, time_window)
                fitness = self._calculate_fitness(metrics)
                
                if fitness > current_fitness:
                    current_strategy = neighbor
                    current_metrics = metrics
                    current_fitness = fitness
                    improvement_found = True
                    self.best_strategy = current_strategy
                    self.best_metrics = current_metrics
                    print(f"  迭代{iteration}: 改进! 空闲时间={metrics.idle_time:.1f}ms")
                    break
            
            iteration += 1
        
        return current_strategy
    
    def _initialize_population(self) -> List[LaunchStrategy]:
        """初始化种群"""
        population = []
        
        # 添加基础策略
        for base in ["eager", "lazy", "balanced"]:
            population.append(LaunchStrategy(strategy_type=base))
        
        # 添加随机变体
        while len(population) < self.config.population_size:
            strategy = LaunchStrategy(strategy_type="custom")
            
            # 随机延迟因子
            for task_id in self.launcher.task_configs:
                if random.random() < 0.5:
                    # 延迟0-50ms
                    strategy.delay_factors[task_id] = random.uniform(0, 50)
            
            population.append(strategy)
        
        return population
    
    def _tournament_select(self, fitness_scores: List[Tuple[float, LaunchStrategy, Any]]) -> LaunchStrategy:
        """锦标赛选择"""
        tournament_size = 3
        selected = random.sample(fitness_scores, tournament_size)
        winner = max(selected, key=lambda x: x[0])
        return winner[1]
    
    def _crossover(self, parent1: LaunchStrategy, parent2: LaunchStrategy) -> LaunchStrategy:
        """交叉操作"""
        child = LaunchStrategy(strategy_type="custom")
        
        # 交叉延迟因子
        all_tasks = set(parent1.delay_factors.keys()) | set(parent2.delay_factors.keys())
        for task_id in all_tasks:
            if random.random() < 0.5:
                if task_id in parent1.delay_factors:
                    child.delay_factors[task_id] = parent1.delay_factors[task_id]
            else:
                if task_id in parent2.delay_factors:
                    child.delay_factors[task_id] = parent2.delay_factors[task_id]
        
        return child
    
    def _mutate(self, strategy: LaunchStrategy) -> LaunchStrategy:
        """变异操作"""
        mutated = LaunchStrategy(strategy_type=strategy.strategy_type)
        mutated.delay_factors = strategy.delay_factors.copy()
        
        # 随机修改一些延迟因子
        for task_id in self.launcher.task_configs:
            if random.random() < 0.3:  # 30%概率变异每个任务
                if task_id in mutated.delay_factors:
                    # 修改现有延迟
                    current = mutated.delay_factors[task_id]
                    delta = random.uniform(-10, 10)
                    mutated.delay_factors[task_id] = max(0, current + delta)
                else:
                    # 添加新延迟
                    mutated.delay_factors[task_id] = random.uniform(0, 30)
        
        return mutated
    
    def _generate_neighbors(self, strategy: LaunchStrategy) -> List[LaunchStrategy]:
        """生成邻域策略"""
        neighbors = []
        
        # 尝试调整每个任务的延迟
        for task_id in self.launcher.task_configs:
            # 增加延迟
            neighbor1 = LaunchStrategy(strategy_type="custom")
            neighbor1.delay_factors = strategy.delay_factors.copy()
            current_delay = neighbor1.delay_factors.get(task_id, 0)
            neighbor1.delay_factors[task_id] = current_delay + 5
            neighbors.append(neighbor1)
            
            # 减少延迟
            if current_delay > 5:
                neighbor2 = LaunchStrategy(strategy_type="custom")
                neighbor2.delay_factors = strategy.delay_factors.copy()
                neighbor2.delay_factors[task_id] = current_delay - 5
                neighbors.append(neighbor2)
        
        return neighbors
    
    def apply_best_strategy(self) -> Optional[LaunchPlan]:
        """应用最优策略生成发射计划"""
        if not self.best_strategy:
            return None
        
        # 使用最优策略生成计划
        base_plan = self.launcher.create_launch_plan(
            self.best_metrics.time_window,
            self.best_strategy.strategy_type
        )
        
        return self.best_strategy.apply_to_plan(base_plan)
