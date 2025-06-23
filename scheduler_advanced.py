#!/usr/bin/env python3
"""
Advanced scheduling algorithms using metaheuristics and machine learning
"""

import random
import math
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import copy
from collections import defaultdict

from enums import (
    ResourceType, TaskPriority, RuntimeType, TaskState,
    SchedulingAlgorithm, OptimizationObjective, SchedulerConfig, SegmentationStrategy
)
from models import (
    ResourceUnit, TaskScheduleInfo, ResourceBinding,
    SchedulingDecision, SystemState, SchedulingMetrics
)
from task import NNTask, TaskSet
from scheduler_base import BaseScheduler


@dataclass
class Individual:
    """Individual solution for genetic algorithm"""
    chromosome: List[Tuple[str, int, Dict[str, Any]]]  # [(task_id, priority, config), ...]
    fitness: float = 0.0
    schedule: List[TaskScheduleInfo] = field(default_factory=list)
    
    def mutate(self, mutation_rate: float = 0.1):
        """Mutate the chromosome"""
        for i in range(len(self.chromosome)):
            if random.random() < mutation_rate:
                task_id, priority, config = self.chromosome[i]
                
                # Mutate priority
                if random.random() < 0.5:
                    new_priority = random.randint(0, 3)
                    self.chromosome[i] = (task_id, new_priority, config)
                
                # Mutate configuration
                else:
                    new_config = config.copy()
                    if 'runtime_type' in new_config:
                        new_config['runtime_type'] = random.choice(['ACPU', 'DSP'])
                    if 'segmentation' in new_config:
                        new_config['segmentation'] = random.choice(['none', 'adaptive', 'aggressive'])
                    self.chromosome[i] = (task_id, priority, new_config)


class GeneticScheduler(BaseScheduler):
    """Genetic Algorithm based scheduler"""
    
    def __init__(self, resources: Dict[str, ResourceUnit],
                 population_size: int = 50,
                 generations: int = 100,
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1):
        super().__init__(resources)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        
    def get_algorithm_name(self) -> str:
        return "GeneticAlgorithm"
    
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Schedule using genetic algorithm"""
        # Initialize population
        population = self._initialize_population(tasks)
        
        best_individual = None
        best_fitness = -float('inf')
        
        # Evolution loop
        for generation in range(self.generations):
            # Evaluate fitness
            for individual in population:
                self._evaluate_fitness(individual, tasks, time_limit_ms)
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if population[0].fitness > best_fitness:
                best_fitness = population[0].fitness
                best_individual = copy.deepcopy(population[0])
            
            # Selection and reproduction
            new_population = []
            
            # Elitism - keep best individuals
            elite_size = self.population_size // 10
            new_population.extend(population[:elite_size])
            
            # Crossover and mutation
            while len(new_population) < self.population_size:
                # Tournament selection
                parent1 = self._tournament_selection(population)
                parent2 = self._tournament_selection(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
                
                # Mutation
                child1.mutate(self.mutation_rate)
                child2.mutate(self.mutation_rate)
                
                new_population.extend([child1, child2])
            
            population = new_population[:self.population_size]
        
        # Return best schedule found
        return best_individual.schedule if best_individual else []
    
    def _initialize_population(self, tasks: TaskSet) -> List[Individual]:
        """Create initial population"""
        population = []
        
        for _ in range(self.population_size):
            chromosome = []
            
            for task in tasks.tasks.values():
                # Random priority assignment
                priority = random.randint(0, 3)
                
                # Random configuration
                config = {
                    'runtime_type': random.choice(['ACPU', 'DSP']),
                    'segmentation': random.choice(['none', 'adaptive', 'aggressive']),
                    'resource_preference': random.choice(['bandwidth', 'thermal', 'balanced'])
                }
                
                chromosome.append((task.id, priority, config))
            
            # Shuffle order
            random.shuffle(chromosome)
            
            population.append(Individual(chromosome=chromosome))
        
        return population
    
    def _evaluate_fitness(self, individual: Individual, tasks: TaskSet, time_limit_ms: float):
        """Evaluate fitness of an individual"""
        # Apply chromosome to tasks
        modified_tasks = self._apply_chromosome(individual.chromosome, tasks)
        
        # Create simple scheduler to evaluate
        from scheduler_base import SimpleScheduler
        simple_scheduler = SimpleScheduler(self.resources)
        
        # Generate schedule
        schedule = simple_scheduler.schedule(modified_tasks, time_limit_ms)
        individual.schedule = schedule
        
        # Calculate metrics
        metrics = self.calculate_metrics(schedule)
        
        # Multi-objective fitness
        fitness = 0.0
        
        # Minimize makespan
        if metrics.makespan_ms > 0:
            fitness += 1000.0 / metrics.makespan_ms
        
        # Minimize average latency
        if metrics.average_latency_ms > 0:
            fitness += 100.0 / metrics.average_latency_ms
        
        # Maximize utilization
        avg_util = np.mean(list(metrics.average_utilization.values())) if metrics.average_utilization else 0
        fitness += avg_util * 50.0
        
        # Minimize deadline misses
        fitness -= metrics.deadline_miss_count * 10.0
        
        # Energy efficiency
        if metrics.energy_per_task_j > 0:
            fitness += 10.0 / metrics.energy_per_task_j
        
        individual.fitness = fitness
    
    def _apply_chromosome(self, chromosome: List[Tuple], tasks: TaskSet) -> TaskSet:
        """Apply chromosome configuration to tasks"""
        modified_tasks = TaskSet()
        
        for task_id, priority, config in chromosome:
            task = tasks.get_task(task_id)
            if task:
                # Create modified copy
                modified_task = copy.deepcopy(task)
                
                # Apply priority
                modified_task.priority = TaskPriority(priority)
                
                # Apply configuration
                if config.get('runtime_type') == 'DSP':
                    modified_task.runtime_type = RuntimeType.DSP_RUNTIME
                else:
                    modified_task.runtime_type = RuntimeType.ACPU_RUNTIME
                
                # Apply segmentation strategy
                seg_map = {
                    'none': SegmentationStrategy.NO_SEGMENTATION,
                    'adaptive': SegmentationStrategy.ADAPTIVE_SEGMENTATION,
                    'aggressive': SegmentationStrategy.AGGRESSIVE_SEGMENTATION
                }
                seg_strategy = config.get('segmentation', 'adaptive')
                modified_task.segmentation_strategy = seg_map.get(seg_strategy, 
                                                                 SegmentationStrategy.ADAPTIVE_SEGMENTATION)
                
                modified_tasks.add_task(modified_task)
        
        return modified_tasks
    
    def _tournament_selection(self, population: List[Individual], 
                            tournament_size: int = 3) -> Individual:
        """Tournament selection"""
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Two-point crossover"""
        size = len(parent1.chromosome)
        
        # Select crossover points
        point1 = random.randint(0, size - 1)
        point2 = random.randint(point1, size)
        
        # Create children
        child1_chromosome = (parent1.chromosome[:point1] + 
                           parent2.chromosome[point1:point2] + 
                           parent1.chromosome[point2:])
        
        child2_chromosome = (parent2.chromosome[:point1] + 
                           parent1.chromosome[point1:point2] + 
                           parent2.chromosome[point2:])
        
        return Individual(chromosome=child1_chromosome), Individual(chromosome=child2_chromosome)


class SimulatedAnnealingScheduler(BaseScheduler):
    """Simulated Annealing based scheduler"""
    
    def __init__(self, resources: Dict[str, ResourceUnit],
                 initial_temp: float = 100.0,
                 cooling_rate: float = 0.95,
                 min_temp: float = 0.1,
                 iterations_per_temp: int = 10):
        super().__init__(resources)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        
    def get_algorithm_name(self) -> str:
        return "SimulatedAnnealing"
    
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Schedule using simulated annealing"""
        # Generate initial solution
        current_solution = self._generate_initial_solution(tasks)
        current_schedule = self._evaluate_solution(current_solution, tasks, time_limit_ms)
        current_cost = self._calculate_cost(current_schedule)
        
        best_solution = copy.deepcopy(current_solution)
        best_schedule = current_schedule
        best_cost = current_cost
        
        temperature = self.initial_temp
        
        while temperature > self.min_temp:
            for _ in range(self.iterations_per_temp):
                # Generate neighbor
                neighbor = self._generate_neighbor(current_solution)
                neighbor_schedule = self._evaluate_solution(neighbor, tasks, time_limit_ms)
                neighbor_cost = self._calculate_cost(neighbor_schedule)
                
                # Calculate delta
                delta = neighbor_cost - current_cost
                
                # Accept or reject
                if delta < 0 or random.random() < math.exp(-delta / temperature):
                    current_solution = neighbor
                    current_schedule = neighbor_schedule
                    current_cost = neighbor_cost
                    
                    # Update best if improved
                    if current_cost < best_cost:
                        best_solution = copy.deepcopy(current_solution)
                        best_schedule = current_schedule
                        best_cost = current_cost
            
            # Cool down
            temperature *= self.cooling_rate
        
        return best_schedule
    
    def _generate_initial_solution(self, tasks: TaskSet) -> Dict[str, Dict[str, Any]]:
        """Generate initial solution"""
        solution = {}
        
        for task in tasks.tasks.values():
            solution[task.id] = {
                'priority': task.priority.value,
                'order': random.random(),  # Random ordering
                'runtime_type': task.runtime_type.value,
                'segmentation': 'adaptive',
                'resource_affinity': {}  # task_id -> resource_id mapping
            }
        
        return solution
    
    def _generate_neighbor(self, solution: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Generate neighbor solution"""
        neighbor = copy.deepcopy(solution)
        
        # Choose random modification
        modification = random.choice(['swap', 'priority', 'runtime', 'segmentation'])
        
        if modification == 'swap':
            # Swap order of two tasks
            if len(neighbor) >= 2:
                task_ids = list(neighbor.keys())
                id1, id2 = random.sample(task_ids, 2)
                neighbor[id1]['order'], neighbor[id2]['order'] = neighbor[id2]['order'], neighbor[id1]['order']
        
        elif modification == 'priority':
            # Change priority of random task
            task_id = random.choice(list(neighbor.keys()))
            neighbor[task_id]['priority'] = random.randint(0, 3)
        
        elif modification == 'runtime':
            # Change runtime type
            task_id = random.choice(list(neighbor.keys()))
            neighbor[task_id]['runtime_type'] = random.choice(['ACPU', 'DSP'])
        
        elif modification == 'segmentation':
            # Change segmentation strategy
            task_id = random.choice(list(neighbor.keys()))
            neighbor[task_id]['segmentation'] = random.choice(['none', 'adaptive', 'aggressive'])
        
        return neighbor
    
    def _evaluate_solution(self, solution: Dict[str, Dict[str, Any]], 
                         tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Evaluate a solution by generating schedule"""
        # Apply solution to tasks
        modified_tasks = TaskSet()
        
        # Sort tasks by order in solution
        sorted_tasks = sorted(tasks.tasks.values(), 
                            key=lambda t: solution[t.id]['order'])
        
        for task in sorted_tasks:
            config = solution[task.id]
            modified_task = copy.deepcopy(task)
            
            # Apply configuration
            modified_task.priority = TaskPriority(config['priority'])
            modified_task.runtime_type = RuntimeType(config['runtime_type'])
            
            seg_map = {
                'none': SegmentationStrategy.NO_SEGMENTATION,
                'adaptive': SegmentationStrategy.ADAPTIVE_SEGMENTATION,
                'aggressive': SegmentationStrategy.AGGRESSIVE_SEGMENTATION
            }
            modified_task.segmentation_strategy = seg_map.get(config['segmentation'],
                                                             SegmentationStrategy.ADAPTIVE_SEGMENTATION)
            
            modified_tasks.add_task(modified_task)
        
        # Use simple scheduler to evaluate
        from scheduler_base import SimpleScheduler
        scheduler = SimpleScheduler(self.resources)
        return scheduler.schedule(modified_tasks, time_limit_ms)
    
    def _calculate_cost(self, schedule: List[TaskScheduleInfo]) -> float:
        """Calculate cost of a schedule (to minimize)"""
        metrics = self.calculate_metrics(schedule)
        
        # Multi-objective cost
        cost = 0.0
        
        # Makespan
        cost += metrics.makespan_ms
        
        # Average latency
        cost += metrics.average_latency_ms * 10
        
        # Deadline misses
        cost += metrics.deadline_miss_count * 100
        
        # Energy
        cost += metrics.total_energy_j
        
        # Utilization (inverted - we want high utilization)
        avg_util = np.mean(list(metrics.average_utilization.values())) if metrics.average_utilization else 0
        cost += (1.0 - avg_util) * 50
        
        return cost


class ReinforcementLearningScheduler(BaseScheduler):
    """Reinforcement Learning based scheduler (simplified Q-learning)"""
    
    def __init__(self, resources: Dict[str, ResourceUnit],
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.9,
                 epsilon: float = 0.1,
                 episodes: int = 100):
        super().__init__(resources)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.episodes = episodes
        
        # Q-table: state -> action -> value
        self.q_table = defaultdict(lambda: defaultdict(float))
        
    def get_algorithm_name(self) -> str:
        return "ReinforcementLearning"
    
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Schedule using reinforcement learning"""
        best_schedule = []
        best_reward = -float('inf')
        
        # Training episodes
        for episode in range(self.episodes):
            # Decay epsilon
            current_epsilon = self.epsilon * (1 - episode / self.episodes)
            
            # Run episode
            schedule = self._run_episode(tasks, time_limit_ms, current_epsilon)
            
            # Calculate reward
            reward = self._calculate_reward(schedule)
            
            # Update best
            if reward > best_reward:
                best_reward = reward
                best_schedule = schedule
        
        return best_schedule
    
    def _run_episode(self, tasks: TaskSet, time_limit_ms: float, epsilon: float) -> List[TaskScheduleInfo]:
        """Run one episode of scheduling"""
        schedule = []
        current_time = 0.0
        completed_tasks = set()
        
        # Create a copy of tasks for this episode
        episode_tasks = copy.deepcopy(tasks)
        
        while current_time < time_limit_ms:
            # Get current state
            state = self._get_state(episode_tasks, current_time, completed_tasks)
            
            # Get ready tasks
            ready_tasks = episode_tasks.get_ready_tasks(current_time, completed_tasks)
            
            if not ready_tasks:
                current_time += 1.0
                continue
            
            # Choose action (task to schedule)
            if random.random() < epsilon:
                # Exploration
                selected_task = random.choice(ready_tasks)
            else:
                # Exploitation
                selected_task = self._select_best_task(state, ready_tasks)
            
            # Execute action
            from scheduler_base import SimpleScheduler
            simple_scheduler = SimpleScheduler(self.resources)
            schedule_info = simple_scheduler._schedule_task_simple(selected_task, current_time)
            
            if schedule_info:
                schedule.append(schedule_info)
                current_time = schedule_info.end_time_ms
                completed_tasks.add(selected_task.id)
                
                # Calculate immediate reward
                immediate_reward = self._calculate_immediate_reward(schedule_info, selected_task)
                
                # Get next state
                next_state = self._get_state(episode_tasks, current_time, completed_tasks)
                
                # Update Q-value
                action = selected_task.id
                old_value = self.q_table[state][action]
                next_max = max(self.q_table[next_state].values()) if self.q_table[next_state] else 0
                
                new_value = (1 - self.learning_rate) * old_value + \
                           self.learning_rate * (immediate_reward + self.discount_factor * next_max)
                
                self.q_table[state][action] = new_value
            else:
                current_time += 1.0
        
        return schedule
    
    def _get_state(self, tasks: TaskSet, current_time: float, 
                  completed_tasks: Set[str]) -> str:
        """Get current state representation"""
        # Simplified state: number of ready tasks per priority
        ready_counts = defaultdict(int)
        
        ready_tasks = tasks.get_ready_tasks(current_time, completed_tasks)
        for task in ready_tasks:
            ready_counts[task.priority.value] += 1
        
        # Resource utilization levels
        util_levels = []
        for resource in self.resources.values():
            if resource.available_at_ms > current_time:
                util_levels.append('busy')
            else:
                util_levels.append('free')
        
        # Create state string
        state = f"ready_{ready_counts[0]}_{ready_counts[1]}_{ready_counts[2]}_{ready_counts[3]}"
        state += f"_res_{''.join(util_levels)}"
        
        return state
    
    def _select_best_task(self, state: str, ready_tasks: List[NNTask]) -> NNTask:
        """Select best task based on Q-values"""
        best_task = None
        best_value = -float('inf')
        
        for task in ready_tasks:
            value = self.q_table[state][task.id]
            if value > best_value:
                best_value = value
                best_task = task
        
        return best_task or ready_tasks[0]
    
    def _calculate_immediate_reward(self, schedule_info: TaskScheduleInfo, 
                                  task: NNTask) -> float:
        """Calculate immediate reward for scheduling decision"""
        reward = 0.0
        
        # Reward for meeting deadline
        latency = schedule_info.get_latency()
        if latency <= task.constraints.latency_requirement_ms:
            reward += 10.0
        else:
            reward -= 5.0
        
        # Reward based on priority
        reward += (4 - task.priority.value) * 2.0
        
        # Penalty for waiting time
        wait_time = schedule_info.start_time_ms - task.last_scheduled_ms
        if wait_time > task.constraints.get_period_ms():
            reward -= 1.0
        
        return reward
    
    def _calculate_reward(self, schedule: List[TaskScheduleInfo]) -> float:
        """Calculate total reward for a complete schedule"""
        metrics = self.calculate_metrics(schedule)
        
        reward = 0.0
        
        # Reward for short makespan
        if metrics.makespan_ms > 0:
            reward += 1000.0 / metrics.makespan_ms
        
        # Reward for meeting deadlines
        reward += (1.0 - metrics.deadline_miss_rate) * 100
        
        # Reward for high utilization
        avg_util = np.mean(list(metrics.average_utilization.values())) if metrics.average_utilization else 0
        reward += avg_util * 50
        
        # Penalty for high energy
        if metrics.energy_per_task_j > 0:
            reward -= metrics.energy_per_task_j * 10
        
        return reward


class HybridScheduler(BaseScheduler):
    """Hybrid scheduler combining multiple algorithms"""
    
    def __init__(self, resources: Dict[str, ResourceUnit]):
        super().__init__(resources)
        
        # Initialize component schedulers
        self.ga_scheduler = GeneticScheduler(resources, population_size=20, generations=10)
        self.sa_scheduler = SimulatedAnnealingScheduler(resources)
        
    def get_algorithm_name(self) -> str:
        return "Hybrid_GA_SA"
    
    def schedule(self, tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Schedule using hybrid approach"""
        # Phase 1: Use GA to find good initial solution
        ga_schedule = self.ga_scheduler.schedule(tasks, time_limit_ms)
        
        # Convert GA solution to SA format
        sa_solution = self._convert_schedule_to_solution(ga_schedule, tasks)
        
        # Phase 2: Refine with SA
        refined_schedule = self._refine_with_sa(sa_solution, tasks, time_limit_ms)
        
        # Choose best between GA and refined
        ga_metrics = self.calculate_metrics(ga_schedule)
        refined_metrics = self.calculate_metrics(refined_schedule)
        
        ga_score = ga_metrics.calculate_composite_score({
            'makespan': 0.3,
            'latency': 0.2,
            'throughput': 0.2,
            'utilization': 0.2,
            'deadline': 0.1
        })
        
        refined_score = refined_metrics.calculate_composite_score({
            'makespan': 0.3,
            'latency': 0.2,
            'throughput': 0.2,
            'utilization': 0.2,
            'deadline': 0.1
        })
        
        return refined_schedule if refined_score > ga_score else ga_schedule
    
    def _convert_schedule_to_solution(self, schedule: List[TaskScheduleInfo], 
                                    tasks: TaskSet) -> Dict[str, Dict[str, Any]]:
        """Convert schedule to solution format for SA"""
        solution = {}
        
        # Sort schedule by start time to get ordering
        sorted_schedule = sorted(schedule, key=lambda s: s.start_time_ms)
        
        for i, sched in enumerate(sorted_schedule):
            task = tasks.get_task(sched.task_id)
            if task:
                solution[task.id] = {
                    'priority': task.priority.value,
                    'order': i / len(sorted_schedule),  # Normalized order
                    'runtime_type': task.runtime_type.value,
                    'segmentation': task.segmentation_strategy.value,
                    'resource_affinity': {}
                }
        
        return solution
    
    def _refine_with_sa(self, initial_solution: Dict[str, Dict[str, Any]], 
                       tasks: TaskSet, time_limit_ms: float) -> List[TaskScheduleInfo]:
        """Refine solution using simulated annealing"""
        # Use SA with warm start
        sa_scheduler = SimulatedAnnealingScheduler(
            self.resources,
            initial_temp=50.0,  # Lower initial temp since we have good solution
            cooling_rate=0.98,
            iterations_per_temp=5
        )
        
        # Override initial solution generation
        sa_scheduler._generate_initial_solution = lambda t: initial_solution
        
        return sa_scheduler.schedule(tasks, time_limit_ms)
