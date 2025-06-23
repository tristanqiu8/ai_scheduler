#!/usr/bin/env python3
"""
Core enumeration definitions for heterogeneous DSP+NPU scheduler
"""

from enum import Enum, IntEnum, auto


class ResourceType(Enum):
    """Hardware resource types"""
    NPU = "NPU"
    DSP = "DSP"
    CPU = "CPU"  # For pre/post processing
    
    def __str__(self):
        return self.value


class TaskPriority(IntEnum):
    """Task priority levels (0 = highest priority)"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    
    @classmethod
    def from_string(cls, priority_str: str) -> 'TaskPriority':
        """Create priority from string"""
        return cls[priority_str.upper()]
    
    def wait_time_ms(self) -> float:
        """Get wait time before this priority can execute"""
        wait_times = {
            TaskPriority.CRITICAL: 0.0,
            TaskPriority.HIGH: 0.05,
            TaskPriority.NORMAL: 0.10,
            TaskPriority.LOW: 0.15
        }
        return wait_times.get(self, 0.05)


class RuntimeType(Enum):
    """Runtime execution modes"""
    ACPU_RUNTIME = "ACPU"  # NPU can be preempted, flexible scheduling
    DSP_RUNTIME = "DSP"    # DSP+NPU bound together, non-preemptable
    
    def is_preemptable(self) -> bool:
        """Check if runtime allows preemption"""
        return self == RuntimeType.ACPU_RUNTIME
    
    def requires_binding(self) -> bool:
        """Check if runtime requires resource binding"""
        return self == RuntimeType.DSP_RUNTIME


class SegmentationStrategy(Enum):
    """Network segmentation strategies"""
    NO_SEGMENTATION = "none"              # Keep original segments
    FIXED_SEGMENTATION = "fixed"          # Use predefined cut points
    ADAPTIVE_SEGMENTATION = "adaptive"    # Dynamic selection based on resources
    AGGRESSIVE_SEGMENTATION = "aggressive" # Maximum parallelism
    BALANCED_SEGMENTATION = "balanced"    # Balance overhead vs benefit
    
    def should_segment(self) -> bool:
        """Check if strategy involves segmentation"""
        return self != SegmentationStrategy.NO_SEGMENTATION


class SchedulingAlgorithm(Enum):
    """Available scheduling algorithms"""
    # Simple algorithms
    FIFO = "fifo"                    # First In First Out
    PRIORITY_BASED = "priority"      # Simple priority-based
    ROUND_ROBIN = "round_robin"      # Round-robin with priority
    EDF = "edf"                      # Earliest Deadline First
    
    # Advanced algorithms
    GENETIC = "genetic"              # Genetic Algorithm
    SIMULATED_ANNEALING = "sa"       # Simulated Annealing
    PARTICLE_SWARM = "pso"           # Particle Swarm Optimization
    REINFORCEMENT_LEARNING = "rl"    # RL-based scheduling
    HYBRID_GA_SA = "hybrid_ga_sa"    # Hybrid Genetic + SA
    
    def is_simple(self) -> bool:
        """Check if this is a simple algorithm"""
        return self in [
            SchedulingAlgorithm.FIFO,
            SchedulingAlgorithm.PRIORITY_BASED,
            SchedulingAlgorithm.ROUND_ROBIN,
            SchedulingAlgorithm.EDF
        ]
    
    def is_metaheuristic(self) -> bool:
        """Check if this is a metaheuristic algorithm"""
        return self in [
            SchedulingAlgorithm.GENETIC,
            SchedulingAlgorithm.SIMULATED_ANNEALING,
            SchedulingAlgorithm.PARTICLE_SWARM,
            SchedulingAlgorithm.HYBRID_GA_SA
        ]


class TaskState(Enum):
    """Task execution states"""
    PENDING = "pending"          # Waiting to be scheduled
    READY = "ready"              # Ready to execute
    RUNNING = "running"          # Currently executing
    PREEMPTED = "preempted"      # Preempted by higher priority
    BLOCKED = "blocked"          # Blocked on dependencies
    COMPLETED = "completed"      # Execution completed
    FAILED = "failed"            # Execution failed


class OptimizationObjective(Enum):
    """Optimization objectives for scheduling"""
    MINIMIZE_MAKESPAN = "makespan"           # Total schedule length
    MINIMIZE_LATENCY = "latency"             # Average task latency
    MAXIMIZE_THROUGHPUT = "throughput"       # Tasks completed per second
    MINIMIZE_ENERGY = "energy"               # Energy consumption
    MAXIMIZE_UTILIZATION = "utilization"     # Resource utilization
    MINIMIZE_DEADLINE_MISS = "deadline_miss" # Missed deadlines
    BALANCED = "balanced"                    # Multi-objective balance


class CutPointType(Enum):
    """Types of network cut points"""
    LAYER_BOUNDARY = "layer"         # Between layers
    TENSOR_SPLIT = "tensor"          # Split within tensor
    OPERATOR_BOUNDARY = "operator"   # Between operators
    MEMORY_BOUNDARY = "memory"       # Based on memory constraints
    COMPUTE_BOUNDARY = "compute"     # Based on compute balance


class ResourceAllocationPolicy(Enum):
    """Resource allocation policies"""
    FIRST_FIT = "first_fit"          # First available resource
    BEST_FIT = "best_fit"            # Best performance match
    WORST_FIT = "worst_fit"          # Load balancing
    THERMAL_AWARE = "thermal"        # Consider thermal constraints
    ENERGY_AWARE = "energy"          # Consider energy efficiency
    AFFINITY_BASED = "affinity"      # Consider task-resource affinity


class PreemptionPolicy(Enum):
    """Preemption policies for ACPU runtime"""
    NO_PREEMPTION = "none"           # No preemption allowed
    PRIORITY_PREEMPTION = "priority" # Higher priority can preempt
    DEADLINE_PREEMPTION = "deadline" # Urgent deadline can preempt
    QUANTUM_PREEMPTION = "quantum"   # Time-slice based preemption


class SchedulingConstraint(Enum):
    """Types of scheduling constraints"""
    HARD_DEADLINE = "hard_deadline"   # Must meet deadline
    SOFT_DEADLINE = "soft_deadline"   # Try to meet deadline
    DEPENDENCY = "dependency"         # Task dependencies
    RESOURCE_EXCLUSIVE = "exclusive"  # Exclusive resource access
    PRECEDENCE = "precedence"         # Execution order
    PERIODIC = "periodic"             # Periodic execution
    AFFINITY = "affinity"            # Resource affinity
    ANTI_AFFINITY = "anti_affinity"  # Resource anti-affinity


class PerformanceMetric(Enum):
    """Performance metrics for evaluation"""
    LATENCY = "latency"                  # Task execution latency
    THROUGHPUT = "throughput"            # System throughput
    UTILIZATION = "utilization"          # Resource utilization
    RESPONSE_TIME = "response_time"      # Task response time
    TURNAROUND_TIME = "turnaround_time"  # Total task time
    WAITING_TIME = "waiting_time"        # Time spent waiting
    DEADLINE_MISS_RATE = "miss_rate"     # Deadline miss percentage
    ENERGY_EFFICIENCY = "energy_eff"     # Performance per watt
    THERMAL_EFFICIENCY = "thermal_eff"   # Thermal management
    SCHEDULING_OVERHEAD = "overhead"     # Scheduling algorithm overhead


# Configuration constants
class SchedulerConfig:
    """Global scheduler configuration constants"""
    # Timing constants (in milliseconds)
    DEFAULT_CUT_OVERHEAD_MS = 0.12
    PRIORITY_WAIT_TIME_MS = 0.05
    MIN_TASK_DURATION_MS = 0.1
    MAX_SCHEDULING_TIME_MS = 10.0
    
    # Resource constants
    MAX_PRIORITY_LEVELS = 4
    MAX_PREEMPTION_COUNT = 3
    DEFAULT_TIME_QUANTUM_MS = 5.0
    
    # Optimization constants
    GA_POPULATION_SIZE = 100
    GA_GENERATIONS = 50
    SA_INITIAL_TEMPERATURE = 100.0
    SA_COOLING_RATE = 0.95
    PSO_PARTICLE_COUNT = 50
    PSO_ITERATIONS = 100
    
    # Memory constants (in MB)
    DEFAULT_MEMORY_LIMIT = 1024
    MIN_MEMORY_PER_SEGMENT = 32
    
    # Thermal constants
    MAX_TEMPERATURE_C = 85.0
    THERMAL_THROTTLE_TEMP_C = 80.0
    AMBIENT_TEMPERATURE_C = 25.0
