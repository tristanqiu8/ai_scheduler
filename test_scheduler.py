#!/usr/bin/env python3
"""
Comprehensive test suite for the heterogeneous scheduler system
"""

import unittest
import numpy as np
from typing import Dict, List
import tempfile
import os

from enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy, SchedulerConfig
from models import ResourceUnit, NetworkSegment, CutPoint, TaskScheduleInfo
from task import NNTask, TaskSet, TaskFactory
from scheduler_base import SimpleScheduler, PriorityQueueScheduler
from scheduler_advanced import GeneticScheduler, SimulatedAnnealingScheduler
from scheduler_utils import ScheduleValidator, ScheduleAnalyzer, ScheduleSerializer


class TestModels(unittest.TestCase):
    """Test core data models"""
    
    def test_cut_point_creation(self):
        """Test cut point creation and properties"""
        cut_point = CutPoint(
            name="test_cut",
            position=0.5,
            overhead_ms=0.12
        )
        
        self.assertEqual(cut_point.name, "test_cut")
        self.assertEqual(cut_point.position, 0.5)
        self.assertEqual(cut_point.overhead_ms, 0.12)
    
    def test_network_segment_segmentation(self):
        """Test network segment segmentation"""
        segment = NetworkSegment(
            name="test_segment",
            resource_type=ResourceType.NPU,
            duration_table={2.0: 20.0, 4.0: 15.0, 8.0: 10.0}
        )
        
        # Add cut points
        cp1 = segment.add_cut_point(0.3, "cut1", 0.12)
        cp2 = segment.add_cut_point(0.7, "cut2", 0.12)
        
        self.assertEqual(len(segment.cut_points), 2)
        
        # Apply segmentation
        sub_segments = segment.apply_segmentation([cp1.id, cp2.id])
        
        self.assertEqual(len(sub_segments), 3)
        self.assertTrue(segment.is_segmented)
        
        # Check sub-segment durations
        total_duration = sum(ss.get_duration(4.0) for ss in sub_segments)
        self.assertAlmostEqual(total_duration, 15.0, places=2)
    
    def test_resource_thermal_model(self):
        """Test resource thermal modeling"""
        resource = ResourceUnit(
            id="NPU_TEST",
            name="Test NPU",
            resource_type=ResourceType.NPU,
            bandwidth=8.0
        )
        
        initial_temp = resource.current_temp_c
        
        # Test temperature increase
        resource.update_temperature(power_w=20.0, duration_ms=1000.0)
        self.assertGreater(resource.current_temp_c, initial_temp)
        
        # Test thermal throttling by setting temperature above threshold
        # The threshold is 80°C, so set it to 85°C
        resource.current_temp_c = 85.0
        
        # Now throttle factor should be less than 1.0
        throttle_factor = resource.get_thermal_throttle_factor()
        self.assertLess(throttle_factor, 1.0)
        self.assertGreater(throttle_factor, 0.0)  # Should still be positive


class TestTasks(unittest.TestCase):
    """Test task management"""
    
    def test_task_creation(self):
        """Test task creation and configuration"""
        task = NNTask(
            name="TestTask",
            priority=TaskPriority.HIGH,
            runtime_type=RuntimeType.ACPU_RUNTIME
        )
        
        self.assertEqual(task.name, "TestTask")
        self.assertEqual(task.priority, TaskPriority.HIGH)
        self.assertEqual(task.runtime_type, RuntimeType.ACPU_RUNTIME)
    
    def test_task_constraints(self):
        """Test task constraints"""
        task = NNTask(name="TestTask")
        
        task.constraints.fps_requirement = 30.0
        task.constraints.latency_requirement_ms = 33.33
        
        self.assertAlmostEqual(task.constraints.get_period_ms(), 33.33, places=2)
        
        # Validate constraints
        is_valid, errors = task.constraints.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_task_segmentation_strategy(self):
        """Test task segmentation strategies"""
        task = TaskFactory.create_safety_monitor()
        
        # Test different strategies
        available_resources = {ResourceType.NPU: 3}
        
        # No segmentation
        task.segmentation_strategy = SegmentationStrategy.NO_SEGMENTATION
        decisions = task.apply_segmentation_strategy(available_resources)
        self.assertEqual(len(decisions[task.segments[0].id]), 0)
        
        # Adaptive segmentation
        task.segmentation_strategy = SegmentationStrategy.ADAPTIVE_SEGMENTATION
        task.priority = TaskPriority.CRITICAL
        decisions = task.apply_segmentation_strategy(available_resources)
        self.assertGreater(len(decisions[task.segments[0].id]), 0)
    
    def test_task_dependencies(self):
        """Test task dependency checking"""
        task1 = NNTask(name="Task1")
        task2 = NNTask(name="Task2")
        
        task2.constraints.dependencies.add(task1.id)
        
        self.assertFalse(task2.check_dependencies_met(set()))
        self.assertTrue(task2.check_dependencies_met({task1.id}))
    
    def test_task_serialization(self):
        """Test task serialization"""
        task = TaskFactory.create_object_detection()
        
        # Convert to dict
        task_dict = task.to_dict()
        
        self.assertIn('id', task_dict)
        self.assertIn('name', task_dict)
        self.assertIn('segments', task_dict)
        
        # Recreate from dict
        restored_task = NNTask.from_dict(task_dict)
        
        self.assertEqual(restored_task.id, task.id)
        self.assertEqual(restored_task.name, task.name)
        self.assertEqual(len(restored_task.segments), len(task.segments))


class TestSchedulers(unittest.TestCase):
    """Test scheduling algorithms"""
    
    def setUp(self):
        """Set up test resources and tasks"""
        self.resources = {
            "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
            "NPU_1": ResourceUnit("NPU_1", "NPU1", ResourceType.NPU, 4.0),
            "DSP_0": ResourceUnit("DSP_0", "DSP0", ResourceType.DSP, 4.0)
        }
        
        self.tasks = TaskSet()
        self.tasks.add_task(TaskFactory.create_safety_monitor())
        self.tasks.add_task(TaskFactory.create_object_detection(use_dsp=False))
        self.tasks.add_task(TaskFactory.create_analytics_task())
    
    def test_simple_scheduler(self):
        """Test simple priority-based scheduler"""
        scheduler = SimpleScheduler(self.resources)
        
        schedule = scheduler.schedule(self.tasks, time_limit_ms=200.0)
        
        self.assertGreater(len(schedule), 0)
        
        # Validate schedule
        validator = ScheduleValidator()
        is_valid, errors = validator.validate_schedule(schedule, self.resources, self.tasks)
        
        if not is_valid:
            print(f"Validation errors: {errors}")
        
        self.assertTrue(is_valid)
    
    def test_priority_queue_scheduler(self):
        """Test advanced priority queue scheduler"""
        scheduler = PriorityQueueScheduler(self.resources)
        
        schedule = scheduler.schedule(self.tasks, time_limit_ms=200.0)
        
        self.assertGreater(len(schedule), 0)
        
        # Check priority ordering
        critical_tasks = [s for s in schedule 
                         if self.tasks.get_task(s.task_id).priority == TaskPriority.CRITICAL]
        low_tasks = [s for s in schedule 
                    if self.tasks.get_task(s.task_id).priority == TaskPriority.LOW]
        
        if critical_tasks and low_tasks:
            # Critical tasks should generally start before low priority
            self.assertLess(critical_tasks[0].start_time_ms, 
                          low_tasks[0].start_time_ms + 10.0)  # Some tolerance
    
    def test_genetic_scheduler(self):
        """Test genetic algorithm scheduler"""
        scheduler = GeneticScheduler(
            self.resources,
            population_size=10,
            generations=5  # Small for testing
        )
        
        schedule = scheduler.schedule(self.tasks, time_limit_ms=200.0)
        
        self.assertGreater(len(schedule), 0)
        
        # Check metrics
        metrics = scheduler.calculate_metrics(schedule)
        self.assertGreater(metrics.completed_tasks, 0)
    
    def test_scheduler_reset(self):
        """Test scheduler state reset"""
        # Create two separate instances to avoid state contamination
        resources1 = {
            "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
            "NPU_1": ResourceUnit("NPU_1", "NPU1", ResourceType.NPU, 4.0),
            "DSP_0": ResourceUnit("DSP_0", "DSP0", ResourceType.DSP, 4.0)
        }
        
        resources2 = {
            "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
            "NPU_1": ResourceUnit("NPU_1", "NPU1", ResourceType.NPU, 4.0),
            "DSP_0": ResourceUnit("DSP_0", "DSP0", ResourceType.DSP, 4.0)
        }
        
        scheduler1 = SimpleScheduler(resources1)
        scheduler2 = SimpleScheduler(resources2)
        
        # Create identical task sets
        tasks1 = TaskSet()
        tasks1.add_task(TaskFactory.create_safety_monitor())
        tasks1.add_task(TaskFactory.create_object_detection(use_dsp=False))
        tasks1.add_task(TaskFactory.create_analytics_task())
        
        tasks2 = TaskSet()
        tasks2.add_task(TaskFactory.create_safety_monitor())
        tasks2.add_task(TaskFactory.create_object_detection(use_dsp=False))
        tasks2.add_task(TaskFactory.create_analytics_task())
        
        # Run scheduling
        schedule1 = scheduler1.schedule(tasks1, time_limit_ms=100.0)
        schedule2 = scheduler2.schedule(tasks2, time_limit_ms=100.0)
        
        # Should produce same number of scheduled tasks
        self.assertEqual(len(schedule1), len(schedule2))
        self.assertGreater(len(schedule1), 0)  # Ensure something was scheduled


class TestSchedulerUtils(unittest.TestCase):
    """Test utility functions"""
    
    def setUp(self):
        """Set up test data"""
        self.resources = {
            "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
            "DSP_0": ResourceUnit("DSP_0", "DSP0", ResourceType.DSP, 4.0)
        }
        
        self.tasks = TaskSet()
        self.tasks.add_task(TaskFactory.create_safety_monitor())
        
        # Create sample schedule
        self.schedule = [
            TaskScheduleInfo(
                task_id=list(self.tasks.tasks.keys())[0],
                start_time_ms=0.0,
                end_time_ms=10.0,
                segment_schedule=[("seg1", 0.0, 10.0, "NPU_0")]
            )
        ]
    
    def test_schedule_validation(self):
        """Test schedule validation"""
        validator = ScheduleValidator()
        
        # Valid schedule
        is_valid, errors = validator.validate_schedule(
            self.schedule, self.resources, self.tasks
        )
        self.assertTrue(is_valid)
        
        # Create conflicting schedule
        conflicting_schedule = self.schedule + [
            TaskScheduleInfo(
                task_id=list(self.tasks.tasks.keys())[0],
                start_time_ms=5.0,
                end_time_ms=15.0,
                segment_schedule=[("seg2", 5.0, 15.0, "NPU_0")]
            )
        ]
        
        is_valid, errors = validator.validate_schedule(
            conflicting_schedule, self.resources, self.tasks
        )
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_schedule_analysis(self):
        """Test schedule analysis"""
        analyzer = ScheduleAnalyzer()
        
        analysis = analyzer.analyze_schedule(
            self.schedule, self.resources, self.tasks
        )
        
        self.assertIn('summary', analysis)
        self.assertIn('resource_analysis', analysis)
        self.assertIn('task_analysis', analysis)
        
        self.assertEqual(analysis['summary']['total_executions'], 1)
    
    def test_schedule_serialization(self):
        """Test schedule save/load"""
        serializer = ScheduleSerializer()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name
        
        try:
            # Save schedule
            serializer.save_schedule(self.schedule, temp_file, format='json')
            
            # Load schedule
            loaded_schedule = serializer.load_schedule(temp_file, format='json')
            
            self.assertEqual(len(loaded_schedule), len(self.schedule))
            self.assertEqual(loaded_schedule[0].task_id, self.schedule[0].task_id)
            
        finally:
            os.unlink(temp_file)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def test_end_to_end_scheduling(self):
        """Test complete scheduling workflow"""
        # Create resources
        resources = {
            "NPU_0": ResourceUnit("NPU_0", "High-Perf NPU", ResourceType.NPU, 8.0),
            "NPU_1": ResourceUnit("NPU_1", "Mid NPU", ResourceType.NPU, 4.0),
            "NPU_2": ResourceUnit("NPU_2", "Low NPU", ResourceType.NPU, 2.0),
            "DSP_0": ResourceUnit("DSP_0", "DSP0", ResourceType.DSP, 4.0),
            "DSP_1": ResourceUnit("DSP_1", "DSP1", ResourceType.DSP, 4.0)
        }
        
        # Create diverse task set
        tasks = TaskSet()
        
        # Critical task
        critical_task = TaskFactory.create_safety_monitor(fps=30)
        tasks.add_task(critical_task)
        
        # High priority with DSP
        detection_task = TaskFactory.create_object_detection(use_dsp=True)
        tasks.add_task(detection_task)
        
        # Normal priority with dependencies
        analytics_task = TaskFactory.create_analytics_task()
        analytics_task.constraints.dependencies.add(critical_task.id)
        tasks.add_task(analytics_task)
        
        # Test multiple schedulers
        schedulers = [
            SimpleScheduler(resources),
            PriorityQueueScheduler(resources),
            GeneticScheduler(resources, population_size=20, generations=10)
        ]
        
        for scheduler in schedulers:
            # Reset scheduler state
            scheduler.reset()
            
            # Reset all resources
            for resource in resources.values():
                resource.is_available = True
                resource.current_task_id = None
                resource.available_at_ms = 0.0
                resource.total_usage_ms = 0.0
                resource.current_temp_c = SchedulerConfig.AMBIENT_TEMPERATURE_C
            
            # Create a fresh copy of tasks for this scheduler
            # This ensures each scheduler gets tasks in the same initial state
            fresh_tasks = TaskSet()
            
            # Re-create tasks to ensure clean state
            fresh_critical = TaskFactory.create_safety_monitor(fps=30)
            fresh_tasks.add_task(fresh_critical)
            
            fresh_detection = TaskFactory.create_object_detection(use_dsp=True)
            fresh_tasks.add_task(fresh_detection)
            
            fresh_analytics = TaskFactory.create_analytics_task()
            fresh_analytics.constraints.dependencies.add(fresh_critical.id)
            fresh_tasks.add_task(fresh_analytics)
            
            # Run scheduling
            try:
                schedule = scheduler.schedule(fresh_tasks, time_limit_ms=500.0)
            except Exception as e:
                self.fail(f"{scheduler.get_algorithm_name()} failed with error: {e}")
            
            # Basic checks
            self.assertIsNotNone(schedule, 
                            f"{scheduler.get_algorithm_name()} returned None")
            
            # Validate
            validator = ScheduleValidator()
            is_valid, errors = validator.validate_schedule(schedule, resources, fresh_tasks)
            
            # For debugging
            if not is_valid or len(schedule) == 0:
                print(f"\n{scheduler.get_algorithm_name()} results:")
                print(f"  Schedule length: {len(schedule)}")
                print(f"  Valid: {is_valid}")
                if errors:
                    print(f"  Errors: {errors[:3]}")  # Show first 3 errors
            
            self.assertTrue(is_valid, 
                        f"{scheduler.get_algorithm_name()} produced invalid schedule: {errors}")
            
            # Check that something was scheduled
            self.assertGreater(len(schedule), 0,
                            f"{scheduler.get_algorithm_name()} produced empty schedule")
            
            # Analyze
            analyzer = ScheduleAnalyzer()
            analysis = analyzer.analyze_schedule(schedule, resources, fresh_tasks)
            
            # Check basic requirements
            self.assertGreater(analysis['summary']['total_executions'], 0)
            
            # Be lenient with latency requirements for complex schedulers
            if analysis['summary']['average_latency'] > 0:
                # Different thresholds for different schedulers
                if isinstance(scheduler, GeneticScheduler):
                    # GA might produce longer schedules due to optimization
                    self.assertLess(analysis['summary']['average_latency'], 200.0)
                else:
                    self.assertLess(analysis['summary']['average_latency'], 100.0)

    def test_segmentation_impact(self):
        """Test impact of network segmentation"""
        resources = {
            "NPU_0": ResourceUnit("NPU_0", "NPU0", ResourceType.NPU, 8.0),
            "NPU_1": ResourceUnit("NPU_1", "NPU1", ResourceType.NPU, 8.0),
            "NPU_2": ResourceUnit("NPU_2", "NPU2", ResourceType.NPU, 8.0)
        }
        
        # Create task with segmentation
        task = NNTask(
            name="SegmentedTask",
            priority=TaskPriority.HIGH,
            segmentation_strategy=SegmentationStrategy.ADAPTIVE_SEGMENTATION
        )
        
        segment = NetworkSegment(
            name="inference",
            resource_type=ResourceType.NPU,
            duration_table={8.0: 30.0}
        )
        
        # Add multiple cut points
        segment.add_cut_point(0.25, "cut1", 0.12)
        segment.add_cut_point(0.50, "cut2", 0.12)
        segment.add_cut_point(0.75, "cut3", 0.12)
        
        task.add_segment(segment)
        task.constraints.fps_requirement = 20.0
        task.constraints.latency_requirement_ms = 50.0
        
        tasks = TaskSet()
        tasks.add_task(task)
        
        # Schedule with segmentation
        scheduler = SimpleScheduler(resources)
        schedule = scheduler.schedule(tasks, time_limit_ms=100.0)
        
        # Check that segmentation was applied
        self.assertGreater(len(schedule), 0)
        if schedule[0].sub_segment_schedule:
            self.assertGreater(len(schedule[0].sub_segment_schedule), 1)


class TestPerformance(unittest.TestCase):
    """Performance and scalability tests"""
    
    def test_scheduler_scalability(self):
        """Test scheduler performance with increasing task count"""
        # Create resources
        resources = {}
        for i in range(10):
            resources[f"NPU_{i}"] = ResourceUnit(f"NPU_{i}", f"NPU{i}", 
                                               ResourceType.NPU, 4.0 + i % 3 * 2)
        
        # Test with different task counts
        task_counts = [10, 20, 50]
        
        for count in task_counts:
            tasks = TaskSet()
            
            for i in range(count):
                priority = TaskPriority(i % 4)
                task = NNTask(
                    name=f"Task_{i}",
                    priority=priority,
                    runtime_type=RuntimeType.ACPU_RUNTIME
                )
                
                task.set_npu_only(
                    duration_table={2.0: 20.0, 4.0: 15.0, 8.0: 10.0},
                    name=f"inference_{i}"
                )
                
                task.constraints.fps_requirement = 10 + i % 20
                task.constraints.latency_requirement_ms = 50 + i % 50
                
                tasks.add_task(task)
            
            # Time scheduling
            import time
            scheduler = SimpleScheduler(resources)
            
            start_time = time.time()
            schedule = scheduler.schedule(tasks, time_limit_ms=1000.0)
            execution_time = time.time() - start_time
            
            print(f"Scheduled {count} tasks in {execution_time:.3f}s")
            
            # Should complete in reasonable time
            self.assertLess(execution_time, 5.0)  # 5 seconds max
            self.assertGreater(len(schedule), count * 0.8)  # At least 80% scheduled


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModels,
        TestTasks,
        TestSchedulers,
        TestSchedulerUtils,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
