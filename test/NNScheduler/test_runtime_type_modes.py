from collections import defaultdict

import pytest

from NNScheduler.core.enums import RuntimeType, ResourceType
from NNScheduler.core.task import NNTask
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.executor import ScheduleExecutor


def _run_simple_schedule(hybrid_runtime: RuntimeType):
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 100.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 100.0)

    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)

    hybrid = NNTask("HYBRID", runtime_type=hybrid_runtime)
    hybrid.add_segment(ResourceType.NPU, {100.0: 5.0}, "npu_stage")
    hybrid.add_segment(ResourceType.DSP, {100.0: 5.0}, "dsp_stage")
    launcher.register_task(hybrid)

    dsp_only = NNTask("DSP_ONLY")
    dsp_only.add_segment(ResourceType.DSP, {100.0: 3.0}, "dsp_only")
    launcher.register_task(dsp_only)

    plan = launcher.create_launch_plan(40.0, "eager")
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    executor.segment_mode = True
    executor.execute_plan(plan, 40.0)

    hybrid_dsp_exec = next(
        exec_record
        for exec_record in tracer.executions
        if exec_record.root_task_id == "HYBRID" and exec_record.resource_id == "DSP_0"
    )
    dsp_only_exec = next(
        exec_record
        for exec_record in tracer.executions
        if exec_record.root_task_id == "DSP_ONLY"
    )

    return (
        hybrid_dsp_exec.start_time,
        hybrid_dsp_exec.end_time,
        dsp_only_exec.start_time,
    )


def test_acpu_runtime_allows_parallel_dsp_usage():
    hybrid_dsp_start, _, dsp_only_start = _run_simple_schedule(RuntimeType.ACPU_RUNTIME)
    assert dsp_only_start < hybrid_dsp_start


def test_dsp_runtime_enforces_resource_binding():
    hybrid_dsp_start, hybrid_dsp_end, dsp_only_start = _run_simple_schedule(RuntimeType.DSP_RUNTIME)
    assert dsp_only_start >= hybrid_dsp_end - 1e-6


def _run_dual_hybrid_schedule():
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 100.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 100.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 100.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 100.0)

    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)

    hybrid_a = NNTask("HYBRID_A", runtime_type=RuntimeType.DSP_RUNTIME)
    hybrid_a.add_segment(ResourceType.NPU, {100.0: 4.0}, "npu_front")
    hybrid_a.add_segment(ResourceType.DSP, {100.0: 4.0}, "dsp_back")
    launcher.register_task(hybrid_a)

    hybrid_b = NNTask("HYBRID_B", runtime_type=RuntimeType.DSP_RUNTIME)
    hybrid_b.add_segment(ResourceType.NPU, {100.0: 4.5}, "npu_front")
    hybrid_b.add_segment(ResourceType.DSP, {100.0: 4.5}, "dsp_back")
    launcher.register_task(hybrid_b)

    plan = launcher.create_launch_plan(60.0, "balanced")
    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    executor.segment_mode = True
    executor.execute_plan(plan, 60.0)

    return tracer.executions


def test_dsp_runtime_pairs_npu_and_dsp_indices():
    executions = _run_dual_hybrid_schedule()

    usage_by_task = defaultdict(lambda: {ResourceType.NPU: set(), ResourceType.DSP: set()})

    for record in executions:
        if record.root_task_id not in {"HYBRID_A", "HYBRID_B"}:
            continue

        suffix = record.resource_id.split("_")[-1]
        if not suffix.isdigit():
            continue

        index = int(suffix)
        usage_by_task[record.root_task_id][record.resource_type].add(index)

    for task_id, usage in usage_by_task.items():
        assert usage[ResourceType.NPU], f"{task_id} missing NPU execution"
        assert usage[ResourceType.DSP], f"{task_id} missing DSP execution"
        assert len(usage[ResourceType.NPU]) == 1
        assert len(usage[ResourceType.DSP]) == 1
        assert usage[ResourceType.NPU] == usage[ResourceType.DSP]
