#!/usr/bin/env python3
"""
演示 time_plan 与 event（事件驱动）两种执行模式的差异。

场景：
- 资源：1×NPU@40，1×DSP@40
- 任务：
  1) T_PRE (DSP, 10ms, CRITICAL)：预处理，周期较低（FPS=10）
  2) T_HIGH (NPU, 5ms, CRITICAL)：依赖 T_PRE，完成后尽快在 NPU 执行
  3) T_LOW  (NPU, 15ms, LOW)：可随时跑的长段

期望：
- time_plan（基于计划的执行）可能在 t=0 就把 T_LOW 排到 NPU 上，导致 T_HIGH 在 ~10ms 就绪时被阻塞到 ~15ms 后。
- event（事件驱动 + slack）在看到 T_HIGH 即将到达时，为其留空或只填充能赶得上的短段，从而显著降低 T_HIGH 的等待/总延迟。
"""

from NNScheduler.core.enums import TaskPriority, ResourceType
from NNScheduler.core.task import NNTask
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.executor import ScheduleExecutor
from NNScheduler.core.executor_event_driven import EventDrivenExecutor
from NNScheduler.core.evaluator import PerformanceEvaluator


def make_tasks():
    # T_PRE: DSP 10ms，FPS=10（min interval=100ms）
    t_pre = NNTask("T_PRE", "Preprocess", priority=TaskPriority.CRITICAL)
    t_pre.set_dsp_only({40.0: 10.0})
    t_pre.set_performance_requirements(fps=10.0, latency=100.0)

    # T_HIGH: NPU 5ms，依赖 T_PRE，优先级高
    t_high = NNTask("T_HIGH", "HighNPU", priority=TaskPriority.CRITICAL)
    t_high.set_npu_only({40.0: 5.0})
    t_high.set_performance_requirements(fps=10.0, latency=20.0)
    t_high.add_dependency("T_PRE")

    # T_LOW: NPU 15ms，低优先级，容易挡道
    t_low = NNTask("T_LOW", "LowNPU", priority=TaskPriority.LOW)
    t_low.set_npu_only({40.0: 15.0})
    t_low.set_performance_requirements(fps=50.0, latency=100.0)

    return [t_pre, t_high, t_low]


def run_time_plan():
    qm = ResourceQueueManager()
    qm.add_resource("NPU_0", ResourceType.NPU, 40.0)
    qm.add_resource("DSP_0", ResourceType.DSP, 40.0)
    tracer = ScheduleTracer(qm)
    launcher = TaskLauncher(qm, tracer)
    tasks = make_tasks()
    for t in tasks:
        launcher.register_task(t)
    time_window = 200.0
    # 使用 eager 计划，展示时间表模式可能先占用NPU导致T_HIGH等待
    plan = launcher.create_launch_plan(time_window, "eager")
    ex = ScheduleExecutor(qm, tracer, launcher.tasks)
    ex.execute_plan(plan, time_window, segment_mode=True)
    ev = PerformanceEvaluator(tracer, launcher.tasks, qm)
    metrics = ev.evaluate(time_window, plan.events)
    return metrics, tracer


def run_event():
    qm = ResourceQueueManager()
    qm.add_resource("NPU_0", ResourceType.NPU, 40.0)
    qm.add_resource("DSP_0", ResourceType.DSP, 40.0)
    tracer = ScheduleTracer(qm)
    tasks = {t.task_id: t for t in make_tasks()}
    ev_exec = EventDrivenExecutor(qm, tracer, tasks)
    res = ev_exec.execute(200.0, segment_mode=True, launch_strategy="balanced")
    ev = PerformanceEvaluator(tracer, tasks, qm)
    metrics = ev.evaluate(200.0, ev_exec.launch_events)
    return metrics, tracer


def main():
    m_plan, tr_plan = run_time_plan()
    m_event, tr_event = run_event()

    def print_key(title, m):
        print(f"\n=== {title} ===")
        print(f"avg_latency={m.avg_latency:.2f}ms, max_latency={m.max_latency:.2f}ms")
        print(f"NPU_util={m.avg_npu_utilization:.1f}%, DSP_util={m.avg_dsp_utilization:.1f}%")

    print_key("time_plan", m_plan)
    print_key("event", m_event)

    # 重点对比 T_HIGH 的延迟
    def high_latency(tracer):
        # 仅统计实例#0 的端到端延迟，避免多个实例跨度过大
        hs = [e for e in tracer.executions if e.task_id.startswith("T_HIGH#0")]
        if not hs:
            return None
        st = min(x.start_time for x in hs)
        ed = max(x.end_time for x in hs)
        return ed - st

    lat_plan = high_latency(tr_plan)
    lat_event = high_latency(tr_event)
    print(f"\nT_HIGH end-to-end latency: time_plan={lat_plan:.2f}ms, event={lat_event:.2f}ms")

    if lat_event is not None and lat_plan is not None:
        diff = lat_plan - lat_event
        print(f"Improvement (plan - event) = {diff:.2f}ms")


if __name__ == "__main__":
    main()
