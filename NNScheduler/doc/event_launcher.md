确认两点我都帮你完成好了：

- 优化环路用事件驱动
  - 在 evaluate_configuration 中新增了 event 分支，优化器在评分每一组优先级配置时也可以走事件驱动执行器（和最终展示执行语义一致），由 JSON 开关控制。
  - 接线方式：
    - 新增参数传递：在构建 JsonPriorityOptimizer 时从 JSON 读取 optimization.executor_mode 并传入优化器
      - NNScheduler/interface/optimization_interface.py:101
    - 优化器新增属性并在评估时分支执行
      - NNScheduler/interface/optimization_interface.py:292（executor_mode 属性）
      - NNScheduler/interface/optimization_interface.py:456-465（event 分支）/ 466-470（time_plan 分支）
  - 用法：在你的 JSON（比如 config_hybrid_balanced_200ms.json）的 optimization 段添加
    - "executor_mode": "event"
    - 其余配置不变。若不指定则默认 time_plan。
- 对比 demo（time_plan vs event）
  - 新增文件：test/NNScheduler/demo_executor_modes.py
  - 设计
    - 资源：1×NPU@40、1×DSP@40
    - 任务：
      - T_PRE (DSP, 10ms, CRITICAL)：预处理，FPS=10
      - T_HIGH (NPU, 5ms, CRITICAL)：依赖 T_PRE，希望尽早跑
      - T_LOW (NPU, 15ms, LOW)：容易挡道的长段
    - 运行两次：
      - time_plan：使用 TaskLauncher.create_launch_plan(..., "eager") + ScheduleExecutor
      - event：使用 EventDrivenExecutor（balanced + slack gating）
    - 输出关键指标及 T_HIGH#0 的端到端延迟（以及首次开始时间，可自行添加打印）
  - 运行（ais 环境）：
    - conda run -n ais python test/NNScheduler/demo_executor_modes.py
  - 说明
    - 这个 demo 的意图是展示“time_plan（尤其是 eager）在起点可能先把 NPU 占满，而 event 模式会为即将到来的高优留出 slack 保护”，从而在某些参数下 event 能降低 T_HIGH 的等待。
    - 由于你的 Balanced 计划器和分段估时较智能，有时两者差异不明显。如果你想放大差异，建议：
      - 把 time_plan 改为 "eager"（demo 里已设置）；
      - 增大 T_LOW 的时长（比如 20ms+），并调小 T_PRE 使 T_HIGH 的 ETA 更“紧”；
      - 降低 T_LOW 优先级或增大 event 的 slack（默认 NPU≈1.0ms，DSP≈0.8ms，event 执行器里可调）。

运行参考

- 让优化环路也走事件驱动（示例）：
  - 生成一个事件驱动版本配置并运行
    - python3 - << 'PY' 脚本（我刚刚跑过）：tmp_runs/config_hybrid_balanced_200ms_event_loop.json 已生成
    - conda run -n ais python main_api.py tmp_runs/config_hybrid_balanced_200ms_event_loop.json -o out/hybrid_balanced_200_event_loop
  - 注意：事件驱动的优化环路与 time_plan 的评分语义不同，结果分布可能明显变化（这符合预期，因为执行模型不同）。如果你需要绝对可重复，还可以再加固定随机种子（random/numpy）我也可以顺手接上。

文件改动清单（核心）

- 新增事件执行器：NNScheduler/core/executor_event_driven.py
- 优化环路接 event：
  - NNScheduler/interface/optimization_interface.py: optimizer 构造传入 executor_mode
  - NNScheduler/interface/optimization_interface.py: evaluate_configuration 支持 event 分支
- 对比 demo：test/NNScheduler/demo_executor_modes.py

是否要把 event 执行器的 slack 参数也做成 JSON 可配（比如 optimization.slack_ms.NPU/DSP）？我可以再补一个小开关，便于你从配置上微调。