# Linesync Sync Scheduling Scenario

本示例展示如何利用自定义 `sync` 发射策略，实现两个 ISP→NPU 任务在资源上紧密交错运行的流水效果。

## 场景目标

- **任务结构**：`linesync_small`、`linesync_big` 均由一个 ISP 段和一个 NPU 段串联组成。
- **延迟约束**：`linesync_big` 的执行时长恰为 `linesync_small` 的两倍。
- **发射节奏**：启用 `sync` 策略后，两任务在 ISP/NPU 上交替排队，几乎无空闲间隔，形成严丝合缝的流水。
- **优先级策略**：关闭优先级搜索 (`search_priority=false`)，直接使用 JSON 中提供的固定优先级。

## 关键代码改动

1. **同步策略实现**  
   在 `NNScheduler/core/enhanced_launcher.py` 中新增 `_create_sync_plan`，并在 `create_launch_plan` 中增加 `sync` 分支。该策略根据每个任务 ISP 段的执行时长分配周期间偏移，使得多个任务在一个周期内依次发射。发生异常时自动降级为 `balanced`。

2. **接口允许 `sync`**  
   `OptimizationInterface` 的所有策略校验处均新增 `"sync"`，避免配置被回退为 `balanced`。

3. **示例配置文件**  
   `test/sample_config/linesync_sync.json` 定义两个示例任务，带宽、时长和 FPS 设置如下：
   - `linesync_small`：ISP 1.0 ms、NPU 0.5 ms，`latency=1.5`。
   - `linesync_big`：ISP 2.0 ms、NPU 1.0 ms，`latency=3.0`。
   - 资源仅包含 `ISP_0`（50 GB/s）与 `NPU_0`（80 GB/s）。

## 运行与验证

- **命令**  
  ```bash
  python main_api.py test/sample_config/linesync_sync.json
  ```

- **生成工件**（时间戳为示例）  
  - Trace JSON：`artifacts_sim/optimized_schedule_chrome_trace_Linesync_Sync_Pipeline_20251016_010029.json`
  - 资源时间线：`artifacts_sim/optimized_schedule_timeline_Linesync_Sync_Pipeline_20251016_010029.png`
  - 结果摘要：`artifacts_sim/optimization_result_Linesync_Sync_Pipeline_20251016_010029.json`

- **Trace 核对**  
  - ISP (`tid=2`) 出现顺序：`small_ISP` → `big_ISP` 循环，段时分别为 1 ms 与 2 ms；周期为 3 ms。
  - NPU (`tid=1`) 中的段在 ISP 结束后立刻衔接：`small_NPU` 0.5 ms 后紧跟 `big_NPU` 1.0 ms。
  - 两个任务的首帧延迟分别约为 1.5 ms 与 4.0 ms，满足 “big 为 small 两倍” 的需求。

- **性能指标**  
  - 满足率：100%。
  - 平均延迟：≈2.1 ms。
  - ISP 利用率：100%，NPU 利用率 ≈43.3%，系统利用率 100%。

## 使用建议

- 调整 ISP/NPU 持续时间或 FPS，可轻松扩展到更多 `sync` 工作负载。
- 若需要进一步观察 Perfetto 可视化，可安装 Perfetto Python 依赖以导出 `.pftrace` 或 JSON 描述。
- 默认启用了段级模式 (`segment_mode=true`)，如需传统整段调度，请在配置中关闭该选项。 
