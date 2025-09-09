## 当前TK任务

涵盖了大部分NPU+DSP的混合场景：

1. NPU+DSP的多段混合（MOTR）
2. 中等NPU+DSP，NPU可分段（Yolo Small）
3. 长时NPU+DSP，NPU可分段（Yolo Big）
4. 短时NPU，无须分段（CNN TK Template & Search）
5. 高频率NPU，无须分段（ReID）
6. 中等NPU，无分段，有依赖（Pose 2D，依赖MOTR之后）
7. 短时NPU+DSP，有依赖（QIM，依赖MOTR之后）
8. 长时纯DSP任务，有依赖（Pose2D_to_3D，依赖Pose2D之后）

## 段级发射器

**混合发射模式**：

- T1 (MOTR) 仍然整体发射，保持ACPU_Runtime语义
- T2-T8 使用段级发射，充分利用资源

**智能段调度**：

- 不同资源类型的段可以并行执行
- 同资源类型的段保持顺序依赖

**资源利用最大化**：

- NPU执行时，DSP可以被其他任务使用
- 减少资源空闲时间

## 段级执行器

改进的执行器实现了真正的 **段级调度** ：

关键改进

**段作为独立调度单元** ：

* 每个段都有自己的 `SegmentInstance`
* 段可以独立进入就绪队列
* 不再强制所有段必须连续执行

**动态依赖管理** ：

* 只保持必要的依赖（前序段）
* 段完成后自动激活后续段
* 支持更灵活的执行顺序

**资源感知调度** ：

* 就绪段可以立即竞争可用资源
* 不同任务的段可以交织执行
* 最大化资源利用率

## 潜在的段级调度器

当前没有真正的段级调度器

**段感知执行器在这个场景下已经足够好**

- 简单有效
- 达到理论最优

**真正的段级调度器需要更智能的策略**

- 不仅考虑当前段，还要考虑后续段
- 需要全局优化视角

**实际应用建议**：

- 对于简单场景，段感知执行器可能就够了
- 对于复杂场景（更多任务、动态负载），真正的段级调度器会更有优势

## 当前四种发射策略选择

从代码中可以看到，系统实现了以下发射策略：

### 1. Eager（激进/急切）策略

* 所有任务都在最早可能的时间发射
* 一旦满足依赖关系就立即启动任务
* 优点：最大化并行度，充分利用资源
* 缺点：可能造成资源竞争，产生更多的空闲碎片

### 2. **Lazy（延迟）策略**

* 尽可能晚地发射任务
* 考虑任务执行时间，在deadline前才发射
* 优点：可能减少资源竞争，改善调度连续性
* 缺点：可能降低资源利用率

### 3. **Balanced（均衡）策略**

* 在eager和lazy之间取平衡
* 按优先级分组，组内交错发射
* 考虑资源负载均衡
* 优点：更好的负载分布

### 4. **Custom（自定义/优化）策略**

* 通过优化算法（遗传算法或爬山算法）生成
* 为每个任务设置特定的延迟因子
* 目标是最大化空闲时间同时满足FPS要求

## 现有Demo的发射算法分析

### 1. **demo_real_task_segmentation.py**

从代码结构看，这个demo主要使用：

* **Eager策略**作为基线
* 展示了段级调度的效果
* 重点在于演示任务分段后的调度行为

### 2. **test_scenarios_example.py**

这个文件展示了：

* 多种策略对比（eager、lazy、balanced）
* 使用了 **优化策略** （通过LaunchOptimizer）
* 重点在于不同场景下的策略效果对比

### 3. **demo_complete_system.py**

完整系统演示使用了：

* **Eager策略**作为基线
* **优化策略**通过遗传算法生成
* 展示了从基线到优化的完整流程

### 总结

系统的发射算法设计非常灵活：

1. 提供了三种基础策略（eager、lazy、balanced）
2. 通过优化器可以生成自定义策略
3. 虽然没有显式的JIT实现，但架构支持动态调度
4. 现有demo主要使用eager作为基线，然后通过优化器改进

这种设计让系统既有简单直接的基础策略，又能通过优化算法适应不同的场景需求。

## 不同的资源适配

### 📊 对比结果汇总

配置                       完成实例      System利用率      平均NPU利用率       平均DSP利用率       FPS满足率

带宽20Gbps                 33        100.5%         100.5%         39.8%          88.9%
带宽40Gbps                 42        86.5%          69.8%          41.3%          100.0%
带宽60Gbps                 42        86.5%          69.8%          41.3%          100.0%
带宽80Gbps                 42        86.4%          69.4%          41.2%          100.0%

带宽100Gbps                42        67.9%          42.4%          39.4%          100.0%

带宽120Gbps                42        67.9%          42.4%          39.5%          100.0%
带宽160Gbps                42        67.9%          42.4%          39.5%          100.0%

## 利用率计算方法

### 1. **NPU/DSP 利用率**

在 `ScheduleTracer` 类的 `get_resource_utilization()` 方法中计算：

```python
utilization[resource_id] = (busy_time / total_time) * 100
```

- **busy_time**: 资源实际执行任务的总时间
- **total_time**: 分析时间窗口（如200ms）
- 每个资源独立计算其利用率

### 2. **System 利用率**

在 `calculate_system_utilization()` 函数中计算：

```python
def calculate_system_utilization(tracer, window_size):
    """计算系统利用率（至少有一个硬件单元忙碌的时间比例）"""
    busy_intervals = []
  
    # 收集所有执行时间段
    for exec in tracer.executions:
        if exec.start_time is not None and exec.end_time is not None:
            busy_intervals.append((exec.start_time, exec.end_time))
  
    # 合并重叠的时间段
    busy_intervals.sort()
    merged_intervals = []
  
    for start, end in busy_intervals:
        if merged_intervals and start <= merged_intervals[-1][1]:
            merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
        else:
            merged_intervals.append((start, end))
  
    # 计算总忙碌时间
    total_busy_time = sum(end - start for start, end in merged_intervals)
  
    return (total_busy_time / window_size) * 100.0
```

**System利用率**表示系统中至少有一个资源在工作的时间比例。它通过合并所有资源的执行时间段，计算系统整体的忙碌时间。

### 3. 利用率计算说明：

1. **NPU/DSP利用率**：单个资源的忙碌时间占总时间的百分比
   - 公式：`(资源忙碌时间 / 时间窗口) × 100%`
   - 例如：NPU_0利用率 75.8% 表示该NPU在200ms中有151.6ms在执行任务
2. **System利用率**：系统整体的忙碌程度
   - 计算方法：合并所有资源的执行时间段，避免重复计算
   - 表示至少有一个资源在工作的时间比例
   - 例如：System利用率 75.5% 表示系统在200ms中有151ms至少有一个资源在工作
