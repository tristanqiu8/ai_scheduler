## 不同的资源适配


### 📊 对比结果汇总

#### 配置                       完成实例      System利用率      平均NPU利用率       平均DSP利用率       FPS满足率

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
