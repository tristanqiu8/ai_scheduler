您说得非常对！这是一个重要的硬件约束。让我修改设计来支持共享带宽池：现在让我更新 `resource_queue.py` 来集成带宽管理器：现在我创建一个使用示例来展示如何使用这个系统：## 总结

我已经创建了一个灵活的带宽管理系统，主要特点：

### 1. **带宽池概念** (`BandwidthPool`)
- NPU共享一个总带宽池（如120）
- DSP共享另一个总带宽池（如80）
- 支持设置最小带宽保证

### 2. **动态带宽分配** (`BandwidthManager`)
- 自动计算每个活跃硬件单元的带宽
- 跟踪带宽随时间的变化
- 支持带宽预估和实际分配

### 3. **向后兼容的设计**
- `ResourceQueue` 可以在固定带宽模式下工作（无需 BandwidthManager）
- 也可以在动态带宽模式下工作（传入 BandwidthManager）
- 通过 `bandwidth_manager` 参数控制模式

### 4. **使用方式**

```python
# 固定带宽模式（向后兼容）
static_manager = ResourceQueueManager()
static_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)

# 动态带宽模式
bw_manager = BandwidthManager()
bw_manager.add_bandwidth_pool(ResourceType.NPU, total_bandwidth=120.0)
dynamic_manager = ResourceQueueManager(bandwidth_manager=bw_manager)
dynamic_manager.add_resource("NPU_0", ResourceType.NPU)  # 动态分配
```

### 5. **实际影响**
- 当只有NPU_0运行时，它获得全部120带宽
- 当NPU_0和NPU_1同时运行时，各获得60带宽
- 任务执行时间会根据实际分配的带宽动态调整

这种设计既保留了简单场景的易用性，又为复杂的带宽共享场景留出了空间。您觉得这样的设计如何？需要调整什么地方吗？