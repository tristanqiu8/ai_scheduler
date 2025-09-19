 配置文件覆盖情况总览：

  config1_long_npu_segmented.json (1NPU+1DSP) - 4种任务类型：
  - ✅ 长时NPU可分段任务 (15ms+8.5ms)
  - ✅ 3段NPU+DSP混合任务 (NPU→DSP→NPU)
  - ✅ 纯DSP任务
  - ✅ DSP→NPU依赖任务链

  config2_5segment_hybrid.json (2NPU+1DSP) - 4种任务类型：
  - ✅ 5段NPU+DSP混合任务 (NPU→DSP→NPU→DSP→NPU)
  - ✅ 长时NPU可分段任务 (12.5ms+7.8ms)
  - ✅ NPU→DSP依赖任务链
  - ✅ 纯DSP任务

  config3_npu_to_dsp_dependency.json (2NPU+2DSP) - 4种任务类型：
  - ✅ NPU→DSP依赖任务链
  - ✅ 3段NPU+DSP混合任务 (NPU→DSP→NPU)
  - ✅ 纯DSP任务
  - ✅ 5段混合任务 (DSP→NPU→DSP→NPU→DSP)

  config4_dsp_to_npu_dependency.json (1NPU+1DSP) - 4种任务类型：
  - ✅ DSP→NPU依赖任务链
  - ✅ 长时NPU可分段任务 (16.2ms+9.8ms)
  - ✅ 3段NPU+DSP混合任务 (NPU→DSP→NPU)
  - ✅ 纯DSP任务

  config5_pure_dsp_and_3seg_hybrid.json (2NPU+1DSP) - 4种任务类型：
  - ✅ 纯DSP任务
  - ✅ 3段NPU+DSP混合任务 (NPU→DSP→NPU，NPU段有切点)
  - ✅ 5段混合任务 (NPU→DSP→NPU→DSP→NPU)
  - ✅ NPU→DSP依赖任务链

  任务类型全覆盖验证：

  - ✅ 长时NPU(>10ms)可分段: config1, config2, config4
  - ✅ 5段NPU+DSP混合: config2, config3, config5
  - ✅ NPU→DSP依赖: config2, config3, config5
  - ✅ DSP→NPU依赖: config1, config4
  - ✅ 纯DSP任务: config1, config2, config3, config4, config5
  - ✅ 3段NPU+DSP混合(NPU段有切点): config1, config3, config4, config5

  资源配置覆盖：

  - ✅ 1NPU+1DSP: config1, config4
  - ✅ 2NPU+1DSP: config2, config5
  - ✅ 2NPU+2DSP: config3

  所有配置都启用了search_priority=true，并且混合使用了normal和detailed日志级别。任务时长都在1-10ms范围内，分段后每段约2-3ms，每个配置包含5个任务