#!/usr/bin/env python3
"""
调度系统配置类 - 集中管理所有配置参数
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
from core.enums import ResourceType


class ScenarioType(Enum):
    """预定义的测试场景"""
    BASELINE = "baseline"  # 基准场景：1 NPU + 1 DSP
    HIGH_BANDWIDTH = "high_bandwidth"  # 高带宽场景
    MULTI_NPU = "multi_npu"  # 多NPU场景
    MULTI_DSP = "multi_dsp"  # 多DSP场景
    BALANCED = "balanced"  # 平衡配置
    CUSTOM = "custom"  # 自定义配置


@dataclass
class ResourceConfig:
    """单个资源的配置"""
    resource_id: str
    resource_type: ResourceType
    bandwidth: float
    
    def __str__(self):
        return f"{self.resource_id}({self.resource_type.value}, {self.bandwidth}Gbps)"


@dataclass
class SchedulingConfig:
    """调度系统配置"""
    # 资源配置
    resources: List[ResourceConfig] = field(default_factory=list)
    
    # 时间窗口配置
    simulation_duration: float = 200.0  # 仿真时长(ms)
    analysis_window: float = 200.0  # 分析窗口(ms)
    
    # 调度策略
    segment_mode: bool = True  # 是否使用段级调度
    launch_strategy: str = "eager"  # 发射策略
    
    # 场景类型
    scenario: ScenarioType = ScenarioType.BASELINE
    scenario_name: str = ""
    
    @classmethod
    def create_baseline(cls) -> 'SchedulingConfig':
        """创建基准配置：1 NPU(40) + 1 DSP(40)"""
        config = cls(scenario=ScenarioType.BASELINE, scenario_name="基准配置")
        config.add_resource("NPU_0", ResourceType.NPU, 40.0)
        config.add_resource("DSP_0", ResourceType.DSP, 40.0)
        return config
    
    @classmethod
    def create_high_bandwidth(cls, npu_bw: float = 120.0, dsp_bw: float = 120.0) -> 'SchedulingConfig':
        """创建高带宽配置"""
        config = cls(scenario=ScenarioType.HIGH_BANDWIDTH, scenario_name="高带宽配置")
        config.add_resource("NPU_0", ResourceType.NPU, npu_bw)
        config.add_resource("DSP_0", ResourceType.DSP, dsp_bw)
        return config
    
    @classmethod
    def create_multi_npu(cls, npu_count: int = 2, npu_bw: float = 60.0, dsp_bw: float = 40.0) -> 'SchedulingConfig':
        """创建多NPU配置"""
        config = cls(scenario=ScenarioType.MULTI_NPU, scenario_name=f"{npu_count}×NPU配置")
        for i in range(npu_count):
            config.add_resource(f"NPU_{i}", ResourceType.NPU, npu_bw)
        config.add_resource("DSP_0", ResourceType.DSP, dsp_bw)
        return config
    
    @classmethod
    def create_multi_dsp(cls, dsp_count: int = 2, npu_bw: float = 40.0, dsp_bw: float = 60.0) -> 'SchedulingConfig':
        """创建多DSP配置"""
        config = cls(scenario=ScenarioType.MULTI_DSP, scenario_name=f"{dsp_count}×DSP配置")
        config.add_resource("NPU_0", ResourceType.NPU, npu_bw)
        for i in range(dsp_count):
            config.add_resource(f"DSP_{i}", ResourceType.DSP, dsp_bw)
        return config
    
    @classmethod
    def create_balanced(cls, npu_count: int = 2, dsp_count: int = 2, 
                       npu_bw: float = 60.0, dsp_bw: float = 60.0) -> 'SchedulingConfig':
        """创建平衡配置：多个NPU和DSP"""
        config = cls(scenario=ScenarioType.BALANCED, scenario_name=f"{npu_count}×NPU + {dsp_count}×DSP")
        for i in range(npu_count):
            config.add_resource(f"NPU_{i}", ResourceType.NPU, npu_bw)
        for i in range(dsp_count):
            config.add_resource(f"DSP_{i}", ResourceType.DSP, dsp_bw)
        return config
    
    def add_resource(self, resource_id: str, resource_type: ResourceType, bandwidth: float):
        """添加资源"""
        self.resources.append(ResourceConfig(resource_id, resource_type, bandwidth))
    
    def get_npu_bandwidth(self) -> float:
        """获取NPU的平均带宽（用于理论计算）"""
        npu_resources = [r for r in self.resources if r.resource_type == ResourceType.NPU]
        if not npu_resources:
            return 0.0
        return sum(r.bandwidth for r in npu_resources) / len(npu_resources)
    
    def get_dsp_bandwidth(self) -> float:
        """获取DSP的平均带宽（用于理论计算）"""
        dsp_resources = [r for r in self.resources if r.resource_type == ResourceType.DSP]
        if not dsp_resources:
            return 0.0
        return sum(r.bandwidth for r in dsp_resources) / len(dsp_resources)
    
    def get_resource_summary(self) -> str:
        """获取资源配置摘要"""
        npu_count = sum(1 for r in self.resources if r.resource_type == ResourceType.NPU)
        dsp_count = sum(1 for r in self.resources if r.resource_type == ResourceType.DSP)
        
        npu_bw = [r.bandwidth for r in self.resources if r.resource_type == ResourceType.NPU]
        dsp_bw = [r.bandwidth for r in self.resources if r.resource_type == ResourceType.DSP]
        
        summary = f"{self.scenario_name}: "
        if npu_count > 0:
            summary += f"{npu_count}×NPU({','.join(map(str, npu_bw))}Gbps)"
        if dsp_count > 0:
            if npu_count > 0:
                summary += " + "
            summary += f"{dsp_count}×DSP({','.join(map(str, dsp_bw))}Gbps)"
        
        return summary
    
    def print_config(self):
        """打印配置详情"""
        print(f"\n📋 {self.scenario_name}")
        print("="*60)
        print("资源配置:")
        for resource in self.resources:
            print(f"  - {resource}")
        print(f"\n时间配置:")
        print(f"  - 仿真时长: {self.simulation_duration}ms")
        print(f"  - 分析窗口: {self.analysis_window}ms")
        print(f"\n调度配置:")
        print(f"  - 段级模式: {'启用' if self.segment_mode else '禁用'}")
        print(f"  - 发射策略: {self.launch_strategy}")
