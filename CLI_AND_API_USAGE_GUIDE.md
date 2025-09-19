# AI Scheduler CLI 和 Python API 使用指南

## ✅ CLI 接口功能

### 1. 列出预设配置
```bash
python -m ai_scheduler.cli --list-samples
```

显示所有5个预设配置：
- `config_1npu_1dsp.json` - 1NPU+1DSP配置，包含4种混合任务类型
- `config_1npu_1dsp_alt.json` - 1NPU+1DSP替代配置
- `config_2npu_1dsp.json` - 2NPU+1DSP配置
- `config_2npu_1dsp_alt.json` - 2NPU+1DSP替代配置
- `config_2npu_2dsp.json` - 2NPU+2DSP配置

### 2. 直接运行预设配置
```bash
# 使用 sample: 前缀直接运行预设配置
python -m ai_scheduler.cli sample:config_1npu_1dsp.json
python -m ai_scheduler.cli sample:config_2npu_1dsp.json --out results/

# 支持所有5个预设配置
python -m ai_scheduler.cli sample:config_1npu_1dsp_alt.json
python -m ai_scheduler.cli sample:config_2npu_1dsp_alt.json
python -m ai_scheduler.cli sample:config_2npu_2dsp.json
```

### 3. 其他CLI功能
```bash
# 版本信息
python -m ai_scheduler.cli --version

# 验证配置文件
python -m ai_scheduler.cli --validate sample:config_1npu_1dsp.json

# 详细输出
python -m ai_scheduler.cli sample:config_1npu_1dsp.json --verbose

# 自定义输出目录
python -m ai_scheduler.cli sample:config_1npu_1dsp.json --out my_results/
```

## ✅ Python API 功能

### 1. 基本导入和使用
```python
import ai_scheduler

# 最简单的使用方式
result = ai_scheduler.optimize_from_json('config.json', 'output/')
print(f"满足率: {result['best_configuration']['satisfaction_rate']:.1%}")
```

### 2. 使用预设配置
```python
# 获取特定预设配置路径
sample_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')
if sample_path:
    result = ai_scheduler.optimize_from_json(sample_path, './demo_output')
    print(f"满足率: {result['best_configuration']['satisfaction_rate']:.1%}")
    print(f"平均延迟: {result['best_configuration']['avg_latency']:.1f}ms")
    print(f"系统利用率: {result['best_configuration']['system_utilization']:.1f}%")
```

### 3. 列出所有预设配置
```python
# 获取所有预设配置路径
configs = ai_scheduler.get_sample_configs()
print(f"找到 {len(configs)} 个预设配置:")
for config in configs:
    print(f"  {config}")

# 遍历运行所有配置
for i, config_path in enumerate(configs):
    print(f"运行配置 {i+1}: {config_path}")
    result = ai_scheduler.optimize_from_json(config_path, f'output_{i+1}/')
    satisfaction = result['best_configuration']['satisfaction_rate']
    print(f"  满足率: {satisfaction:.1%}")
```

### 4. 使用API类
```python
# 创建优化器实例
api = ai_scheduler.OptimizationAPI()

# 运行优化
result = api.optimize_from_json('config.json', 'output/')

# 验证配置文件
validation = api.validate_config('config.json')
if validation['valid']:
    print("配置文件有效")
else:
    print("配置错误:", validation['errors'])

# 列出预设配置
samples = api.list_sample_configs()
for sample in samples:
    print(f"配置: {sample['name']}")
    print(f"场景: {sample['scenario_name']}")
    print(f"描述: {sample['description']}")
```

### 5. 版本和信息获取
```python
# 获取版本信息
version_info = ai_scheduler.version_info()
print(f"版本: {version_info['version']}")
print(f"维护者: {version_info['maintainer']}")
print(f"团队: {version_info['team']}")
print(f"描述: {version_info['description']}")

# 检查包安装
try:
    import ai_scheduler
    print(f"AI Scheduler {ai_scheduler.__version__} 已安装")
except ImportError:
    print("AI Scheduler 未安装")
```

## 💡 参考 example_test.py

完整的使用示例可以参考项目根目录下的 `example_test.py` 文件第65-115行，其中包含：

1. **便捷函数使用** (第68-71行)
2. **预设配置运行** (第74-93行)
3. **配置列表获取** (第95-102行)
4. **版本信息获取** (第104-111行)

## 🎯 实际测试验证

### CLI 测试
```bash
# ✅ 列出预设配置
python -m ai_scheduler.cli --list-samples

# ✅ 运行预设配置
python -m ai_scheduler.cli sample:config_1npu_1dsp.json --out test_sample
# 结果: 100.0% 满足率，7.5ms平均延迟，81.6%系统利用率
```

### Python API 测试
```python
# ✅ API 功能验证
import ai_scheduler
configs = ai_scheduler.get_sample_configs()  # 5个配置
sample_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')
result = ai_scheduler.optimize_from_json(sample_path, 'python_api_test')
# 结果: 100.0% 满足率，7.5ms平均延迟
```

## 📝 总结

- ✅ **CLI支持**: 可以直接运行所有5个预设配置，使用 `sample:` 前缀
- ✅ **Python API支持**: 完整的programmatic接口，支持所有example_test.py中的功能
- ✅ **便捷功能**: 列出配置、验证配置、获取版本信息等
- ✅ **输出控制**: 支持自定义输出目录和详细日志
- ✅ **跨平台**: Windows和Linux环境均支持