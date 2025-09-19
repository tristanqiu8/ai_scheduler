# AI Scheduler 开发和打包指南

## 项目结构

```
ai_scheduler/
├── ai_scheduler/                 # 主包目录
│   ├── __init__.py              # 包初始化，导出主要API
│   ├── cli.py                   # 命令行接口
│   ├── core/                    # 核心API模块
│   │   ├── __init__.py
│   │   └── optimization_api.py  # 优化API实现
│   ├── sample_config/           # 预设配置文件
│   │   ├── config_1npu_1dsp.json
│   │   ├── config_1npu_1dsp_alt.json
│   │   ├── config_2npu_1dsp.json
│   │   ├── config_2npu_1dsp_alt.json
│   │   └── config_2npu_2dsp.json
│   └── NNScheduler/             # 核心调度算法（从根目录复制）
│       ├── core/                # 核心算法
│       ├── interface/           # 接口层
│       ├── scenario/            # 场景模块
│       └── viz/                 # 可视化
├── NNScheduler/                 # 原始开发目录（不打包）
├── test/                        # 测试文件
├── setup.py                     # 打包配置
├── requirements.txt             # 依赖声明
├── README.md                    # 项目说明
└── LICENSE                      # 许可证
```

## WHL 文件打包步骤

### 1. 环境准备

确保安装了打包工具：
```bash
pip install build wheel twine
```

### 2. 清理旧的构建文件

```bash
# 删除旧的构建缓存
rm -rf build/
rm -rf dist/
rm -rf ai_scheduler.egg-info/

# Windows 用户使用
rmdir /s build
rmdir /s dist
rmdir /s ai_scheduler.egg-info
```

### 3. 构建 WHL 文件

#### 方法一：使用 setup.py（当前使用）
```bash
python setup.py bdist_wheel
```

#### 方法二：使用 build 工具（推荐）
```bash
python -m build --wheel
```

### 4. 验证构建结果

构建完成后，检查 `dist/` 目录：
```bash
ls dist/
# 应该看到: ai_scheduler-1.0.0rc0-py3-none-any.whl
```

验证包内容：
```bash
# 检查包结构
python -c "
import zipfile
z = zipfile.ZipFile('dist/ai_scheduler-1.0.0rc0-py3-none-any.whl')
files = z.namelist()
print('包含的文件数量:', len(files))
print('主要文件:')
for f in sorted(files)[:10]:
    print('  ', f)
"
```

### 5. 本地安装测试

```bash
# 强制重新安装
pip install --force-reinstall dist/ai_scheduler-1.0.0rc0-py3-none-any.whl

# 验证安装
python -c "import ai_scheduler; print('Version:', ai_scheduler.__version__)"
python -m ai_scheduler.cli --version
```

### 6. 功能测试

```bash
# CLI 测试
python -m ai_scheduler.cli --list-samples
python -m ai_scheduler.cli sample:config_1npu_1dsp.json --out test_output

# Python API 测试
python -c "
import ai_scheduler
configs = ai_scheduler.get_sample_configs()
print(f'找到 {len(configs)} 个配置')
result = ai_scheduler.optimize_from_json(configs[0], 'api_test')
print(f'满足率: {result[\"best_configuration\"][\"satisfaction_rate\"]:.1%}')
"
```

## 打包配置说明

### setup.py 关键配置

```python
setup(
    name="ai-scheduler",                              # 包名
    version=get_version(),                            # 从 __init__.py 读取版本
    packages=find_packages(include=['ai_scheduler*']), # 只打包 ai_scheduler 相关
    install_requires=get_requirements(),              # 从 requirements.txt 读取依赖

    # 命令行入口点
    entry_points={
        "console_scripts": [
            "ai-scheduler=ai_scheduler.cli:main",     # CLI 命令
        ],
    },

    # 包含数据文件
    include_package_data=True,
    package_data={
        "ai_scheduler": [
            "sample_config/*.json",                   # 预设配置
            "NNScheduler/interface/*.json"            # 接口配置
        ],
    },
)
```

### 依赖管理

**requirements.txt**:
```txt
numpy>=1.19.0
matplotlib>=3.3.0
plotly>=4.14.0
python-dateutil>=2.8.0
requests>=2.25.0
```

**自动依赖检测**:
```python
def get_requirements():
    req_file = Path(__file__).parent / "requirements.txt"
    if req_file.exists():
        with open(req_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return ["numpy>=1.19.0", ...]  # fallback
```

## 版本管理

### 版本号规则
- **开发版本**: `1.0.0preview`
- **发布候选**: `1.0.0rc0`（setuptools 自动转换）
- **正式版本**: `1.0.0`

### 版本更新流程
1. 修改 `ai_scheduler/__init__.py` 中的 `__version__`
2. 重新构建包：`python setup.py bdist_wheel`
3. 测试新版本功能
4. 发布或分发

## 发布流程

### 本地分发
```bash
# 直接分发 whl 文件
scp dist/ai_scheduler-1.0.0rc0-py3-none-any.whl user@server:/path/to/install/
```

### PyPI 发布（可选）
```bash
# 检查包
python -m twine check dist/*

# 上传到测试 PyPI
python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# 上传到正式 PyPI
python -m twine upload dist/*
```

## 常见问题和解决方案

### 1. 导入路径问题
**问题**: `ModuleNotFoundError: No module named 'NNScheduler.interface'`
**解决**: 确保使用相对导入或正确的包路径

### 2. 文件路径问题
**问题**: 相对路径在不同工作目录下失效
**解决**: 在 `optimization_api.py` 中使用 `config_path.resolve()` 转为绝对路径

### 3. 重复打包问题
**问题**: 根目录和包目录都有 NNScheduler
**解决**: 使用 `find_packages(include=['ai_scheduler*'])` 只打包指定包

### 4. 依赖缺失问题
**问题**: Linux 环境缺少 requests 库
**解决**: 在 requirements.txt 和 setup.py 中添加所有必要依赖

## 开发工作流

### 1. 日常开发
```bash
# 修改代码后重新安装测试
pip install -e .  # 开发模式安装
python -m ai_scheduler.cli --version
```

### 2. 功能测试
```bash
# 运行所有预设配置测试
for config in config_1npu_1dsp config_2npu_1dsp config_2npu_2dsp; do
    echo "Testing $config..."
    python -m ai_scheduler.cli sample:$config.json --out test_$config
done
```

### 3. 兼容性测试
```bash
# 运行兼容性测试脚本
python test_linux_compatibility.py
```

### 4. 发布准备
```bash
# 清理 → 构建 → 安装 → 测试
rm -rf build/ dist/ *.egg-info/
python setup.py bdist_wheel
pip install --force-reinstall dist/*.whl
python -m ai_scheduler.cli sample:config_1npu_1dsp.json
```

## 调试技巧

### 查看包内容
```bash
# 列出 whl 文件内容
python -c "
import zipfile
with zipfile.ZipFile('dist/ai_scheduler-1.0.0rc0-py3-none-any.whl') as z:
    for name in sorted(z.namelist()):
        print(name)
"
```

### 检查依赖
```bash
# 查看包依赖
pip show ai-scheduler
python -c "
import zipfile
z = zipfile.ZipFile('dist/ai_scheduler-1.0.0rc0-py3-none-any.whl')
metadata = z.read('ai_scheduler-1.0.0rc0.dist-info/METADATA').decode('utf-8')
for line in metadata.split('\n'):
    if 'Requires-Dist' in line:
        print(line)
"
```

### 测试导入
```bash
# 测试所有主要导入
python -c "
try:
    import ai_scheduler
    from ai_scheduler.core.optimization_api import OptimizationAPI
    from ai_scheduler.cli import main
    print('✅ 所有导入成功')
except Exception as e:
    print(f'❌ 导入失败: {e}')
"
```

## 最佳实践

1. **总是清理构建缓存**：避免旧文件影响新构建
2. **测试多个环境**：Windows 和 Linux 都要测试
3. **验证所有功能**：CLI 和 Python API 都要测试
4. **检查文件大小**：whl 文件不应过大（当前约 500KB）
5. **版本一致性**：确保 `__init__.py` 和实际版本匹配