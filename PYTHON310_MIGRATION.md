# NRI-MD Python 3.10+ 迁移指南

## 概述

本文档描述了将NRI-MD项目从Python 3.7/PyTorch 1.2迁移到Python 3.10+/PyTorch 2.0+的完整过程。

## 主要更新

### 1. Python版本要求
- **原版本**: Python 3.7
- **新版本**: Python 3.10+
- **兼容性**: 支持Python 3.10, 3.11, 3.12

### 2. PyTorch版本升级
- **原版本**: PyTorch 1.2
- **新版本**: PyTorch 2.0+
- **CUDA支持**: 可选CUDA 11.8支持

### 3. 项目结构现代化

#### 新增文件
```
NRI-MD/
├── pyproject.toml          # 现代Python项目配置
├── requirements.txt        # CPU版本依赖
├── requirements-cuda118.txt # CUDA 11.8版本依赖
├── setup.py               # 向后兼容的安装脚本
├── Makefile              # 便捷的开发命令
├── environment.yml       # Conda环境配置
├── .python-version       # Python版本指定
├── nri_md/               # 重构的包结构
│   ├── __init__.py
│   ├── main.py           # 更新的训练脚本
│   ├── modules.py        # 现代化的神经网络模块
│   ├── utils.py          # 更新的工具函数
│   ├── convert_dataset.py # 数据转换工具
│   ├── postanalysis_visual.py # 可视化工具
│   └── postanalysis_path.py   # 路径分析工具
├── tests/                # 测试套件
│   ├── __init__.py
│   ├── test_utils.py
│   └── test_modules.py
├── demo.py               # 演示脚本
└── install_and_test.py   # 安装测试脚本
```

### 4. 代码现代化

#### 类型注解
```python
# 原版本
def my_softmax(input, axis=1):
    # ...

# 新版本
def my_softmax(input_tensor: torch.Tensor, axis: int = 1) -> torch.Tensor:
    # ...
```

#### 现代PyTorch API
```python
# 原版本 - 使用已弃用的Variable
from torch.autograd import Variable
y = Variable(y_hard.data - y_soft.data) + y_soft

# 新版本 - 使用现代API
y = (y_hard - y_soft).detach() + y_soft
```

#### 错误处理改进
```python
# 新版本包含更好的错误处理和验证
try:
    coordinates, num_atoms = read_pdb_trajectory(pdb_file)
    # ...
except Exception as e:
    print(f"Error processing {pdb_file}: {e}")
    continue
```

### 5. 命令行工具

新版本提供了便捷的命令行工具：

```bash
# 数据转换
nri-md-convert --data-dir data/pdb --output-dir data

# 模型训练
nri-md-train --num-residues 77 --epochs 500

# 结果可视化
nri-md-visualize --filename logs/out_probs_train.npy

# 路径分析
nri-md-path --filename logs/out_probs_train.npy --source-node 46
```

## 安装指南

### 方法1: 使用pip安装
```bash
# 克隆仓库
git clone https://github.com/juexinwang/NRI-MD.git
cd NRI-MD

# CPU版本
pip install -r requirements.txt
pip install -e .

# CUDA 11.8版本
pip install -r requirements-cuda118.txt
pip install -e .
```

### 方法2: 使用Conda
```bash
# 创建环境
conda env create -f environment.yml
conda activate nri-md

# 安装包
pip install -e .
```

### 方法3: 使用Makefile
```bash
# CPU版本
make install

# CUDA版本
make install-cuda

# 开发版本
make install-dev
```

## 依赖更新

### 核心依赖
| 包名 | 原版本 | 新版本 |
|------|--------|--------|
| torch | 1.2 | ≥2.0.0 |
| numpy | - | ≥1.21.0,<2.0.0 |
| scipy | - | ≥1.7.0 |
| matplotlib | - | ≥3.5.0 |
| seaborn | - | ≥0.11.0 |
| pandas | - | ≥1.3.0 |
| networkx | 2.5 | ≥2.6.0 |

### 开发依赖
- pytest ≥7.0.0
- pytest-cov ≥4.0.0
- black ≥22.0.0
- flake8 ≥4.0.0
- mypy ≥0.991

## 兼容性说明

### 向后兼容
- 保持了原有的API接口
- 支持原有的数据格式
- 命令行参数保持一致

### 新特性
- 类型提示支持
- 更好的错误处理
- 现代化的包管理
- 自动化测试
- 代码质量检查

## 测试验证

运行完整测试套件：
```bash
# 基本功能测试
python demo.py

# 单元测试
pytest tests/ -v

# 代码质量检查
make all-checks
```

## 性能对比

### PyTorch 2.0+ 优势
- 更快的训练速度（torch.compile支持）
- 更好的内存效率
- 改进的CUDA支持
- 更稳定的梯度计算

### 兼容性测试结果
- ✅ Python 3.10.x
- ✅ Python 3.11.x  
- ✅ Python 3.12.x
- ✅ PyTorch 2.0+
- ✅ CUDA 11.8+
- ✅ CPU-only模式

## 迁移检查清单

- [x] 更新Python版本要求到3.10+
- [x] 升级PyTorch到2.0+
- [x] 添加类型注解
- [x] 更新已弃用的API调用
- [x] 创建现代项目结构
- [x] 添加测试套件
- [x] 创建安装脚本
- [x] 更新文档
- [x] 验证功能完整性

## 故障排除

### 常见问题

1. **ImportError: No module named 'nri_md'**
   ```bash
   pip install -e .
   ```

2. **CUDA不可用**
   ```bash
   pip install -r requirements-cuda118.txt
   ```

3. **依赖冲突**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt --force-reinstall
   ```

## 贡献指南

### 开发环境设置
```bash
# 克隆仓库
git clone https://github.com/juexinwang/NRI-MD.git
cd NRI-MD

# 安装开发依赖
make install-dev

# 运行测试
make test

# 代码格式化
make format

# 代码检查
make lint
```

### 代码标准
- 使用类型注解
- 遵循PEP 8规范
- 编写单元测试
- 更新文档

## 总结

NRI-MD项目已成功迁移到Python 3.10+环境，具备以下优势：

1. **现代化**: 使用最新的Python和PyTorch特性
2. **可维护性**: 改进的代码结构和类型安全
3. **易用性**: 提供命令行工具和安装脚本
4. **可扩展性**: 模块化设计便于功能扩展
5. **稳定性**: 完整的测试覆盖和错误处理

项目现在完全兼容Python 3.10+环境，同时保持了与原版本的API兼容性。