# NRI-MD 版本信息

## 当前版本: 1.0.0 (Python 3.10+ 兼容版)

### 发布日期
2025年1月

### 主要特性
- ✅ Python 3.10+ 支持
- ✅ PyTorch 2.0+ 兼容
- ✅ 现代化项目结构
- ✅ 类型注解支持
- ✅ 命令行工具
- ✅ 自动化测试
- ✅ CUDA 11.8+ 支持（可选）

### 系统要求
- Python: ≥3.10
- PyTorch: ≥2.0.0
- NumPy: ≥1.21.0, <2.0.0
- 其他依赖见 requirements.txt

### 安装验证
```bash
python -c "import nri_md; print('NRI-MD安装成功!')"
```

### 快速开始
```bash
# 1. 安装
pip install -e .

# 2. 运行演示
python demo.py

# 3. 查看帮助
nri-md-train --help
```

### 更新历史
- v1.0.0: Python 3.10+兼容版本，完全重构
- v0.x.x: 原始版本（Python 3.7, PyTorch 1.2）

### 支持
- 文档: README.md, PYTHON310_MIGRATION.md
- 问题报告: GitHub Issues
- 演示: demo.py