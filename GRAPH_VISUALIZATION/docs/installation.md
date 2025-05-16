# 安装指南

## 系统要求

- Python 3.8 或更高版本
- Windows 10/11 或 Linux/macOS
- 至少 4GB 可用内存
- 至少 1GB 可用磁盘空间

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/GRAPH_VISUALIZATION.git
cd GRAPH_VISUALIZATION
```

### 2. 创建虚拟环境（推荐）

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

运行以下命令验证安装是否成功：

```bash
python main.py
```

## 依赖说明

主要依赖包包括：

- PyQt5：用于图形用户界面
- NetworkX：用于图数据处理
- Matplotlib：用于基础绘图功能
- NumPy：用于数值计算
- Pandas：用于数据处理

完整的依赖列表可以在 `requirements.txt` 文件中查看。

## 常见安装问题

### 1. 依赖安装失败

如果安装依赖时遇到问题，可以尝试：

```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 2. PyQt5 安装问题

在 Windows 系统上，如果 PyQt5 安装失败，可以尝试：

```bash
pip install PyQt5-tools
```

### 3. 内存不足

如果安装过程中遇到内存不足的问题，可以尝试：

1. 关闭其他占用内存的程序
2. 使用 `--no-cache-dir` 选项安装
3. 增加系统虚拟内存

## 开发环境设置

如果你想要参与项目开发，建议安装以下额外工具：

```bash
pip install pytest  # 用于测试
pip install black   # 用于代码格式化
pip install pylint  # 用于代码检查
```

## 更新

要更新到最新版本：

```bash
git pull origin main
pip install -r requirements.txt --upgrade
``` 