# 安装指南

本指南将帮助您安装和配置 Graph Visualization 项目。

## 系统要求

- Python 3.8 或更高版本
- Windows 10/11 或 Linux/macOS
- 至少 4GB 可用内存
- 至少 1GB 可用磁盘空间

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/graph-visualization.git
cd graph-visualization
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 验证安装

```bash
python main.py
```

## 常见问题

### 依赖安装失败

如果安装依赖时遇到问题，请确保：

1. Python 版本符合要求
2. pip 已更新到最新版本
3. 网络连接正常

### 运行程序报错

如果运行程序时遇到错误，请检查：

1. 虚拟环境是否正确激活
2. 所有依赖是否安装成功
3. 配置文件是否正确

## 配置说明

项目的主要配置文件位于 `config/` 目录下：

- `config.yaml`: 主配置文件
- `logging.yaml`: 日志配置

## 更新说明

要更新到最新版本：

```bash
git pull
pip install -r requirements.txt --upgrade
``` 