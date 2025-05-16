# GRAPH_VISUALIZATION 项目文档

## 1. 项目概述

GRAPH_VISUALIZATION 是一个基于Python的图可视化工具，用于展示和分析复杂的网络结构。该项目支持多种布局算法、交互式操作、节点分层显示等功能。

## 2. 项目结构

## 3. 核心功能

### 3.1 图数据管理 (GraphManager)

#### 基本功能
- 图数据的加载和保存
- 节点和边的属性管理
- 布局计算和更新
- 动态布局支持

#### 主要方法

##### load_graph
```python
def load_graph(file_path, progress_window=None):
    """
    加载GraphML文件
    1. 读取GraphML文件
    2. 初始化节点属性
    3. 处理自定义属性
    4. 更新进度条
    返回: 是否加载成功
    """
```

##### calculate_layout
```python
def calculate_layout(layout_name, node_size, progress_window=None):
    """
    计算图布局
    1. 根据布局类型选择算法
    2. 计算节点位置
    3. 优化布局
    4. 更新显示
    支持的布局:
    - fruchterman_reingold
    - spring
    - floor_based (按楼层分层)
    """
```

##### update_node_position
```python
def update_node_position(node, new_pos):
    """
    更新节点位置
    1. 计算移动向量
    2. 更新主节点位置
    3. 递归更新相邻节点
    4. 应用衰减系数
    """
```

### 3.2 布局算法

#### Fruchterman-Reingold布局
```python
def fruchterman_reingold_layout():
    """
    力导向布局算法
    1. 计算排斥力
    2. 计算吸引力
    3. 应用重力
    4. 更新节点位置
    """
```

#### 楼层分层布局
```python
def floor_based_layout():
    """
    按楼层分层的布局算法
    1. 按楼层分组节点
    2. 对每层应用Fruchterman-Reingold布局
    3. 设置层间距
    4. 优化跨层边
    """
```

### 3.3 交互功能

#### 节点操作
- 拖拽节点
- 点击选择
- 显示/隐藏
- 属性编辑

#### 视图控制
- 缩放
- 平移
- 重置视图
- 自适应大小

#### 布局控制
- 布局算法选择
- 重新布局
- 动态布局开关
- 布局参数调整