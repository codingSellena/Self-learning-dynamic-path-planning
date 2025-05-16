# 使用说明

## 基本使用

### 启动程序

```bash
python main.py
```

### 主界面说明

程序启动后，你会看到以下主要界面元素：

1. 菜单栏
   - 文件：用于打开/保存文件
   - 编辑：用于编辑操作
   - 视图：用于调整显示选项
   - 帮助：查看帮助信息

2. 工具栏
   - 常用操作按钮
   - 视图控制按钮
   - 布局算法选择

3. 主视图区
   - 图的可视化显示区域
   - 支持鼠标交互

4. 属性面板
   - 节点属性编辑
   - 边属性编辑
   - 图属性设置

## 基本操作

### 1. 加载数据

支持多种数据格式：

- CSV 文件
- JSON 文件
- TXT 文件
- 数据库导入

示例：
```python
# 从 CSV 文件加载
graph = load_from_csv("data.csv")

# 从 JSON 文件加载
graph = load_from_json("data.json")
```

### 2. 视图操作

- 缩放：使用鼠标滚轮或工具栏按钮
- 平移：按住鼠标左键拖动
- 旋转：按住鼠标右键拖动
- 重置视图：点击工具栏的"重置"按钮

### 3. 节点操作

- 选择节点：点击节点
- 多选节点：按住 Ctrl 键点击
- 移动节点：拖动节点
- 编辑节点属性：双击节点

### 4. 边操作

- 选择边：点击边
- 编辑边属性：双击边
- 添加边：按住 Shift 键连接两个节点
- 删除边：选中边后按 Delete 键

## 高级功能

### 1. 布局算法

支持多种布局算法：

- 力导向布局
- 圆形布局
- 网格布局
- 层次布局
- 随机布局

### 2. 样式设置

可以自定义：

- 节点颜色
- 节点大小
- 边颜色
- 边宽度
- 标签显示

### 3. 数据分析

提供多种分析工具：

- 度分布分析
- 中心性分析
- 社区检测
- 路径分析

### 4. 导出功能

支持导出为：

- PNG 图片
- SVG 矢量图
- PDF 文档
- JSON 数据

## 示例

### 1. 创建简单图

```python
from graph_visualization import Graph

# 创建新图
graph = Graph()

# 添加节点
graph.add_node(1, label="Node 1")
graph.add_node(2, label="Node 2")

# 添加边
graph.add_edge(1, 2, weight=1.0)

# 设置布局
graph.set_layout("force_directed")

# 显示图
graph.show()
```

### 2. 加载并分析数据

```python
# 加载数据
graph = load_from_csv("data.csv")

# 计算中心性
centrality = graph.calculate_centrality()

# 设置节点大小
graph.set_node_size(centrality)

# 应用布局
graph.set_layout("circular")

# 导出结果
graph.export("result.png")
```

## 快捷键

- Ctrl+O：打开文件
- Ctrl+S：保存文件
- Ctrl+Z：撤销
- Ctrl+Y：重做
- Delete：删除选中元素
- Ctrl+A：全选
- Ctrl+C：复制
- Ctrl+V：粘贴
- Ctrl+X：剪切
- F5：刷新视图
- Esc：取消选择 