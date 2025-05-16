import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.colors import to_rgb


class ShortestPathDialog:
    """最短路径对话框类"""

    def __init__(self, parent, graph_manager, visualizer):
        self.parent = parent
        self.graph_manager = graph_manager
        self.visualizer = visualizer

    def show_dialog(self):
        """显示最短路径对话框"""
        if not self.graph_manager.graph:
            messagebox.showerror("错误", "没有图形数据，请先加载图形", parent=self.parent)
            return

        dialog = tk.Toplevel(self.parent)
        dialog.title("计算最短路径")
        dialog.geometry("300x150")
        dialog.grab_set()

        # 创建输入框架
        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=20, padx=10, fill='both', expand=True)

        # 起点输入
        ttk.Label(input_frame, text="起点节点ID:").grid(row=0, column=0, sticky='w', pady=5)
        start_entry = ttk.Entry(input_frame)
        start_entry.grid(row=0, column=1, sticky='ew', pady=5)

        # 终点输入
        ttk.Label(input_frame, text="终点节点ID:").grid(row=1, column=0, sticky='w', pady=5)
        end_entry = ttk.Entry(input_frame)
        end_entry.grid(row=1, column=1, sticky='ew', pady=5)

        # 按钮框架
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)

        def calculate_path():
            start_node = start_entry.get().strip()
            end_node = end_entry.get().strip()

            if not start_node or not end_node:
                messagebox.showerror("错误", "请输入起点和终点节点ID", parent=dialog)
                return

            # 计算最短路径
            path, length, success = self.graph_manager.calculate_shortest_path(start_node, end_node)

            if success:
                # 高亮显示路径
                highlight_path(self.visualizer, path)
                # 显示路径信息
                path_info = f"路径: {' -> '.join(path)}\n总长度: {length:.2f}"
                messagebox.showinfo("最短路径", path_info, parent=dialog)
                # **更新状态栏**
                status_text = f"最短路径: {' → '.join(path)} | 距离: {length:.2f}"
                self.visualizer.status_bar.set_status(status_text)
                # 关闭对话框
                dialog.destroy()
            else:
                messagebox.showerror("错误", "无法找到最短路径", parent=dialog)

        ttk.Button(button_frame, text="计算", command=calculate_path).pack(side='left', padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side='left', padx=5)

        # 设置输入焦点
        start_entry.focus_set()


def highlight_path(visualizer, path, emergency_areas=None, highlighted_edges=None, highlighted_nodes=None,
                   highlight_color='red'):
    """高亮显示最短路径：
    - 清除之前的高亮状态
    - 路径上的边和节点颜色设为红色
    - 非路径上的边、节点和标签透明度降低
    - 紧急区域的节点标为黄色
    """
    if not path or len(path) < 2:
        return

    try:
        highlight_rgb = to_rgb(highlight_color)
    except ValueError:
        raise ValueError(
            f"Invalid highlight color: {highlight_color}. Please provide a valid color name or RGB string.")
    # **创建路径的集合**
    path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
    path_edges |= {(b, a) for a, b in path_edges}  # 添加反向边
    path_nodes = set(path)  # 路径上的节点

    # **获取当前所有节点颜色**
    node_colors = visualizer.nodes_scatter.get_facecolors()

    # # **首先重置所有节点样式（避免残留上一次的高亮效果）**
    if highlighted_nodes is None:
        highlighted_nodes = set()

    # **然后再高亮路径节点**
    for i, node in enumerate(visualizer.graph_manager.graph.nodes()):
        color = node_colors[i].copy()

        # **高亮路径节点为红色**
        if node in path_nodes:
            color[:3] = list(highlight_rgb)  # **路径节点变红 (RGB: 红色)**
            color[3] = 1.0  # 确保完全可见
            highlighted_nodes.add(node)
        # **高亮紧急区域节点为黄色**
        elif emergency_areas and node in emergency_areas:
            color[:3] = [1.0, 1.0, 0.0]  # **紧急区域节点变黄 (RGB: 黄色)**
            color[3] = 1.0  # 确保完全可见
            highlighted_nodes.add(node)
        # **非路径节点透明度降低**
        elif node not in highlighted_nodes:
            color[3] = 0.1  # **非路径节点透明度降低**

        node_colors[i] = color  # 更新颜色数组

        # **调整非路径节点的标签透明度**
        if node in visualizer.node_labels:
            label_alpha = 1.0 if node in path_nodes or (
                    emergency_areas and node in emergency_areas) else 0.1  # 路径和紧急区域的标签保持可见，其他变淡
            visualizer.node_labels[node].set_alpha(label_alpha)

    # **应用颜色更新**
    visualizer.nodes_scatter.set_facecolors(node_colors)

    # **重置已高亮的边**
    if highlighted_edges is None:
        highlighted_edges = set()

    # **然后高亮路径边**
    for edge, line in zip(visualizer.graph_manager.graph.edges(), visualizer.edges_lines):
        # 只高亮当前路径的边
        if edge in path_edges and edge not in highlighted_edges:
            line.set_color(highlight_color)  # **路径边变红**
            line.set_alpha(1.0)  # 确保完全可见
            highlighted_edges.add(edge)  # 标记为已高亮
        elif edge not in highlighted_edges:
            line.set_alpha(0.1)  # **非路径边透明度降低**

    # **刷新画布**
    visualizer.canvas.draw_idle()

class VisualizePathDialog:
    def __init__(self, parent, graph_manager, visualizer):
        self.parent = parent
        self.graph_manager = graph_manager
        self.visualizer = visualizer

    def show_dialog(self):
        """显示最短路径对话框"""
        if not self.graph_manager.graph:
            messagebox.showerror("错误", "没有图形数据，请先加载图形", parent=self.parent)
            return

        dialog = tk.Toplevel(self.parent)
        dialog.title("可视化路径")
        dialog.geometry("400x300")
        dialog.grab_set()

        # 创建输入框架，使其随窗口变化
        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=10, padx=10, fill='both', expand=True)

        # 路径输入
        ttk.Label(input_frame, text="路径:").grid(row=0, column=0, sticky='w', pady=5)
        paths_entry = ttk.Entry(input_frame)
        paths_entry.grid(row=0, column=1, sticky='ew', pady=5)

        # 紧急区域输入框
        ttk.Label(input_frame, text="紧急区域:").grid(row=1, column=0, sticky='w', pady=5)
        emergency_areas_entry = ttk.Entry(input_frame)
        emergency_areas_entry.grid(row=1, column=1, sticky='ew', pady=5)

        # 使文本框可伸缩
        input_frame.columnconfigure(1, weight=1)

        # 按钮框架
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10, padx=10, fill='x')

        # 提示信息框架
        message_frame = ttk.Frame(dialog)
        message_frame.pack(pady=10, padx=10, fill='both', expand=True)

        ttk.Label(message_frame,
                  text="请输入路径格式：[['start', 'node1', ..., 'end'], ['start', 'node1', ..., 'end']]").pack(
            anchor='w', pady=5)
        ttk.Label(message_frame, text="请输入紧急区域格式：['node1', 'node2', ...]").pack(anchor='w', pady=5)

        # 设置输入焦点
        paths_entry.focus_set()

        def visualizePaths():
            # 获取路径输入
            paths_input = paths_entry.get().strip()

            if not paths_input:
                messagebox.showerror("错误", "请输入至少一条路径", parent=dialog)
                return

            # 将输入的路径字符串转换为列表形式
            try:
                paths = eval(paths_input)  # 将字符串转化为列表格式
            except:
                messagebox.showerror("错误", "路径格式不正确", parent=dialog)
                return

            # 确保路径是一个包含路径列表的列表
            if not isinstance(paths, list) or not all(isinstance(path, list) for path in paths):
                messagebox.showerror("错误", "路径格式不正确", parent=dialog)
                return

            # 获取紧急区域
            emergency_areas_input = emergency_areas_entry.get().strip()
            try:
                emergency_areas = eval(emergency_areas_input) if emergency_areas_input else []
                if not isinstance(emergency_areas, list) or not all(isinstance(area, str) for area in emergency_areas):
                    raise ValueError
            except:
                messagebox.showerror("错误", "紧急区域格式不正确", parent=dialog)
                return
            # 预设颜色列表，循环使用
            color_list = ['red', 'green', 'blue', 'purple', 'orange', 'yellow']
            path_info = ""  # 用于累积所有路径信息

            # 记录已高亮节点和边
            highlighted_nodes = set()
            highlighted_edges = set()

            # 为每条路径指定颜色
            for idx, path in enumerate(paths):
                # 轮流使用颜色列表中的颜色
                highlight_color = color_list[idx % len(color_list)]

                # 高亮显示路径
                highlight_path(self.visualizer, path,
                               emergency_areas=emergency_areas,
                               highlighted_nodes=highlighted_nodes,
                               highlighted_edges=highlighted_edges,
                               highlight_color=highlight_color)
                # 累积路径信息
                path_info += f"路径: {' -> '.join(path)}\n"
                # 更新状态栏
                status_text = f"路径: {' → '.join(path)}"
                self.visualizer.status_bar.set_status(status_text)

            # 显示所有路径信息
            messagebox.showinfo("路径", path_info, parent=dialog)
            # 关闭对话框
            dialog.destroy()

        ttk.Button(button_frame, text="高亮", command=visualizePaths).pack(side='left', padx=5)
        ttk.Button(button_frame, text="取消", command=dialog.destroy).pack(side='left', padx=5)

        # 设置输入焦点
        paths_entry.focus_set()

