import tkinter as tk
from tkinter import messagebox

from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()


class SearchWidget:
    def __init__(self, toolbar, visualizer,node_panel):
        self.visualizer = visualizer
        self.toolbar = toolbar
        self.node_panel = node_panel

        # 保存当前高亮的节点和边
        self.highlighted_node = None
        self.highlighted_edges = []
        self.original_colors = {}

    def _handle_search(self):
        """处理搜索请求"""
        # 清除之前的高亮
        global TypeError, ValueError
        self._clear_highlights()

        # 获取搜索文本
        search_text = self.toolbar.search_entry.get().strip()
        if not search_text:
            return

        # 在图中查找节点
        graph = self.visualizer.graph_manager.graph
        if search_text not in graph.nodes:
            messagebox.showwarning("未找到", f"节点 '{search_text}' 不存在！")
            return

        # 获取当前的楼层过滤状态
        unselected_floors = self.visualizer.node_panel.get_unselected_floors()
        logger.info(f"当前楼层过滤状态: {unselected_floors}")

        # 获取节点楼层信息
        node_lc = graph.nodes[search_text].get("lc")
        try:
            node_floor = int(float(node_lc))  # 转换为整数
            if node_floor in unselected_floors:
                logger.info(f"节点 {search_text} 在未选中楼层 {node_floor}")
                messagebox.showinfo("提示", f"隐藏节点 '{search_text}' 在楼层 {node_floor} 中")
                return
        except (ValueError, TypeError):
            pass  # lc 解析失败，直接跳过

        # 高亮节点和相关边 - 传递节点ID而不是节点字典
        self.visualizer.select_node(search_text)
        self.visualizer.update_styles()

    def _highlight_node(self, node_id):
        """高亮节点和相关边"""
        graph = self.visualizer.graph_manager.graph

        # 保存当前高亮的节点
        self.highlighted_node = node_id

        # 获取节点索引
        nodes = list(graph.nodes())
        node_idx = nodes.index(node_id)

        # 保存原始颜色
        self.original_colors['node'] = self.visualizer.nodes_scatter.get_facecolor()[node_idx]

        # 设置节点颜色为红色
        colors = self.visualizer.nodes_scatter.get_facecolor()
        colors[node_idx] = [1, 0, 0, 1]  # 红色
        self.visualizer.nodes_scatter.set_facecolor(colors)

        # 高亮相关边
        for i, edge in enumerate(graph.edges()):
            if node_id in edge:
                self.highlighted_edges.append(i)
                # 保存原始颜色
                self.original_colors[f'edge_{i}'] = self.visualizer.edges_lines[i].get_color()
                # 设置边为红色
                self.visualizer.edges_lines[i].set_color('red')

        # 更新显示
        self.visualizer.canvas.draw_idle()

    def _clear_highlights(self):
        """清除所有高亮"""
        if self.highlighted_node is None:
            return

        # 恢复节点颜色
        graph = self.visualizer.graph_manager.graph
        nodes = list(graph.nodes())
        node_idx = nodes.index(self.highlighted_node)
        colors = self.visualizer.nodes_scatter.get_facecolor()
        colors[node_idx] = self.original_colors['node']
        self.visualizer.nodes_scatter.set_facecolor(colors)

        # 恢复边的颜色
        for i in self.highlighted_edges:
            self.visualizer.edges_lines[i].set_color(self.original_colors[f'edge_{i}'])

        # 清除记录
        self.highlighted_node = None
        self.highlighted_edges = []
        self.original_colors = {}

        # 更新显示
        self.visualizer.canvas.draw_idle()

    def _clear_search(self):
        """清除搜索"""
        self.toolbar.search_entry.delete(0, tk.END)
        self.visualizer.clear_selection()
