import time
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
import logging
from tkinter import ttk, messagebox, filedialog, simpledialog
import sys
import os
import numpy as np

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from GRAPH_VISUALIZATION.ui.toolbar import ToolBar
from GRAPH_VISUALIZATION.ui.status_bar import StatusBar
from GRAPH_VISUALIZATION.ui.canvas_manager import CanvasManager
from GRAPH_VISUALIZATION.ui.search_widget import SearchWidget
from GRAPH_VISUALIZATION.ui.progress_window import ProgressWindow
from GRAPH_VISUALIZATION.ui.shortest_path_dialog import ShortestPathDialog,VisualizePathDialog
from GRAPH_VISUALIZATION.handlers.mouse_handler import MouseHandler
from GRAPH_VISUALIZATION.handlers.zoom_handler import ZoomHandler
from GRAPH_VISUALIZATION.handlers.color_handler import ColorHandler
from GRAPH_VISUALIZATION.utils.style_utils import setup_matplotlib_style
from GRAPH_VISUALIZATION.utils.config import UI_CONFIG
from GRAPH_VISUALIZATION.core.graph_manager import GraphManager
from GRAPH_VISUALIZATION.core.node_attribute_panel import NodeAttributePanel
from GRAPH_VISUALIZATION.core.xlsx_to_graphml import Excel2GraphML
from GRAPH_VISUALIZATION.core.add_attr_dialog import AddAttributeDialog
from GRAPH_VISUALIZATION.utils.logger_config import setup_logger
from mc_learning.generate_paths import DataSetGenerate

# 配置日志记录器
logger = setup_logger()


class GraphVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("GraphML 可视化工具")

        # 配置主窗口网格
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 初始化变量
        self.graph_manager = GraphManager()
        self.graph_manager.visualizer = self  # 设置visualizer引用
        self.selected_node = None  # 选中结点 type:str
        self.show_labels = True  # 添加标签显示控制变量

        # 添加图形元素属性
        self.nodes_scatter = None  # 存储节点散点图对象
        self.edges_lines = []  # 存储边线条对象
        self.node_labels = {}  # 存储节点标签对象

        # 初始化处理器
        self.setup_handlers()  # 初始化处理器
        self.setup_ui()  # 设置UI
        self.setup_events()

        # 添加基础样式配置
        self.base_node_size = 500
        self.base_font_size = 10
        self.base_line_width = 1.0

        self.cur_base_node_size = (
                self.base_node_size * self.toolbar.get_node_size_scale() * self.zoom_handler.current_scale)

        # 添加缓存
        self._node_style_cache = {}  # 缓存节点样式
        self._edge_style_cache = {}  # 缓存边样式
        self._last_update_time = 0  # 上次更新时间
        self._update_threshold = 0.016  # 更新阈值（约60fps）

    def setup_handlers(self):
        """初始化所有处理器"""
        self.mouse_handler = MouseHandler(self)
        self.zoom_handler = ZoomHandler(self)
        self.color_handler = ColorHandler(self)

    def setup_ui(self):
        # 配置主窗口网格
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # 创建主框架
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # 配置主框架网格
        self.main_frame.grid_rowconfigure(1, weight=1)  # 画布行
        self.main_frame.grid_columnconfigure(0, weight=1)

        # 创建节点属性面板
        self.node_panel = NodeAttributePanel(self.main_frame, self.graph_manager)

        # 保存当前文件路径
        self.current_file = None

        # 创建画布管理器
        self.canvas_manager = CanvasManager(self.main_frame)
        self.fig = self.canvas_manager.fig
        self.ax = self.canvas_manager.ax
        self.canvas = self.canvas_manager.canvas

        # 创建状态栏
        self.status_bar = StatusBar(self.main_frame)

        callbacks = {
            'load_file': self.load_file,
            'save_image': self.save_image,
            'save_file': self.save_file,
            'relayout': self.relayout,
            'choose_color': self.color_handler.choose_color,
            'reset_colors': self.color_handler.reset_colors,
            'zoom': lambda x: self.zoom_handler.zoom_to_level(
                self.toolbar.get_zoom_value() + x  # 使用增量
            ),
            'zoom_entry': lambda e: self.zoom_handler.zoom_to_level(
                e  # 直接使用新值
            ),
            'add_custom_node_attribute': self.add_custom_node_attribute,
            'excel2graphml': self.open_convert_dialog,
            'toggle_labels': self.toggle_labels,  # 添加标签切换回调
            'update_sizes': self.update_sizes,  # 添加大小更新回调
            'shortest_path': self.show_shortest_path_dialog,  # 添加最短路径回调
            'visualize_path':self.show_hightlight_path_dialog
            # 'generate_paths': self.show_generate_paths_dialog,
            # 'self_learning': self.start_self_learning_iteration,  # 添加自学习迭代回调
            # 'start_performance_test': self.start_performance_test,  # 添加性能测试回调
            # 'stop_performance_test': self.stop_performance_test  # 添加性能测试回调
        }
        # 创建工具栏
        self.toolbar = ToolBar(self.main_frame, callbacks)
        # 创建搜索组件
        self.search_widget = SearchWidget(self.toolbar, self, self.node_panel)

        callbacks.update({'search': self.search_widget._handle_search,
                          'clear_search': self.search_widget._clear_search})

    def setup_events(self):
        """设置事件绑定"""
        self.canvas.mpl_connect('button_press_event', self.mouse_handler.on_press)
        self.canvas.mpl_connect('button_release_event', self.mouse_handler.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.mouse_handler.on_motion)
        self.canvas.mpl_connect('scroll_event', self.zoom_handler.zoom_by_scroll)

    def open_convert_dialog(self):
        dialog = tk.Toplevel(self.root)  # 创建子窗口
        app = Excel2GraphML(dialog)  # 让 Excel2GraphML 绑定到该子窗口
        dialog.mainloop()  # 运行子窗口事件循环

    def load_file(self):
        """加载文件"""
        file_path = filedialog.askopenfilename(
            filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")]
        )
        if file_path:
            self.current_file = file_path

            progress_window = ProgressWindow(self.root, "加载文件")

            try:
                # 启动KD树更新线程
                self.graph_manager.start_kd_tree_update_thread()

                # 加载图形
                if self.graph_manager.load_graph(file_path, progress_window):
                    # 将图转换为无向图
                    self.graph_manager.graph = self.graph_manager.graph.to_undirected()
                    
                    # 计算布局
                    progress_window.update_progress(0, "正在计算布局...")
                    self.graph_manager.calculate_layout("floor_based", self.base_node_size, progress_window)
                    progress_window.update_progress(50, "正在更新显示...")

                    # 创建属性编辑器
                    self.node_panel.create_attribute_editors()

                    # 刷新过滤模块
                    self.node_panel.refresh_filter_module()

                    # 初始化可见节点缓存
                    floor_filters = getattr(self.node_panel, 'floor_filters', {})
                    self.graph_manager.update_visible_nodes_cache(floor_filters)

                    # 更新显示
                    self.update_display()
                    progress_window.update_progress(80, "正在完成初始化...")

                    # 更新状态栏
                    self.status_bar.set_status(f"已加载文件: {file_path}")
                    progress_window.update_progress(100, "加载完成！")
                else:
                    messagebox.showerror("错误", "无法加载文件")
            finally:
                # 关闭进度条窗口
                progress_window.close()

    def save_file(self):
        """保存GraphML文件"""
        if not self.graph_manager.graph:
            messagebox.showwarning("警告", "没有图形可保存")
            return

        # 如果已有文件路径，直接保存
        if self.current_file:
            filename = self.current_file
        else:
            # 否则打开保存对话框
            filename = filedialog.asksaveasfilename(
                defaultextension=".graphml",
                filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")]
            )

        if filename:
            self.current_file = filename
            # 保存前确保图是无向的
            if self.graph_manager.graph.is_directed():
                self.graph_manager.graph = self.graph_manager.graph.to_undirected()
            if self.graph_manager.save_to_graphml(filename):
                messagebox.showinfo("成功", "文件保存成功")
            else:
                messagebox.showerror("错误", "保存文件时出错")

    def save_image(self):
        """保存图片"""
        if not self.graph_manager.graph:
            messagebox.showwarning("警告", "没有图形可以保存")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")]
        )
        if file_path:
            self.fig.savefig(file_path, bbox_inches='tight', dpi=500)
            self.status_bar.set_status(f"图片已保存至: {file_path}")

    def relayout(self):
        """重新布局"""
        if not self.graph_manager.graph:
            return

        # 创建进度条窗口
        progress_window = ProgressWindow(self.root, "重新布局")
        progress_window.update_progress(0, "正在计算新布局...")

        try:
            layout_name = self.toolbar.get_layout()
            self.graph_manager.calculate_layout(layout_name, self.base_node_size, progress_window)
            progress_window.update_progress(80, "正在更新显示...")

            # 更新KD树
            self.graph_manager.update_kd_tree()

            # 清除样式缓存，因为布局变化会影响节点样式
            self.clear_style_cache()

            # 更新显示
            self.update_display()
            
            # 更新样式，确保节点和边的样式正确
            self.update_styles()

            # 重置视图范围和缩放值
            self.zoom_handler.reset_view_after_layout()

            progress_window.update_progress(100, "布局更新完成！")
        except Exception as e:
            logger.error(f"重新布局时出错: {e}")
            progress_window.update_progress(100, f"布局更新失败: {str(e)}")
        finally:
            # 延迟关闭进度窗口，让用户看到完成状态
            self.root.after(1000, progress_window.close)

    def update_display(self):
        """更新图形显示"""
        if not self.graph_manager.graph:
            logger.warning("没有图形数据，无法更新显示")
            return

        logger.info("开始更新显示")
        self.ax.clear()
        self.edges_lines = []  # 清空边列表
        self.node_labels = {}  # 清空标签字典

        # 获取当前的楼层过滤状态
        floor_filters = getattr(self.node_panel, 'floor_filters', {})
        for floor, var in floor_filters.items():
            is_visible = var.get() if hasattr(var, 'get') else bool(var)
            logger.info(f"楼层 {floor}: {'可见' if is_visible else '不可见'}")

        # 更新KD树
        self.graph_manager.update_kd_tree()

        # 获取可见节点和边
        visible_nodes = self.graph_manager.get_visible_nodes()
        visible_edges = self.graph_manager.get_visible_edges()

        logger.info(f"可见节点数量: {len(visible_nodes)}")
        logger.info(f"可见边数量: {len(visible_edges)}")

        if not visible_nodes:
            logger.warning("没有可见节点，检查过滤条件是否正确")
            return

        # 绘制边
        for edge in visible_edges:
            try:
                # 使用graph_manager中的positions
                x1, y1 = self.graph_manager.pos[edge[0]][:2]  # 只取x,y坐标
                x2, y2 = self.graph_manager.pos[edge[1]][:2]
                # 使用楼层颜色
                edge_color = self.color_handler.get_edge_color(edge)
                line = self.ax.plot(
                    [x1, x2], [y1, y2],
                    color=edge_color,
                    linewidth=self.get_cur_line_width(edge),
                    zorder=1
                )[0]
                self.edges_lines.append(line)
            except Exception as e:
                logger.error(f"绘制边 {edge} 时出错: {e}")

        # 绘制节点
        try:
            node_positions = np.array([
                self.graph_manager.pos[node][:2]  # 只取x,y坐标
                for node in visible_nodes
            ])

            # 使用楼层颜色
            node_colors = [
                self.color_handler.get_node_color(node)
                for node in visible_nodes
            ]

            node_sizes = [
                self.get_cur_node_size(node)
                for node in visible_nodes
            ]

            # 创建散点图时使用调整后的大小
            if len(node_positions) > 0:  # 确保有可见的节点
                self.nodes_scatter = self.ax.scatter(
                    node_positions[:, 0],
                    node_positions[:, 1],
                    c=node_colors,
                    s=node_sizes,
                    zorder=2
                )
                logger.info(f"成功创建节点散点图，节点数量: {len(node_positions)}")
            else:
                logger.warning("没有节点位置数据，无法创建散点图")

            # 绘制标签时使用调整后的字体大小
            font_size = self.base_font_size / (self.zoom_handler.current_scale ** 0.5)
            for node in visible_nodes:
                try:
                    # 使用楼层颜色
                    node_color = self.color_handler.get_node_color(node)
                    x, y = self.graph_manager.pos[node][:2]  # 只取x,y坐标
                    label = self.ax.text(
                        x, y, str(node),
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=font_size,
                        color=node_color,  # 设置标签颜色
                        zorder=3,
                        bbox=dict(
                            facecolor='white',
                            edgecolor='none',
                            alpha=0.7,
                            pad=0.1 / (self.zoom_handler.current_scale ** 0.5),
                            boxstyle=f"round,pad={0.3 / (self.zoom_handler.current_scale ** 0.5)}"
                        ),
                        visible=self.show_labels  # 根据显示状态设置可见性
                    )
                    self.node_labels[node] = label
                except Exception as e:
                    logger.error(f"为节点 {node} 创建标签时出错: {e}")

            self.canvas.draw_idle()
            logger.info("显示更新完成")

        except Exception as e:
            logger.error(f"更新显示时出错: {e}")
            raise

    def toggle_labels(self):
        """切换标签显示状态"""
        self.show_labels = self.toolbar.get_label_state()
        # 更新所有标签的可见性
        for label in self.node_labels.values():
            label.set_visible(self.show_labels)
        self.canvas.draw_idle()

    def add_custom_node_attribute(self):
        """打开添加自定义属性的对话框"""
        dialog = AddAttributeDialog(self.root, self.graph_manager, self.node_panel)
        dialog.show_dialog()

    def update_sizes(self):
        """更新边宽度和节点大小"""
        if not self.graph_manager.graph or not self.nodes_scatter:
            return

        # 清除样式缓存，因为大小变化会影响样式
        self.clear_style_cache()

        # 获取可见节点和边
        visible_nodes = self.graph_manager.get_visible_nodes()
        visible_edges = self.graph_manager.get_visible_edges()

        # 更新节点大小
        self.cur_base_node_size = (
                self.base_node_size * self.toolbar.get_node_size_scale() * self.zoom_handler.current_scale)

        node_sizes = [self.get_cur_node_size(node) for node in visible_nodes]
        self.nodes_scatter.set_sizes(node_sizes)

        # 更新边的宽度
        # 确保edges_lines列表长度与可见边数量相同
        if len(self.edges_lines) > len(visible_edges):
            self.edges_lines = self.edges_lines[:len(visible_edges)]

        # 更新每条可见边的宽度
        for i, edge in enumerate(visible_edges):
            if i < len(self.edges_lines):  # 确保索引在有效范围内
                self.edges_lines[i].set_linewidth(self.get_cur_line_width(edge))

        # 重绘画布
        self.canvas.draw_idle()

    def update_styles(self):
        """更新节点和边的样式"""
        if not self.nodes_scatter:
            return

        current_time = time.time()
        if current_time - self._last_update_time < self._update_threshold:
            return

        self._last_update_time = current_time
        start_time = time.time()
        # 获取可见节点和边
        visible_nodes = self.graph_manager.get_visible_nodes()
        visible_edges = self.graph_manager.get_visible_edges()

        # 计算选中的节点及其邻居
        selected_node = self.selected_node
        if selected_node and selected_node in self.graph_manager.graph.nodes:
            get_neighbors_start = time.time()
            # neighbors_by_depth = self.mouse_handler._get_neighbors_by_depth(selected_node, 1)
            # visible_neighbors = {
            #     depth: {n for n in nodes if self.graph_manager.is_node_visible(n)}
            #     for depth, nodes in neighbors_by_depth.items()
            # }
            # all_visible_neighbors = set().union(*visible_neighbors.values()) if visible_neighbors else set()
            # 替换为直接获取1层邻居（时间复杂度 O(1)）
            if selected_node:
                all_neighbors = set(self.graph_manager.graph.neighbors(selected_node))
                all_visible_neighbors = {n for n in all_neighbors if self.graph_manager.is_node_visible(n)}
            else:
                all_visible_neighbors = set(visible_nodes)

            get_neighbors_end = time.time()
            logger.info(f"计算邻居节点耗时: {(get_neighbors_end - get_neighbors_start)*1000:.2f}ms")
        else:
            all_visible_neighbors = set(visible_nodes)

        # 批量更新节点样式
        node_colors = []
        node_sizes = []

        update_nodes_start = time.time()
        for node in visible_nodes:
            # 检查缓存是否需要更新
            cache_key = (node, selected_node, node in all_visible_neighbors)
            if cache_key not in self._node_style_cache:
                final_size = self.get_cur_node_size(node)
                color = self.color_handler.get_node_color(node)
                
                # 转换hex颜色到RGBA
                # 使用 numpy 批量处理
                if color.startswith('#'):
                    hex_bytes = np.frombuffer(bytes.fromhex(color[1:]), dtype=np.uint8)
                    rgba = np.concatenate([hex_bytes / 255, [1.0]])
                else:
                    rgba = [1.0, 1.0, 1.0, 1.0]

                # 设置透明度
                if selected_node is None or node == selected_node or node in all_visible_neighbors:
                    rgba[3] = 1.0
                else:
                    rgba[3] = 0.1

                self._node_style_cache[cache_key] = (rgba, final_size)

            rgba, final_size = self._node_style_cache[cache_key]
            node_colors.append(rgba)
            node_sizes.append(final_size)

            # 更新标签透明度
            if self.toolbar.get_label_state() and node in self.node_labels:
                self.node_labels[node].set_alpha(rgba[3])

        update_nodes_end = time.time()
        logger.info(f"更新节点样式耗时: {(update_nodes_end - update_nodes_start)*1000:.2f}ms")

        # 批量设置节点样式
        set_nodes_start = time.time()
        self.nodes_scatter.set_facecolors(node_colors)
        self.nodes_scatter.set_sizes(node_sizes)
        set_nodes_end = time.time()
        logger.info(f"设置节点样式耗时: {(set_nodes_end - set_nodes_start)*1000:.2f}ms")

        # 确保 edges_lines 长度匹配可见边
        if len(self.edges_lines) < len(visible_edges):
            return

        # 批量更新边样式
        update_edges_start = time.time()
        for i, edge in enumerate(visible_edges):
            # 检查缓存是否需要更新
            cache_key = (edge, selected_node, edge[0] in all_visible_neighbors and edge[1] in all_visible_neighbors)
            if cache_key not in self._edge_style_cache:
                final_width = self.get_cur_line_width(edge)
                edge_color = self.color_handler.get_edge_color(edge)
                
                # 处理边透明度
                if selected_node is None:
                    edge_alpha = 1.0
                elif edge[0] == selected_node or edge[1] == selected_node:
                    edge_alpha = 1.0
                elif edge[0] in all_visible_neighbors and edge[1] in all_visible_neighbors:
                    edge_alpha = 1.0
                else:
                    edge_alpha = 0.1

                self._edge_style_cache[cache_key] = (edge_color, final_width, edge_alpha)

            edge_color, final_width, edge_alpha = self._edge_style_cache[cache_key]
            self.edges_lines[i].set_color(edge_color)
            self.edges_lines[i].set_linewidth(final_width)
            self.edges_lines[i].set_alpha(edge_alpha)

        update_edges_end = time.time()
        logger.info(f"更新边样式耗时: {(update_edges_end - update_edges_start)*1000:.2f}ms")

        # 重绘画布
        draw_start = time.time()
        self.canvas.draw_idle()
        draw_end = time.time()
        logger.info(f"重绘画布耗时: {(draw_end - draw_start)*1000:.2f}ms")

        end_time = time.time()
        logger.info(f"update_styles总耗时: {(end_time - start_time)*1000:.2f}ms")

    def select_node(self, node):
        """选择节点
        Args:
            node: 节点ID或节点字典
        """
        # 确保我们处理的是节点ID
        node_id = node if isinstance(node, str) else node.get('id')

        if node_id == self.selected_node:
            # 如果点击已选中的节点，取消选中
            self.selected_node = None
        else:
            """选择节点"""
            self.selected_node = node_id
            self.status_bar.set_status(f"已选择节点: {node_id}")
            # 更新属性面板
            if node_id in self.graph_manager.graph.nodes:
                node_attrs = self.graph_manager.graph.nodes[node_id]
                self.node_panel.display_node_attributes(node_attrs)

    def clear_selection(self):
        """清除选择"""
        self.selected_node = None
        self.status_bar.clear()
        self.update_styles()

    def update_node_position(self, node, new_pos):
        """更新节点位置"""
        if node not in self.graph_manager.pos:
            return

        # 更新节点位置
        self.graph_manager.update_node_position(node, new_pos)

        # 更新KD树
        self.graph_manager.update_kd_tree()

        # 获取可见节点和边
        visible_nodes = self.graph_manager.get_visible_nodes()
        visible_edges = self.graph_manager.get_visible_edges()

        # 使用numpy进行批量位置更新
        node_positions = np.array([self.graph_manager.pos[n] for n in visible_nodes])
        self.nodes_scatter.set_offsets(node_positions)

        # 批量更新边的位置
        edge_data = []
        for edge in visible_edges:
            if edge[0] in self.graph_manager.pos and edge[1] in self.graph_manager.pos:
                start_pos = self.graph_manager.pos[edge[0]]
                end_pos = self.graph_manager.pos[edge[1]]
                edge_data.append((start_pos, end_pos))

        # 更新边的位置
        for i, (start_pos, end_pos) in enumerate(edge_data):
            if i < len(self.edges_lines):
                self.edges_lines[i].set_data(
                    [start_pos[0], end_pos[0]],
                    [start_pos[1], end_pos[1]]
                )
                self.ax.draw_artist(self.edges_lines[i])

        # 批量更新标签位置
        label_updates = []
        for node in visible_nodes:
            if node in self.node_labels:
                label_updates.append((node, self.graph_manager.pos[node]))

        # 更新标签位置
        for node, pos in label_updates:
            self.node_labels[node].set_position(pos)
            self.ax.draw_artist(self.node_labels[node])

        # 重绘散点图
        self.ax.draw_artist(self.nodes_scatter)

        # 重绘画布
        self.canvas.draw_idle()

    def show_shortest_path_dialog(self):
        """显示最短路径对话框"""
        dialog = ShortestPathDialog(self.root, self.graph_manager, self)
        dialog.show_dialog()

    def show_hightlight_path_dialog(self):
        dialog = VisualizePathDialog(self.root,self.graph_manager,self)
        dialog.show_dialog()

    def show_generate_paths_dialog(self):
        """显示生成路径对话框"""
        # if not self.graph_manager.graph:
        #     messagebox.showerror("错误", "请先加载图形数据")
        #     return
        dataset_generate = DataSetGenerate()
        # dataset_generate = DataSetGenerate(self.root, self)
        # dataset_generate.show_dialog()

    # def start_self_learning_iteration(self):
    #     """启动自学习迭代过程"""
    #     if not self.graph_manager.graph:
    #         messagebox.showerror("错误", "请先加载图形数据")
    #         return
    #
    #     from core.self_learning_iter import SelfLearningTrainer
    #
    #     # 创建 DataSetGenerate 实例
    #     dataset_generate = DataSetGenerate(self.root, self)
    #
    #     # 设置迭代次数（可以根据需要调整）
    #     num_iterations = 5
    #     csv_file = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\GRAPH_VISUALIZATION\\generated_paths_dataset.csv"
    #     weights_path = "/best_model.pt"
    #     self_learning_trainer = SelfLearningTrainer(dataset_generate, csv_file, weights_path)
    #
    #     # 开始训练（自学习 5 轮）
    #     self_learning_trainer.train(num_iterations=5)
    #
    #     messagebox.showinfo("成功", f"完成 {num_iterations} 次自学习迭代")

    def get_cur_node_size(self, node):
        node_scale = self.toolbar.get_node_size_scale()
        return (self.base_node_size * node_scale * self.zoom_handler.current_scale) * (
            1.4 if node == self.selected_node else 1.0)

    def get_cur_line_width(self, edge):
        edge_scale = self.toolbar.get_edge_width_scale()
        base_width = self.base_line_width * edge_scale * self.zoom_handler.zoom_factor
        size_multiplier = 2 if self.selected_node in edge else 1.0
        return base_width * size_multiplier

    # def start_performance_test(self):
    #     """开始性能测试"""
    #     self.mouse_handler.start_performance_test()
    #     self.status_bar.set_status("性能测试已开始")
    #
    # def stop_performance_test(self):
    #     """停止性能测试"""
    #     self.mouse_handler.stop_performance_test()
    #     self.status_bar.set_status("性能测试已结束")

    def __del__(self):
        """析构函数，确保清理资源"""
        if hasattr(self, 'graph_manager'):
            self.graph_manager.stop_kd_tree_update_thread()

    def clear_style_cache(self):
        """清除样式缓存"""
        self._node_style_cache.clear()
        self._edge_style_cache.clear()
