import time
import numpy as np
from threading import Lock
from collections import deque
from concurrent.futures import ThreadPoolExecutor


class ZoomHandler:
    def __init__(self, visualizer):
        self.visualizer = visualizer

        self.base_node_size = 500  # 基础节点大小
        self.base_font_size = 10  # 基础字体大小
        self.base_line_width = 1.0  # 基础线宽
        self.current_scale = 1.0  # 当前缩放比例

        self.zoom_level = 100  # 当前缩放级别（百分比）
        self.min_zoom = 20  # 最小缩放级别（百分比）
        self.max_zoom = 500  # 最大缩放级别（百分比）
        self.zoom_factor = 1.0  # 实际缩放因子

        # 添加初始视图范围记录
        self.initial_xlim = None
        self.initial_ylim = None
        self.initial_center_x = None
        self.initial_center_y = None

        # 性能优化相关属性
        self.last_draw_time = 0
        self.draw_interval = 0.05  # 降低到20fps，提高性能
        self.draw_lock = Lock()
        self.update_queue = deque(maxlen=10)  # 增加队列长度，更好地处理快速滚动
        self.is_drawing = False
        self.executor = ThreadPoolExecutor(max_workers=1)  # 减少线程数，避免竞争

        # 缓存相关
        self.background = None
        self.last_view = None
        self.view_margin = 0.2  # 增加视图边界余量，减少重绘频率

        # 添加累积缩放控制
        self.accumulated_zoom = 1.0
        self.last_zoom_time = 0
        self.zoom_cooldown = 0.1  # 缩放冷却时间

    def zoom_to_level(self, level):
        """缩放到指定百分比级别"""
        try:
            # print(f"[ZoomHandler.zoom_to_level] 接收缩放级别: {level}")
            # 确保level是整数
            level = int(level)
            # 限制范围
            level = max(self.min_zoom, min(self.max_zoom, level))
            # print(f"[ZoomHandler.zoom_to_level] 限制后的级别: {level}%")

            # 更新缩放级别
            self.zoom_level = level
            self.zoom_factor = level / 100.0
            # print(f"[ZoomHandler.zoom_to_level] 更新缩放因子: {self.zoom_factor}")

            # 应用缩放
            self._apply_zoom()

        except ValueError as e:
            print(f"[ZoomHandler.zoom_to_level] 错误: {e}")

    def zoom_by_scroll(self, event):
        """处理滚轮缩放"""
        if event.inaxes != self.visualizer.ax:
            return

        # 获取鼠标位置
        mouse_x, mouse_y = event.xdata, event.ydata
        if mouse_x is None or mouse_y is None:
            return

        # 计算新的缩放级别（每次改变10%）
        if event.button == 'up' or (hasattr(event, 'delta') and event.delta > 0):
            new_zoom = min(self.max_zoom, self.zoom_level + 10)
        else:
            new_zoom = max(self.min_zoom, self.zoom_level - 10)

        # 更新工具栏显示的缩放值
        if hasattr(self.visualizer, 'toolbar'):
            self.visualizer.toolbar.zoom_var.set(f"{new_zoom}%")
            # print(f"[ZoomHandler.zoom_by_scroll] 更新工具栏缩放值: {new_zoom}%")

        # 更新缩放级别
        self.zoom_level = new_zoom
        self.zoom_factor = new_zoom / 100.0

        # 应用缩放，以鼠标位置为中心
        self._apply_zoom(center_x=mouse_x, center_y=mouse_y)

    def _fast_zoom(self, factor, center_x, center_y):
        """快速缩放实现"""
        try:
            with self.draw_lock:
                # 更新缩放级别，添加限制
                new_zoom = np.clip(self.zoom_level * factor, self.min_zoom, self.max_zoom)
                self.zoom_level = new_zoom
                self.zoom_factor = new_zoom / 100.0
                self.current_scale = self.zoom_factor

                # 获取并更新视图范围
                ax = self.visualizer.ax
                x_min, x_max = ax.get_xlim()
                y_min, y_max = ax.get_ylim()

                # 计算新的视图范围
                width = (x_max - x_min)
                height = (y_max - y_min)

                # 相对于鼠标位置进行缩放
                dx = (center_x - x_min) / width
                dy = (center_y - y_min) / height

                new_width = width / factor
                new_height = height / factor

                new_x_min = center_x - dx * new_width
                new_x_max = new_x_min + new_width
                new_y_min = center_y - dy * new_height
                new_y_max = new_y_min + new_height

                # 设置新的视图范围
                ax.set_xlim(new_x_min, new_x_max)
                ax.set_ylim(new_y_min, new_y_max)

                # 更新所有视觉元素
                self.update_visual_elements()

                # 使用快速更新方式
                self.visualizer.canvas.draw_idle()

                self.last_draw_time = time.time()
        except Exception as e:
            print(f"Zoom error: {e}")

    # def _fast_update(self):
    #     """快速更新显示"""
    #     try:
    #         # 检查是否需要重新生成背景
    #         current_view = self.visualizer.ax.get_xlim() + self.visualizer.ax.get_ylim()
    #         if (self.background is None or self.last_view is None or
    #             any(abs(c - l) > self.view_margin * abs(l) for c, l in zip(current_view, self.last_view))):
    #             self.background = self.visualizer.canvas.copy_from_bbox(self.visualizer.ax.bbox)
    #             self.last_view = current_view
    #
    #         # 恢复背景
    #         self.visualizer.canvas.restore_region(self.background)
    #
    #         # 批量更新视觉元素
    #         self._update_visual_elements_fast()
    #
    #         # 快速绘制
    #         self.visualizer.canvas.draw_idle()
    #         self.visualizer.toolbar.set_zoom_value(self.zoom_level)
    #     except Exception as e:
    #         print(f"Update error: {e}")
    #
    # def _update_visual_elements_fast(self):
    #     """快速更新视觉元素"""
    #     try:
    #         if not self.visualizer.nodes_scatter:
    #             return
    #
    #         node_scale = self.visualizer.toolbar.get_node_size_scale()
    #         edge_scale = self.visualizer.toolbar.get_edge_width_scale()
    #
    #         # 批量更新节点大小
    #         node_sizes = np.array([
    #             self.base_node_size * node_scale * (self.current_scale ** 0.5) *
    #             (1.4 if node == self.visualizer.selected_node else 1.0)
    #             for node in self.visualizer.graph_manager.graph.nodes()
    #         ])
    #         self.visualizer.nodes_scatter.set_sizes(node_sizes)
    #         self.visualizer.ax.draw_artist(self.visualizer.nodes_scatter)
    #
    #         # 批量更新边的线宽
    #         base_width = self.base_line_width * edge_scale * (self.current_scale ** 0.25)
    #         for i, (edge, line) in enumerate(zip(self.visualizer.graph_manager.graph.edges(),
    #                                            self.visualizer.edges_lines)):
    #             is_highlighted = self.visualizer.selected_node in edge
    #             line.set_linewidth(base_width * (2.0 if is_highlighted else 1.0))
    #             self.visualizer.ax.draw_artist(line)
    #
    #         # 根据缩放级别决定是否显示标签
    #         if self.zoom_level > 50:  # 只在一定缩放级别以上显示标签
    #             font_scale = max(0.6, min(1.5, 1.0 / (self.current_scale ** 0.2)))
    #             font_size = self.base_font_size * font_scale
    #             pad = 0.1 * font_scale
    #             for label in self.visualizer.node_labels.values():
    #                 label.set_fontsize(font_size)
    #                 label.set_bbox(dict(
    #                     facecolor='white',
    #                     edgecolor='none',
    #                     alpha=0.7,
    #                     pad=pad,
    #                     boxstyle=f"round,pad={0.3 * font_scale}"
    #                 ))
    #                 self.visualizer.ax.draw_artist(label)
    #     except Exception as e:
    #         print(f"Visual update error: {e}")

    def _process_update_queue(self):
        """处理更新队列"""
        if self.is_drawing:
            return

        self.is_drawing = True
        try:
            while self.update_queue:
                time.sleep(self.draw_interval)
                if not self.update_queue:
                    break

                # 处理队列中的最新事件
                _, event = self.update_queue.pop()
                self.zoom_by_scroll(event)

                # 清空队列中的旧事件
                self.update_queue.clear()
        finally:
            self.is_drawing = False

    def set_initial_view(self):
        """设置初始视图范围"""
        ax = self.visualizer.ax
        self.initial_xlim = ax.get_xlim()
        self.initial_ylim = ax.get_ylim()
        self.initial_center_x = (self.initial_xlim[0] + self.initial_xlim[1]) / 2
        self.initial_center_y = (self.initial_ylim[0] + self.initial_ylim[1]) / 2
        # print(f"[ZoomHandler.set_initial_view] 设置初始视图范围: x={self.initial_xlim}, y={self.initial_ylim}")

    def _apply_zoom(self, center_x=None, center_y=None):
        """应用缩放效果"""
        try:
            #  print(f"[ZoomHandler._apply_zoom] 开始应用缩放，当前级别: {self.zoom_level}%")

            # 如果没有初始视图范围，先设置它
            if self.initial_xlim is None:
                self.set_initial_view()

            # 获取当前视图范围
            ax = self.visualizer.ax
            current_xlim = ax.get_xlim()
            current_ylim = ax.get_ylim()

            # 使用初始视图范围计算新的视图范围
            initial_width = self.initial_xlim[1] - self.initial_xlim[0]
            initial_height = self.initial_ylim[1] - self.initial_ylim[0]

            # 计算新的视图范围
            width = initial_width / self.zoom_factor
            height = initial_height / self.zoom_factor
            # (f"[ZoomHandler._apply_zoom] 新视图范围: width={width}, height={height}")

            # 使用初始中心点或提供的中心点
            if center_x is None:
                center_x = self.initial_center_x
            if center_y is None:
                center_y = self.initial_center_y
            # print(f"[ZoomHandler._apply_zoom] 缩放中心点: ({center_x}, {center_y})")

            # 计算当前视图中的相对位置
            current_width = current_xlim[1] - current_xlim[0]
            current_height = current_ylim[1] - current_ylim[0]

            # 计算鼠标在视图中的相对位置
            dx = (center_x - current_xlim[0]) / current_width
            dy = (center_y - current_ylim[0]) / current_height

            # 计算新的视图范围，保持鼠标位置不变
            new_x_min = center_x - dx * width
            new_x_max = new_x_min + width
            new_y_min = center_y - dy * height
            new_y_max = new_y_min + height

            # 设置新的视图范围
            ax.set_xlim(new_x_min, new_x_max)
            ax.set_ylim(new_y_min, new_y_max)

            # 更新视觉元素
            self.update_visual_elements()

            # 重绘画布
            self.visualizer.canvas.draw_idle()
            # print(f"[ZoomHandler._apply_zoom] 缩放应用完成")

        except Exception as e:
            print(f"[ZoomHandler._apply_zoom] 错误: {e}")

    def update_visual_elements(self):
        """更新所有视觉元素"""
        try:
            # 获取当前的楼层过滤状态
            floor_filters = getattr(self.visualizer.node_panel, 'floor_filters', {})

            # 创建节点可见性映射
            node_visibility = {}
            for node in self.visualizer.graph_manager.graph.nodes():
                node_floor = self.visualizer.graph_manager.graph.nodes[node].get('lc')
                if node_floor is not None:
                    try:
                        node_floor_num = int(float(node_floor))
                        if node_floor_num in floor_filters:
                            node_visibility[node] = floor_filters[node_floor_num].get()
                        else:
                            node_visibility[node] = True
                    except (ValueError, TypeError):
                        node_visibility[node] = True
                else:
                    node_visibility[node] = True

            # 获取可见节点列表
            visible_nodes = [node for node in self.visualizer.graph_manager.graph.nodes() if node_visibility[node]]

            # 更新节点大小
            node_size = [self.visualizer.get_cur_node_size(node) * self.current_scale for node in visible_nodes]
            self.visualizer.nodes_scatter.set_sizes(node_size)

            # 更新节点位置
            visible_positions = [self.visualizer.graph_manager.pos[node] for node in visible_nodes]
            if visible_positions:
                self.visualizer.nodes_scatter.set_offsets(np.array(visible_positions))

            # 更新边的宽度
            line_width = self.base_line_width * self.current_scale

            # 获取可见边
            visible_edges = [edge for edge in self.visualizer.graph_manager.graph.edges()
                             if node_visibility[edge[0]] and node_visibility[edge[1]]]

            # 检查edges_lines列表长度是否匹配
            if len(self.visualizer.edges_lines) != len(visible_edges):
                self.visualizer.update_display()
                return

            # 更新边的属性
            for line in self.visualizer.edges_lines:
                line.set_linewidth(line_width)

            # 若标签可见
            if self.visualizer.toolbar.get_label_state():
                # 更新标签大小和位置
                font_size = self.base_font_size * self.current_scale
                for node in visible_nodes:
                    if node in self.visualizer.node_labels:
                        label = self.visualizer.node_labels[node]
                        label.set_fontsize(font_size)
                        pos = self.visualizer.graph_manager.pos[node]
                        label.set_position(pos)
                        label.set_visible(True)

                # 隐藏不可见节点的标签
                for node in self.visualizer.node_labels:
                    if node not in visible_nodes:
                        self.visualizer.node_labels[node].set_visible(False)

        except Exception as e:
            print(f"Visual update error: {e}")
            import traceback
            traceback.print_exc()

    def reset_view_after_layout(self):
        """布局更新后重置视图范围和缩放值"""
        try:
            # print("[ZoomHandler.reset_view_after_layout] 开始重置视图")

            # 重置缩放级别和因子
            self.zoom_level = 100
            self.zoom_factor = 1.0
            self.current_scale = 1.0

            # 重新设置初始视图范围
            self.set_initial_view()

            # 应用初始视图范围
            ax = self.visualizer.ax
            ax.set_xlim(self.initial_xlim)
            ax.set_ylim(self.initial_ylim)

            # 更新视觉元素
            self.update_visual_elements()

            # 重绘画布
            self.visualizer.canvas.draw_idle()

            # 更新工具栏显示的缩放值
            if hasattr(self.visualizer, 'toolbar'):
                self.visualizer.toolbar.zoom_var.set("100%")

            # print("[ZoomHandler.reset_view_after_layout] 视图重置完成")

        except Exception as e:
            print(f"[ZoomHandler.reset_view_after_layout] 错误: {e}")
