import math
import numpy as np
import time
import threading
from queue import Queue
from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()

class LogWorker(threading.Thread):
    """日志工作线程"""
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.daemon = True  # 设置为守护线程，这样主程序退出时线程会自动结束
        self.start()

    def run(self):
        while True:
            try:
                # 从队列中获取日志消息
                message = self.queue.get()
                if message is None:  # 退出信号
                    break
                logger.info(message)
            except Exception as e:
                print(f"Error in log worker: {e}")

    def log(self, message):
        """添加日志消息到队列"""
        self.queue.put(message)

    def stop(self):
        """停止日志工作线程"""
        self.queue.put(None)

class MouseHandler:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.dragged_node = None
        self.is_panning = False
        self.pan_start = None
        self.initial_pos = None
        self.hover_annotation = None
        self.background = None  # 用于存储背景
        self.last_update_time = 0  # 添加最后更新时间记录
        self.update_threshold = 0.016  # 更新阈值，约60fps
        # 添加性能测试相关变量
        self.click_times = []  # 存储点击响应时间
        self.last_click_time = 0  # 上次点击时间
        self.performance_testing = False  # 是否进行性能测试
        # 创建日志工作线程
        self.log_worker = LogWorker()

    def __del__(self):
        """析构函数，确保日志线程正确停止"""
        if hasattr(self, 'log_worker'):
            self.log_worker.stop()

    def start_performance_test(self):
        """开始性能测试"""
        self.performance_testing = True
        self.click_times = []

    def stop_performance_test(self):
        """停止性能测试并输出统计信息"""
        self.performance_testing = False
        if self.click_times:
            avg_time = sum(self.click_times) / len(self.click_times)
            max_time = max(self.click_times)
            min_time = min(self.click_times)
            self.log_worker.log(f"Performance Test Results:")
            self.log_worker.log(f"Total clicks: {len(self.click_times)}")
            self.log_worker.log(f"Average response time: {avg_time*1000:.2f}ms")
            self.log_worker.log(f"Min response time: {min_time*1000:.2f}ms")
            self.log_worker.log(f"Max response time: {max_time*1000:.2f}ms")

    def _handle_node_drag(self, event):
        """处理节点拖动"""
        if not self.dragged_node:
            return

        current_time = time.time()
        if current_time - self.last_update_time < self.update_threshold:
            return

        self.last_update_time = current_time
        # 获取当前的楼层过滤状态
        floor_filters = getattr(self.visualizer.node_panel, 'floor_filters', {})

        # 检查被拖动节点的可见性
        node_floor = self.visualizer.graph_manager.graph.nodes[self.dragged_node].get('lc')
        node_visible = True
        if node_floor is not None:
            try:
                node_floor_num = int(float(node_floor))
                if node_floor_num in floor_filters:
                    node_visible = floor_filters[node_floor_num].get()

            except (ValueError, TypeError):
                pass

        if not node_visible:
            self.dragged_node = None
            return

        # 获取数据坐标
        data_coords = self.visualizer.ax.transData.inverted().transform((event.x, event.y))

        # 第一次拖动时初始化
        if self.background is None:
            try:
                self.background = self.visualizer.canvas.copy_from_bbox(self.visualizer.ax.bbox)
            except Exception as e:
                logger.error(f"Error initializing drag background: {e}")
                return

        # 恢复背景
        try:
            self.visualizer.canvas.restore_region(self.background)
        except Exception as e:
            logger.error(f"Error restoring background: {e}")
            return

        # 更新节点位置
        self.visualizer.update_node_position(self.dragged_node, data_coords)
        self.visualizer.update_styles()

    def _get_neighbors_by_depth(self, node, max_depth):
        """获取指定深度的所有相邻节点
        Args:
            node: 节点ID
            max_depth: 最大深度
        Returns:
            Dict[int, Set[str]]: 按深度分组的邻居节点ID集合
        """
        # logger.debug(f"Getting neighbors by depth for node {node} with max_depth {max_depth}")
        # logger.debug(f"Graph exists: {self.visualizer.graph_manager.graph is not None}")

        neighbors_by_depth = {0: {node}}
        visited = {node}

        for depth in range(1, max_depth + 1):
            # logger.debug(f"Processing depth {depth}")
            neighbors_by_depth[depth] = set()
            for prev_node in neighbors_by_depth[depth - 1]:
                # logger.debug(f"Getting neighbors for node {prev_node} at depth {depth}")
                current_neighbors = set(self.visualizer.graph_manager.get_node_neighbors(prev_node))
                # logger.debug(f"Found {len(current_neighbors)} neighbors for node {prev_node}")
                # 只添加未访问的节点
                new_neighbors = current_neighbors - visited
                # logger.debug(f"New unvisited neighbors: {new_neighbors}")
                neighbors_by_depth[depth].update(new_neighbors)
                visited.update(new_neighbors)

        # logger.debug(f"Final neighbors by depth: {neighbors_by_depth}")
        return neighbors_by_depth

    def on_motion(self, event):
        """处理鼠标移动事件"""
        if event.inaxes != self.visualizer.ax or not self.visualizer.graph_manager.graph:
            return

        if self.dragged_node:
            self._handle_node_drag(event)
        elif self.is_panning:
            self._handle_panning(event)
        else:
            self._handle_hover(event)

    def on_press(self, event):
        """处理鼠标按下事件"""
        if event.inaxes != self.visualizer.ax or not self.visualizer.graph_manager.graph:
            return

        if event.button == 1:  # 左键点击
            start_time = time.time()
            find_node_start = time.time()

            # 过滤出可见的节点
            visible_nodes = self.visualizer.graph_manager.get_visible_nodes()

            # 查找最近的可见节点
            clicked_node = None
            if visible_nodes:  # type:list[str]
                # 将屏幕坐标转换为数据坐标
                data_coords = self.visualizer.ax.transData.inverted().transform((event.x, event.y))
                clicked_node, distance = self.visualizer.graph_manager.get_nearest_node(
                    data_coords[0],  # x坐标
                    data_coords[1],  # y坐标
                    max_distance=self.get_radius()  # 使用节点半径作为最大搜索距离
                )

            find_node_end = time.time()
            self.log_worker.log(f"查找最近节点耗时: {(find_node_end - find_node_start) * 1000:.2f}ms")

            if clicked_node in visible_nodes:
                # 记录选中的节点，并更新样式
                self.dragged_node = clicked_node
                update_color_start = time.time()
                self.visualizer.select_node(clicked_node)
                update_color_end = time.time()
                self.log_worker.log(f"更新节点颜色耗时: {(update_color_end - update_color_start) * 1000:.2f}ms")
            else:
                self.dragged_node = None
                self.visualizer.clear_selection()
            self.visualizer.update_styles()

            # 记录性能数据
            if self.performance_testing:
                end_time = time.time()
                response_time = end_time - start_time
                self.click_times.append(response_time)
                self.log_worker.log(f"总响应时间: {response_time * 1000:.2f}ms")

        elif event.button == 3:  # 右键平移画布
            self.is_panning = True
            self.pan_start = (event.xdata, event.ydata)

            # 保存所有节点的初始位置，将元组转换为列表以便复制
            self.initial_pos = {node: list(pos) for node, pos in self.visualizer.graph_manager.pos.items()}
            self.visualizer.canvas.get_tk_widget().config(cursor="fleur")

    def _handle_panning(self, event):
        """处理画布拖动"""
        if not self.is_panning or not self.pan_start:
            return

        current_time = time.time()
        if current_time - self.last_update_time < self.update_threshold:
            return

        self.last_update_time = current_time

        # 计算拖动距离
        dx = event.xdata - self.pan_start[0]
        dy = event.ydata - self.pan_start[1]

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

        # 更新可见节点的位置
        visible_nodes = [node for node in self.visualizer.graph_manager.graph.nodes() if node_visibility[node]]
        positions_update = []
        for node in visible_nodes:
            if node in self.initial_pos:
                initial_pos = self.initial_pos[node]
                new_pos = [initial_pos[0] + dx, initial_pos[1] + dy]
                self.visualizer.graph_manager.pos[node] = new_pos
                positions_update.append(new_pos)

        # 使用numpy进行批量位置更新
        if positions_update:
            new_positions = np.array(positions_update)
            try:
                self.visualizer.nodes_scatter.set_offsets(new_positions)
            except Exception as e:
                logger.error(f"Error updating node scatter positions: {e}")
                return

        # 恢复背景
        if self.background is not None:
            try:
                self.visualizer.canvas.restore_region(self.background)
            except Exception as e:
                logger.error(f"Error restoring background: {e}")

        # 获取可见边
        visible_edges = [edge for edge in self.visualizer.graph_manager.graph.edges()
                         if node_visibility[edge[0]] and node_visibility[edge[1]]]

        # 更新可见边的位置
        try:
            for i, edge in enumerate(visible_edges):
                if i < len(self.visualizer.edges_lines):
                    start_pos = self.visualizer.graph_manager.pos[edge[0]]
                    end_pos = self.visualizer.graph_manager.pos[edge[1]]
                    self.visualizer.edges_lines[i].set_data(
                        [start_pos[0], end_pos[0]],
                        [start_pos[1], end_pos[1]]
                    )
                    self.visualizer.ax.draw_artist(self.visualizer.edges_lines[i])
        except Exception as e:
            logger.error(f"Error updating edge positions: {e}")
            return

        # 更新节点散点图
        try:
            self.visualizer.ax.draw_artist(self.visualizer.nodes_scatter)
        except Exception as e:
            logger.error(f"Error drawing node scatter: {e}")

        # 更新可见节点的标签位置
        for node in visible_nodes:
            if node in self.visualizer.node_labels:
                try:
                    pos = self.visualizer.graph_manager.pos[node]
                    self.visualizer.node_labels[node].set_position(pos)
                    self.visualizer.ax.draw_artist(self.visualizer.node_labels[node])
                except Exception as e:
                    logger.error(f"Error updating label position for node {node}: {e}")

        # 重绘画布
        try:
            self.visualizer.canvas.draw_idle()
        except Exception as e:
            logger.error(f"Error updating canvas: {e}")

    def on_release(self, event):
        """处理鼠标释放事件"""
        if self.dragged_node:
            self.dragged_node = None
            self.background = None
            # 完整重绘一次以确保显示正确
            self.visualizer.canvas.draw_idle()

        if self.is_panning:
            self.is_panning = False
            self.pan_start = None
            self.initial_pos = None
            self.visualizer.canvas.get_tk_widget().config(cursor="arrow")

    # *******悬停功能********** #
    def _update_tooltip(self, x, y, text):
        """更新工具提示

        Args:
            x: 屏幕坐标x
            y: 屏幕坐标y
            text: 工具提示文本
        """
        # 移除旧的工具提示
        if self.hover_annotation:
            self.hover_annotation.remove()

        # 将屏幕坐标转换为数据坐标
        inv = self.visualizer.ax.transData.inverted()
        data_x, data_y = inv.transform((x, y))

        # 创建新的工具提示
        self.hover_annotation = self.visualizer.ax.annotate(
            text,
            xy=(data_x, data_y),
            xytext=(10, 10),  # 文本偏移量
            textcoords='offset points',
            bbox=dict(
                boxstyle='round,pad=0.5',
                fc='white',
                alpha=0.8,
                ec='gray'
            ),
            zorder=5
        )

        # 重绘画布
        self.visualizer.canvas.draw_idle()

    def _hide_tooltip(self):
        """隐藏工具提示"""
        if self.hover_annotation:
            self.hover_annotation.remove()
            self.hover_annotation = None
            self.visualizer.canvas.draw_idle()

    def _handle_hover(self, event):
        """处理鼠标悬停事件"""
        if not event.inaxes or not self.visualizer.graph_manager.pos:
            self._hide_tooltip()
            return
        # 过滤出可见的节点
        visible_nodes = self.visualizer.graph_manager.get_visible_nodes()
        hovered_node = None
        # logger.debug(f"Number of visible nodes: {len(visible_nodes)}")

        # 查找最近的可见节点
        if visible_nodes:  # type:list[str]
            visible_pos = {node: self.visualizer.graph_manager.pos[node] for node in visible_nodes}
            # 将屏幕坐标转换为数据坐标
            data_coords = self.visualizer.ax.transData.inverted().transform((event.x, event.y))
            hovered_node, distance = self.visualizer.graph_manager.get_nearest_node(
                data_coords[0],  # x坐标
                data_coords[1],  # y坐标
                max_distance=self.get_radius()  # 使用节点半径作为最大搜索距离
            )

        # 如果找到了节点，显示工具提示
        if hovered_node and hovered_node in visible_nodes:
            # 获取节点信息
            node_info = self.visualizer.graph_manager.get_node_info(hovered_node)
            if node_info:
                # 创建工具提示文本
                tooltip_text = f"节点ID: {hovered_node}\n"
                for attr, value in node_info.items():
                    if attr != 'Connect_Entity_Id_List':  # 跳过连接实体列表
                        tooltip_text += f"{attr}: {value}\n"

                # 更新工具提示
                self._update_tooltip(event.x, event.y, tooltip_text)
                return

        # 如果没有找到节点，隐藏工具提示
        self._hide_tooltip()

    def get_radius(self):
        xlim = self.visualizer.ax.get_xlim()
        ylim = self.visualizer.ax.get_ylim()
        width, height = self.visualizer.fig.get_size_inches() * self.visualizer.fig.get_dpi()
        x_scale = width / (xlim[1] - xlim[0])
        y_scale = height / (ylim[1] - ylim[0])

        cur_node_size = self.visualizer.cur_base_node_size
        return math.sqrt(cur_node_size / math.pi) / min(x_scale, y_scale)
