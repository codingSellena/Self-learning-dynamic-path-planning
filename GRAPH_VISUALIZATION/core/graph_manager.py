import time
from typing import Dict, Tuple
import threading
from queue import Queue
import queue

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

from GRAPH_VISUALIZATION.utils.logger_config import setup_logger
from GRAPH_VISUALIZATION.core.grid_layout import GridCrossingLayout

# 配置日志记录器
logger = setup_logger()


class GraphManager:
    """管理图数据和布局的类"""

    def __init__(self):
        self.graph = None
        self.pos = {}
        self.initial_bounds = None
        self.custom_node_attributes = {}
        # 定义楼层颜色映射
        self.floor_colors = {
            '1': '#3498db',  # 蓝色
            '2': '#ff69b4',  # 粉色
            '3': '#ff7f50',  # 橙色
            '4': '#9b59b6',  # 紫色
            '5': '#2ecc71',  # 绿色
            '6': '#e74c3c',  # 红色
            'default': '#95a5a6'  # 默认灰色
        }
        # 定义节点属性
        self.base_attributes = {
            'Id': {
                'type': 'int',
                'default': 0,
                'description': '数据库ID'
            },
            'Elem_Id': {
                'type': 'str',
                'default': '',
                'description': '唯一标识符'
            },
            'Name': {
                'type': 'str',
                'default': '',
                'description': '实体名称'
            },
            'Capacity': {
                'type': 'double',
                'default': 0,
                'min': 0,
                'description': '最大容量'
            },
            'Entity_Type': {
                'type': 'str',
                'default': 'Undefined',
                'options': ['室内', '楼道', '楼梯口', '房间门', '建筑物出口'],
                'description': '实体类型'
            },
            'Safety_Level': {
                'type': 'str',
                'default': 'Undefined',
                'options': ['Very Safe', 'Safe', 'Normal', 'Dangerous', 'Very Dangerous', 'Undefined'],
                'description': '安全等级'
            },
            'Connect_Entity_Id_List': {
                'type': 'list',
                'default': [],
                'description': '连接实体列表'
            },
            'Center': {
                'type': 'tuple',
                'default': (0.0, 0.0, 0.0),
                'length': 3,
                'description': '中心坐标'
            },
            'Passing_People': {
                'type': 'int',
                'default': 0,
                'min': 0,
                'description': '经过人数'
            },
            'is_exit': {
                'type': 'bool',
                'default': False,
                'description': '是否为出口'
            }
        }
        self.movement_factor = 0.5  # 相邻节点移动系数
        self.max_propagation_depth = 5  # 最大传播深度
        self.visualizer = None
        self.kd_tree = None  # 添加KD树属性
        self.node_positions = None  # 存储节点位置数组
        self.node_ids = None  # 存储节点ID列表
        self.kd_tree_lock = threading.Lock()  # 添加线程锁
        self.kd_tree_update_queue = Queue()  # 添加更新队列
        self.kd_tree_update_thread = None  # 添加更新线程
        self.is_running = False  # 添加运行状态标志

        # 添加可见节点缓存相关属性
        self.visible_nodes_cache = None  # 缓存可见节点列表
        self.visible_edges_cache = None  # 缓存可见边列表
        self.node_visibility_cache = {}  # 缓存节点可见性映射
        self.floor_filters_cache = {}  # 缓存楼层过滤状态

    def load_graph(self, file_path, progress_window=None):
        """加载GraphML文件"""
        try:
            start_time = time.time()

            # 更新进度条 - 开始加载文件
            if progress_window:
                progress_window.update_progress(0, "正在加载GraphML文件...")

            # 测量文件加载时间
            self.graph = nx.read_graphml(file_path)

            if progress_window:
                progress_window.update_progress(20, "文件加载完成，正在初始化...")

            self.pos = None
            self.initial_bounds = None

            # 初始化 custom_node_attributes
            if progress_window:
                progress_window.update_progress(30, "正在初始化自定义属性...")
            self.custom_node_attributes = {}

            # 属性映射关系
            attribute_mapping = {
                'x': 'x',
                'y': 'y',
                'lc': 'lc',
                'Entity_Type': 'Entity_Type',
                'Elem_Id': 'Elem_Id',
                'Id': 'Id',
                'Capacity': 'Capacity',
                'length': 'length',
                'Connect_Entity_Id_List': 'Connect_Entity_Id_List'
            }

            if progress_window:
                progress_window.update_progress(40, "正在处理节点属性...")

            # 获取总节点数用于计算进度
            total_nodes = len(self.graph.nodes())
            nodes_processed = 0

            # 缓存节点属性并识别自定义属性
            for node in self.graph.nodes():
                node_data = dict(self.graph.nodes[node])
                for attr, value in node_data.items():
                    # 执行属性映射
                    if attr in attribute_mapping:
                        mapped_attr = attribute_mapping[attr]
                        try:
                            if mapped_attr in ['x', 'y']:
                                # 处理坐标值
                                self.graph.nodes[node][mapped_attr] = float(value)
                                # 更新 Center 属性
                                current_center = list(self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0)))
                                if mapped_attr == 'x':
                                    current_center[0] = float(value)
                                elif mapped_attr == 'y':
                                    current_center[1] = float(value)
                                self.graph.nodes[node]['Center'] = tuple(current_center)
                            elif mapped_attr == 'lc':
                                # 处理楼层值
                                self.graph.nodes[node][mapped_attr] = int(float(value))
                                # 更新 Center 的 z 坐标
                                current_center = list(self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0)))
                                current_center[2] = float(value)
                                self.graph.nodes[node]['Center'] = tuple(current_center)
                            elif mapped_attr in [ 'Capacity','length']:
                                self.graph.nodes[node][mapped_attr] = float(value)
                            elif mapped_attr in ['Id', 'Elem_Id']:
                                self.graph.nodes[node][mapped_attr] = value
                            elif mapped_attr == 'Entity_Type':
                                self.graph.nodes[node][mapped_attr] = str(value)

                                # **如果实体类型是 "建筑物出口"，设置 is_exit = 1**
                                if str(value) == "建筑物出口":
                                    self.graph.nodes[node]["is_exit"] = 1
                                    logger.info(f"节点 {node} 被标记为出口 (is_exit=1)")

                            else:
                                self.graph.nodes[node][mapped_attr] = value
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error converting value for attribute {mapped_attr}: {e}")
                            self.graph.nodes[node][mapped_attr] = value

                        continue

                    # 处理自定义属性
                    if attr not in self.base_attributes:
                        if attr not in self.custom_node_attributes:
                            attr_type = type(value).__name__
                            attr_default = value
                            self.custom_node_attributes[attr] = {
                                'type': attr_type,
                                'default': attr_default,
                                'description': f'{attr} custom attribute'
                            }
                            logger.debug(f"添加自定义属性: {attr}, 类型: {attr_type}, 默认值: {attr_default}")

                # 更新进度
                nodes_processed += 1
                if progress_window and total_nodes > 0:
                    progress = 40 + (nodes_processed / total_nodes * 30)  # 40-70%的进度
                    progress_window.update_progress(progress, f"正在处理节点属性... ({nodes_processed}/{total_nodes})")

            # 处理节点的邻接关系
            if progress_window:
                progress_window.update_progress(70, "正在处理节点邻接关系...")

            for node in self.graph.nodes():
                # 获取节点的所有邻居
                neighbors = list(self.graph.neighbors(node))
                # 更新Connect_Entity_Id_List属性
                self.graph.nodes[node]['Connect_Entity_Id_List'] = neighbors
                logger.debug(f"节点 {node} 的邻接关系已更新: {self.graph.nodes[node]['Connect_Entity_Id_List']}")

            if progress_window:
                progress_window.update_progress(80, "正在缓存节点属性...")

            # 缓存节点属性
            self._cache_node_attributes()

            if progress_window:
                progress_window.update_progress(100, "加载完成！")

            logger.info(f"成功加载图形，节点数: {len(self.graph.nodes)}, 边数: {len(self.graph.edges)}")

            return True
        except Exception as e:
            if progress_window:
                progress_window.update_progress(100, f"加载失败: {str(e)}")
            logger.error(f"加载图形失败: {str(e)}")
            return False

    def calculate_layout(self, layout_name, node_size, progress_window=None):
        """计算图形布局"""
        if not self.graph:
            logger.warning("没有图形数据，无法计算布局")
            return

        try:
            if progress_window:
                progress_window.update_progress(0, "正在初始化布局计算...")

            # 测量布局函数执行时间
            layout_start = time.time()

            # 根据布局类型选择不同的参数
            if layout_name == "fruchterman_reingold":
                if progress_window:
                    progress_window.update_progress(20, "正在计算Fruchterman-Reingold布局...")
                    self.pos = nx.fruchterman_reingold_layout(
                        self.graph,
                        iterations=30,  # 减少迭代次数以提高速度
                        k=0.1
                    )
            elif layout_name == "floor_based":
                self.floor_based_layout(progress_window)
            # elif layout_name == "grid_crossing":
            #     self.pos = self._grid_crossing_layout(node_size, progress_window)
            else:
                if progress_window:
                    progress_window.update_progress(20, f"正在计算{layout_name}布局...")
                layout_func = getattr(nx, f"{layout_name}_layout")
                self.pos = layout_func(self.graph)

            layout_end = time.time()
            layout_duration = layout_end - layout_start
            logger.info(f"基础布局计算时间: {layout_duration:.4f} 秒")

            if progress_window:
                progress_window.update_progress(90, "正在计算边界...")

            self._calculate_bounds()
            bounds_end = time.time()

            total_duration = bounds_end - layout_start
            logger.info(f"使用 {layout_name} 布局算法完成布局计算，总时间: {total_duration:.4f} 秒")

            if progress_window:
                progress_window.update_progress(100, "布局计算完成！")

        except Exception as e:
            logger.error(f"计算布局失败: {str(e)}")
            if progress_window:
                progress_window.update_progress(100, f"布局计算失败: {str(e)}")

    def _calculate_bounds(self):
        """计算图形边界，增加边距以确保节点不会太靠近边缘"""
        if not self.pos:
            return

        # 从三维坐标中提取x和y坐标
        x_coords = []
        y_coords = []
        for pos in self.pos.values():
            if len(pos) == 3:
                x_coords.append(pos[0])
                y_coords.append(pos[1])
            else:
                x_coords.append(pos[0])
                y_coords.append(pos[1])

        # 计算边界
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 增加边距
        margin = 0.1  # 10%的边距
        x_margin = (x_max - x_min) * margin
        y_margin = (y_max - y_min) * margin

        self.initial_bounds = (
            x_min - x_margin,
            x_max + x_margin,
            y_min - y_margin,
            y_max + y_margin
        )

    def _cache_node_attributes(self):
        """缓存节点的属性信息"""
        if not self.graph:
            return

        # 映射关系
        attribute_mapping = {
            'id': 'Elem_Id',
            'y': 'Center_y',
            'x': 'Center_x',
            'lc': 'Center_z',
            'glgx': 'Connect_Entity_Id_List',
            'pointtype': 'Entity_Type',
            'area': 'Capacity',
        }

        # 更新节点的实际值，但保持属性定义不变
        for node in self.graph.nodes():
            node_data = dict(self.graph.nodes[node])

            # 确保所有预定义的属性都有值，并处理映射
            for attr, config in self.base_attributes.items():
                if attr not in node_data:
                    # 如果节点没有这个属性，使用默认值
                    self.graph.nodes[node][attr] = config['default']
                else:
                    # 处理映射关系
                    if attr in attribute_mapping:
                        mapped_attr = attribute_mapping[attr]
                        if mapped_attr == 'Center_y':
                            self.graph.nodes[node]['Center'] = (
                                self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))[0], float(node_data[attr]),
                                self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))[2])
                        elif mapped_attr == 'Center_x':
                            self.graph.nodes[node]['Center'] = (
                                float(node_data[attr]), self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))[1],
                                self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))[2])
                        elif mapped_attr == 'Center_z':
                            self.graph.nodes[node]['Center'] = (
                                self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))[0],
                                self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))[1], float(node_data[attr]))
                        elif mapped_attr == 'Connect_Entity_Id_List':
                            self.graph.nodes[node][mapped_attr] = node_data[attr].split(',')
                        else:
                            self.graph.nodes[node][mapped_attr] = node_data[attr]

            # 确保所有自定义属性都有值
            for attr, config in self.custom_node_attributes.items():
                if attr not in node_data:
                    # 如果节点没有这个属性，使用默认值
                    self.graph.nodes[node][attr] = config['default']

    def get_node_neighbors(self, node):
        """获取节点的邻居"""
        logger.debug(f"Attempting to get neighbors for node: {node}")
        logger.debug(f"Graph exists: {self.graph is not None}")
        if self.graph:
            logger.debug(f"Node type: {type(node)}")
            logger.debug(f"Nodes in graph: {list(self.graph.nodes())[:5]}...")  # 只打印前5个节点避免日志过长
            logger.debug(f"Node in graph: {node in self.graph}")
            if node in self.graph:
                neighbors = list(self.graph.neighbors(node))
                logger.debug(f"Found {len(neighbors)} neighbors for node {node}")
                return neighbors
            else:
                logger.warning(f"Node {node} not found in graph")
        return []

    def get_node_attribute(self, node, attribute):
        """获取节点的指定属性值"""
        if self.graph is None or node not in self.graph:
            return None
        # 检查属性是否在 base_attributes 或 custom_node_attributes 中
        if attribute in self.base_attributes:
            default_value = self.base_attributes[attribute]['default']
        elif attribute in self.custom_node_attributes:
            default_value = self.custom_node_attributes[attribute]['default']
        else:
            return None
        # 返回节点的属性值或默认值
        return self.graph.nodes[node].get(attribute, default_value)

    def get_node_attributes(self, node):
        """获取节点的属性"""
        if node not in self.graph.nodes:
            return {}

        # 获取基础属性和自定义属性
        node_attrs = {}
        for attr in {**self.base_attributes, **self.custom_node_attributes}.keys():
            node_attrs[attr] = self.get_node_attribute(node, attr)

        return node_attrs

    def set_node_attribute(self, node, attribute, value):
        """设置节点的指定属性值"""
        if self.graph is None or node not in self.graph:
            logger.error(f"Cannot set node attribute: node {node} not in graph")
            return False

        if attribute not in self.base_attributes and attribute not in self.custom_node_attributes:
            logger.error(f"Invalid attribute {attribute} for node {node}")
            return False

        # 获取属性信息
        if attribute in self.base_attributes:
            attr_info = self.base_attributes[attribute]
            logger.debug(f"Using base attribute info for {attribute}: {attr_info}")
        else:
            attr_info = self.custom_node_attributes[attribute]
            logger.debug(f"Using custom attribute info for {attribute}: {attr_info}")

        try:
            # 类型检查和转换
            if attr_info['type'] == 'int':
                logger.debug(f"Converting {value} to int for attribute {attribute}")
                value = int(value)
                if 'min' in attr_info and value < attr_info['min']:
                    logger.error(f"Value {value} is below minimum {attr_info['min']} for attribute {attribute}")
                    return False
            elif attr_info['type'] == 'str':
                logger.debug(f"Converting {value} to str for attribute {attribute}")
                value = str(value)
                if 'options' in attr_info and value not in attr_info['options'] and value != '':
                    logger.error(
                        f"Invalid option '{value}' for attribute {attribute}. Valid options are: {attr_info['options']}")
                    return False
            elif attr_info['type'] == 'list':
                logger.debug(f"Converting {value} to list for attribute {attribute}")
                if isinstance(value, str):
                    value = eval(value)
                if not isinstance(value, list):
                    logger.error(f"Invalid list value for attribute {attribute}: {value}")
                    return False
            elif attr_info['type'] == 'tuple':
                logger.debug(f"Converting {value} to tuple for attribute {attribute}")
                if isinstance(value, str):
                    value = eval(value)
                if not isinstance(value, tuple):
                    logger.error(f"Invalid tuple value for attribute {attribute}: {value}")
                    return False
                if 'length' in attr_info and len(value) != attr_info['length']:
                    logger.error(
                        f"Invalid tuple length for attribute {attribute}. Expected {attr_info['length']}, got {len(value)}")
                    return False
            elif attr_info['type'] == 'bool':
                logger.debug(f"Converting {value} to bool for attribute {attribute}")
                if isinstance(value, str):
                    value = value.lower() in ['true', '1', 'yes', 'on']
                value = bool(value)

            # 设置属性值
            old_value = self.graph.nodes[node].get(attribute)
            self.graph.nodes[node][attribute] = value
            logger.info(f"Updated node {node} attribute {attribute}: {old_value} -> {value}")
            return True

        except Exception as e:
            logger.error(f"Error converting value for attribute {attribute}: {str(e)}")
            logger.error(f"Value type: {type(value)}, Value: {value}")
            return False

    def update_node_position(self, node, new_pos):
        """
        更新节点位置，并相应更新相邻节点的位置
        Args:
            node: 需要更新位置的节点
            new_pos: 新位置坐标 (x, y)
        Returns:
            bool: 更新是否成功
        """
        if node not in self.graph:
            logger.warning(f"Node {node} not found in graph")
            return False

        try:
            # 计算移动向量
            old_pos = np.array(self.pos[node])
            new_pos = np.array(new_pos)
            movement_vector = new_pos - old_pos

            # 更新主节点位置
            self.pos[node] = new_pos

            # 更新相邻节点位置（使用广度优先搜索）
            visited = {node}
            queue = [(n, 1) for n in self.graph.neighbors(node)]  # (节点, 深度)

            while queue:
                current_node, depth = queue.pop(0)
                if current_node in visited or depth > self.max_propagation_depth:
                    continue

                visited.add(current_node)

                # 计算衰减系数
                decay = self.movement_factor ** depth

                # 更新当前节点位置
                try:
                    current_pos = np.array(self.pos[current_node])
                    self.pos[current_node] = current_pos + movement_vector * decay

                    # 将未访问的相邻节点加入队列
                    if depth < self.max_propagation_depth:
                        for neighbor in self.graph.neighbors(current_node):
                            if neighbor not in visited:
                                queue.append((neighbor, depth + 1))
                except Exception as e:
                    logger.error(f"Error updating position for node {current_node}: {e}")
                    continue

            return True

        except Exception as e:
            logger.error(f"Error updating node position: {e}")
            return False

    def get_node_info(self, node):
        """获取节点的所有属性信息"""
        if self.graph is None or node not in self.graph:
            return None

        info = {}
        # 获取预定义的属性和自定义属性
        for attr, config in {**self.base_attributes, **self.custom_node_attributes}.items():
            info[attr] = self.graph.nodes[node].get(attr, config['default'])
        return info

    def set_node_info(self, node, info):
        """
        :param node: 结点id，类型为str
        :param info:
        :return:
        """
        """设置节点的多个属性"""
        if self.graph is None or node not in self.graph:
            logger.error(f"Cannot set node info: node {node} not in graph")
            return False

        success = True
        logger.info(f"Setting node info for node {node}: {info}")

        # 处理坐标属性
        if any(coord in info for coord in ['x', 'y', 'lc']):
            current_center = self.graph.nodes[node].get('Center', (0.0, 0.0, 0.0))
            x = info.get('x', current_center[0])
            y = info.get('y', current_center[1])
            z = info.get('lc', current_center[2])
            self.graph.nodes[node]['Center'] = (float(x), float(y), float(z))
            # 同时更新单独的坐标属性
            self.graph.nodes[node]['x'] = float(x)
            self.graph.nodes[node]['y'] = float(y)
            self.graph.nodes[node]['lc'] = float(z)
            logger.debug(f"Updated coordinates for node {node}: ({x}, {y}, {z})")

        # 处理其他属性
        for attr, value in info.items():
            if attr not in ['x', 'y', 'lc']:  # 跳过已处理的坐标属性
                if attr in self.base_attributes or attr in self.custom_node_attributes:
                    if not self.set_node_attribute(node, attr, value):
                        logger.error(f"Failed to set attribute {attr} for node {node}")
                        success = False
                    else:
                        logger.debug(f"Successfully set attribute {attr}={value} for node {node}")

        if success:
            logger.info(f"Successfully updated all attributes for node {node}")
        else:
            logger.warning(f"Some attributes failed to update for node {node}")
        return success

    def clear(self):
        """清除所有数据"""
        self.graph = None
        self.pos = None
        self.initial_bounds = None
        self.base_attributes.clear()
        self.custom_node_attributes.clear()
        self.stop_kd_tree_update_thread()  # 停止KD树更新线程

        # 清除可见节点缓存
        self.visible_nodes_cache = None
        self.visible_edges_cache = None
        self.node_visibility_cache.clear()
        self.floor_filters_cache.clear()

    def save_to_graphml(self, filename):
        """保存图形到GraphML文件"""
        if not self.graph:
            return False

        # 创建新的GraphML对象，使用无向图
        graphml = nx.Graph()

        # 复制所有节点和它们的属性
        for node in self.graph.nodes():
            node_attrs = {}
            for attr in self.base_attributes:
                # 跳过 Connect_Entity_Id_List 属性
                if attr == 'Connect_Entity_Id_List':
                    continue

                value = self.get_node_attribute(node, attr)
                if attr == 'Center':
                    try:
                        # x, y 存储为 double
                        node_attrs['x'] = float(value[0])
                        node_attrs['y'] = float(value[1])
                        # lc 存储为 long，先转换为float再转为int
                        lc_value = float(value[2])
                        node_attrs['lc'] = int(lc_value)
                        logger.debug(f"Converting lc value for node {node}: {value[2]} -> {node_attrs['lc']}")
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error converting coordinates for node {node}: {e}")
                        continue
                    continue  # 跳过原Center属性
                node_attrs[attr] = value

            # 添加自定义属性
            for attr, info in self.custom_node_attributes.items():
                value = self.graph.nodes[node].get(attr, info['default'])
                if isinstance(value, tuple):
                    value = str(value)
                node_attrs[attr] = value

            graphml.add_node(node, **node_attrs)

        # 复制所有边，保持无向图的特性
        for edge in self.graph.edges():
            graphml.add_edge(edge[0], edge[1])

        # 写入文件
        try:
            nx.write_graphml(graphml, filename)
            return True
        except Exception as e:
            logger.error(f"保存文件时出错: {e}")
            return False

    def add_custom_node_attribute(self, name, attr_type, default, list_items=None):
        """添加自定义节点属性"""
        # 检查是否存在图信息
        if not self.graph:
            return False
        if name in self.base_attributes or name in self.custom_node_attributes:
            # 属性已存在
            return False
        if attr_type == "自定义选项":
            attr_type = str
        self.custom_node_attributes[name] = {
            'type': attr_type,
            'default': default,
            'description': f'自定义属性 {name}',
            'options': list_items if list_items is not None else None  # 存储可选值
        }
        # 为所有节点添加该属性
        for node in self.graph.nodes():
            if name not in self.graph.nodes[node]:
                self.graph.nodes[node][name] = default
        logger.info(f"成功添加自定义属性: {name}")
        return True

    def calculate_edge_weight(self, edge):
        """计算边的权重（欧几里得距离）
        Args:
            edge: 边的元组 (u, v)
        Returns:
            float: 边的权重，如果节点不连通则返回无穷大
        """
        if not self.graph:
            return float('inf')

        u, v = edge
        if u not in self.graph.nodes or v not in self.graph.nodes:
            return float('inf')

        # 获取节点的Center属性
        u_center = self.graph.nodes[u].get('Center', None)
        v_center = self.graph.nodes[v].get('Center', None)

        if not u_center or not v_center:
            return float('inf')

        # 计算欧几里得距离
        try:
            # Center属性是一个元组 (x, y, z)
            u_x, u_y, u_z = u_center
            v_x, v_y, v_z = v_center

            # 计算欧几里得距离
            distance = ((u_x - v_x) ** 2 + (u_y - v_y) ** 2 + (u_z - v_z) ** 2) ** 0.5
            return distance
        except (ValueError, TypeError):
            return float('inf')

    def calculate_shortest_path(self, start_node, end_node):
        """计算两点间的最短路径
        Args:
            start_node: 起点节点ID
            end_node: 终点节点ID
        Returns:
            tuple: (路径列表, 路径长度, 是否成功)
        """
        if not self.graph:
            logger.warning("没有图形数据，无法计算最短路径")
            return [], 0, False

        if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
            logger.warning(f"节点 {start_node} 或 {end_node} 不存在")
            return [], 0, False

        try:
            # 计算所有边的权重
            edge_weights = {
                edge: self.calculate_edge_weight(edge)
                for edge in self.graph.edges()
            }

            # 使用NetworkX计算最短路径
            path = nx.shortest_path(self.graph, start_node, end_node,
                                    weight=lambda u, v, d: edge_weights.get((u, v), edge_weights.get((v, u), float('inf'))))
            length = nx.shortest_path_length(self.graph, start_node, end_node,
                                             weight=lambda u, v, d: edge_weights.get((u, v), edge_weights.get((v, u), float('inf'))))

            # logger.info(f"找到从 {start_node} 到 {end_node} 的最短路径，长度为 {length:.2f}")
            return path, length, True

        except nx.NetworkXNoPath:
            logger.warning(f"从 {start_node} 到 {end_node} 没有可行路径")
            return [], 0, False
        except Exception as e:
            logger.error(f"计算最短路径时出错: {str(e)}")
            return [], 0, False

    def _grid_crossing_layout(self, base_node_size: float, progress_window=None) -> Dict[
        str, Tuple[float, float, float]]:
        """计算网格交叉布局，按楼层分别布局
        
        Args:
            base_node_size: 基础节点大小，用于计算网格间距
            progress_window: 进度窗口对象
            
        Returns:
            节点位置字典
        """
        try:
            if progress_window:
                progress_window.update_progress(10, "正在按楼层分组...")

            # 按楼层分组节点
            floor_nodes = {}
            for node in self.graph.nodes():
                try:
                    floor = self.graph.nodes[node].get('lc', 0)
                    if floor not in floor_nodes:
                        floor_nodes[floor] = []
                    floor_nodes[floor].append(node)
                except Exception as e:
                    logger.error(f"处理节点 {node} 时出错: {str(e)}")
                    floor_nodes[0] = floor_nodes.get(0, []) + [node]

            logger.info(f"按楼层分组完成，共 {len(floor_nodes)} 层")
            for floor, nodes in floor_nodes.items():
                logger.info(f"第 {floor} 层有 {len(nodes)} 个节点")

            if progress_window:
                progress_window.update_progress(30, "正在计算每层网格布局...")

            # 对每层分别计算网格布局
            floor_layouts = {}
            for floor, nodes in floor_nodes.items():
                try:
                    # 创建子图
                    subgraph = self.graph.subgraph(nodes)
                    logger.info(f"正在计算第 {floor} 层布局，节点数: {len(nodes)}")

                    # 为每层创建网格布局
                    grid_layout = GridCrossingLayout(subgraph, grid_spacing=base_node_size * 2)
                    layout = grid_layout.compute_layout()

                    # 确保布局包含子图中的所有节点
                    missing_nodes = set(subgraph.nodes()) - set(layout.keys())
                    if missing_nodes:
                        logger.warning(
                            f"第 {floor} 层有 {len(missing_nodes)} 个节点未获得网格布局位置，使用圆形布局补充")
                        # 为缺失的节点创建圆形布局
                        missing_subgraph = subgraph.subgraph(list(missing_nodes))
                        circle_layout = nx.circular_layout(missing_subgraph)
                        # 合并布局
                        layout.update(circle_layout)

                    floor_layouts[floor] = layout
                    logger.info(f"第 {floor} 层布局计算完成，共处理 {len(layout)} 个节点")

                except Exception as e:
                    logger.error(f"计算第 {floor} 层布局时出错: {str(e)}")
                    # 如果网格布局失败，使用简单的圆形布局作为后备
                    floor_layouts[floor] = nx.circular_layout(subgraph)

            if progress_window:
                progress_window.update_progress(60, "正在组合布局...")

            # 计算每层的中心位置
            floors = sorted(floor_nodes.keys())
            n_floors = len(floors)
            floor_spacing = 3000  # 楼层间距

            # 初始化最终位置字典
            final_positions = {}

            # 计算每层的偏移量并应用到节点位置
            for i, floor in enumerate(floors):
                try:
                    # 计算该层的偏移量
                    offset_x = (i - n_floors / 2) * floor_spacing
                    offset_y = 0

                    # 应用偏移量到该层的所有节点
                    for node in floor_nodes[floor]:
                        # 确保节点在布局中
                        if node not in floor_layouts[floor]:
                            logger.warning(f"节点 {node} 在布局数据中未找到，创建默认位置")
                            # 在当前层的边缘创建位置
                            angle = len(final_positions) * (2 * np.pi / len(self.graph.nodes()))
                            radius = floor_spacing / 4
                            x = radius * np.cos(angle)
                            y = radius * np.sin(angle)
                            floor_layouts[floor][node] = np.array([x, y])

                        pos = floor_layouts[floor][node]
                        # 确保pos是三维坐标
                        if isinstance(pos, np.ndarray):
                            pos = pos.tolist()
                        if len(pos) == 2:
                            x, y = pos
                            z = float(floor)
                        else:
                            x, y, z = pos

                        # 应用偏移并保存位置
                        final_positions[node] = (
                            float(x) + offset_x,
                            float(y) + offset_y,
                            float(z)
                        )
                except Exception as e:
                    logger.error(f"处理第 {floor} 层节点位置时出错: {str(e)}")

            if progress_window:
                progress_window.update_progress(80, "正在优化跨层边...")

            # 优化跨层边的布局
            for edge in self.graph.edges():
                try:
                    n1, n2 = edge
                    if n1 not in final_positions or n2 not in final_positions:
                        logger.warning(f"边 {edge} 的节点位置未找到，跳过优化")
                        continue

                    floor1 = self.graph.nodes[n1].get('lc', 0)
                    floor2 = self.graph.nodes[n2].get('lc', 0)
                    if floor1 != floor2:
                        # 对于跨层边，稍微调整节点位置以减少交叉
                        mid_x = (final_positions[n1][0] + final_positions[n2][0]) / 2
                        x1, y1, z1 = final_positions[n1]
                        x2, y2, z2 = final_positions[n2]

                        # 调整x坐标，使跨层的节点更靠近
                        final_positions[n1] = (x1 * 0.9 + mid_x * 0.1, y1, z1)
                        final_positions[n2] = (x2 * 0.9 + mid_x * 0.1, y2, z2)
                except Exception as e:
                    logger.error(f"优化边 {edge} 时出错: {str(e)}")

            # 确保所有节点都有位置
            for node in self.graph.nodes():
                if node not in final_positions:
                    logger.warning(f"节点 {node} 没有位置，使用默认位置")
                    floor = self.graph.nodes[node].get('lc', 0)
                    # 在对应楼层的边缘创建位置
                    angle = len(final_positions) * (2 * np.pi / len(self.graph.nodes()))
                    radius = floor_spacing / 4
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    final_positions[node] = (x, y, float(floor))

            if progress_window:
                progress_window.update_progress(100, "布局计算完成")

            logger.info(f"布局计算完成，共处理 {len(final_positions)} 个节点")
            return final_positions

        except Exception as e:
            logger.error(f"网格布局计算失败: {str(e)}")
            # 如果出错，返回一个简单的圆形布局
            circle_layout = nx.circular_layout(self.graph)
            # 转换为3D坐标
            return {node: (x, y, float(self.graph.nodes[node].get('lc', 0)))
                    for node, (x, y) in circle_layout.items()}

    def floor_based_layout(self, progress_window):
        if progress_window:
            progress_window.update_progress(20, "正在按楼层分组...")

        # 按楼层分组节点
        floor_nodes = {}
        for node in self.graph.nodes():
            floor = self.graph.nodes[node].get('lc', 0)
            if floor not in floor_nodes:
                floor_nodes[floor] = []
            floor_nodes[floor].append(node)

        if progress_window:
            progress_window.update_progress(40, "正在计算每层布局...")

        # 对每层分别计算布局
        floor_layouts = {}
        floor_subgraphs = {}
        for floor, nodes in floor_nodes.items():
            # 创建子图
            subgraph = self.graph.subgraph(nodes)
            floor_subgraphs[floor] = subgraph

            # 计算布局
            floor_layouts[floor] = nx.fruchterman_reingold_layout(
                subgraph,
                k=1 / np.sqrt(len(subgraph)),
                iterations=50  # 减少迭代次数以提高速度
            )

        if progress_window:
            progress_window.update_progress(60, "正在组合布局...")

        # 计算每层的中心位置
        floors = sorted(floor_nodes.keys())
        n_floors = len(floors)
        spacing = 30000  # 楼层间距

        # 初始化最终位置字典
        self.pos = {}

        # 计算每层的偏移量并应用到节点位置
        for i, floor in enumerate(floors):
            # 计算该层的偏移量
            offset_x = (i - n_floors / 2) * spacing
            offset_y = 0

            # 应用偏移量到该层的所有节点
            for node in floor_nodes[floor]:
                if node in floor_layouts[floor]:
                    pos = floor_layouts[floor][node]
                    self.pos[node] = np.array([
                        pos[0] * spacing / 2 + offset_x,
                        pos[1] * spacing / 2 + offset_y
                    ])

        if progress_window:
            progress_window.update_progress(80, "正在优化跨层边...")

        # 优化跨层边的布局
        for edge in self.graph.edges():
            n1, n2 = edge
            floor1 = self.graph.nodes[n1].get('lc', 0)
            floor2 = self.graph.nodes[n2].get('lc', 0)
            if floor1 != floor2:
                # 对于跨层边，稍微调整节点位置以减少交叉
                mid_x = (self.pos[n1][0] + self.pos[n2][0]) / 2
                self.pos[n1][0] = self.pos[n1][0] * 0.9 + mid_x * 0.1
                self.pos[n2][0] = self.pos[n2][0] * 0.9 + mid_x * 0.1

    def start_kd_tree_update_thread(self):
        """启动KD树更新线程"""
        if self.kd_tree_update_thread is None or not self.kd_tree_update_thread.is_alive():
            self.is_running = True
            self.kd_tree_update_thread = threading.Thread(target=self._kd_tree_update_worker)
            self.kd_tree_update_thread.daemon = True  # 设置为守护线程
            self.kd_tree_update_thread.start()
            logger.info("KD树更新线程已启动")

    def stop_kd_tree_update_thread(self):
        """停止KD树更新线程"""
        self.is_running = False
        if self.kd_tree_update_thread and self.kd_tree_update_thread.is_alive():
            self.kd_tree_update_thread.join()
            logger.info("KD树更新线程已停止")

    def _kd_tree_update_worker(self):
        """KD树更新线程的工作函数"""
        logger.info("KD树更新线程已启动")
        while self.is_running:
            try:
                # 从队列中获取更新请求，设置超时时间为0.1秒
                update_request = self.kd_tree_update_queue.get(timeout=0.1)
                logger.debug(f"收到KD树更新请求: {update_request}")

                # 获取线程锁
                with self.kd_tree_lock:
                    # 更新KD树
                    self._update_kd_tree_internal()
                    logger.debug("KD树更新完成")

                # 标记任务完成
                self.kd_tree_update_queue.task_done()

            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                logger.error(f"KD树更新线程发生错误: {e}")
                # 发生错误时等待一段时间再继续
                time.sleep(0.1)

        logger.info("KD树更新线程已停止")

    def _update_kd_tree_internal(self):
        """内部更新KD树的逻辑"""
        if self.pos:
            # 提取节点位置和ID
            self.node_ids = list(self.pos.keys())
            self.node_positions = np.array([self.pos[node_id] for node_id in self.node_ids])

            # 创建KD树
            self.kd_tree = cKDTree(self.node_positions)
            logger.debug("KD树已更新")
        else:
            self.kd_tree = None
            self.node_positions = None
            self.node_ids = None
            logger.debug("KD树已清空")

    def update_kd_tree(self):
        """异步更新KD树"""
        if not hasattr(self, 'kd_tree_update_queue'):
            # 如果线程未启动，先启动线程
            self.start_kd_tree_update_thread()

        # 将更新请求添加到队列
        self.kd_tree_update_queue.put(True)

    def get_nearest_node(self, x, y, max_distance=None):
        """使用KD树查找最近节点
        Args:
            x: x坐标
            y: y坐标
            max_distance: 最大搜索距离
        Returns:
            tuple: (最近节点ID, 距离) 或 (None, None)
        """
        with self.kd_tree_lock:  # 使用线程锁保护KD树访问
            if self.kd_tree is None:
                return None, None

            # 计算查询点
            query_point = np.array([x, y])

            # 如果指定了最大距离，使用query_ball_point
            if max_distance is not None:
                indices = self.kd_tree.query_ball_point(query_point, max_distance)
                if not indices:
                    return None, None
                # 找到最近的点
                distances = np.linalg.norm(self.node_positions[indices] - query_point, axis=1)
                min_idx = np.argmin(distances)
                return self.node_ids[indices[min_idx]], distances[min_idx]

            # 否则使用query
            distance, index = self.kd_tree.query(query_point)
            return self.node_ids[index], distance

    def update_visible_nodes_cache(self, floor_filters):
        """更新可见节点缓存
        Args:
            floor_filters: 楼层过滤状态字典
        """

        # 更新楼层过滤缓存
        self.floor_filters_cache = floor_filters.copy()

        # 更新节点可见性映射
        self.node_visibility_cache = {}
        for node in self.graph.nodes():
            node_floor = self.graph.nodes[node].get('lc')
            if node_floor is not None:
                try:
                    # 确保楼层值是数字
                    node_floor_num = int(float(node_floor))
                    # 检查楼层是否在过滤器中
                    if node_floor_num in floor_filters:
                        # 获取过滤状态
                        filter_var = floor_filters[node_floor_num]
                        if hasattr(filter_var, 'get'):
                            # 如果是tkinter变量，使用get()方法
                            self.node_visibility_cache[node] = bool(filter_var.get())
                        else:
                            # 如果是普通布尔值
                            self.node_visibility_cache[node] = bool(filter_var)
                    else:
                        # 楼层不在过滤器中，默认不可见
                        self.node_visibility_cache[node] = False
                except (ValueError, TypeError) as e:
                    logger.warning(f"节点 {node} 的楼层值 '{node_floor}' 无效: {e}")
                    # 楼层值无效时，默认不可见
                    self.node_visibility_cache[node] = False
            else:
                # 没有楼层值时，默认不可见
                logger.warning(f"节点 {node} 没有楼层值")
                self.node_visibility_cache[node] = False

        # 更新可见节点列表
        self.visible_nodes_cache = [
            node for node in self.graph.nodes()
            if self.node_visibility_cache[node]
        ]

        # 更新可见边列表
        self.visible_edges_cache = [
            edge for edge in self.graph.edges()
            if self.node_visibility_cache[edge[0]] and self.node_visibility_cache[edge[1]]
        ]

        logger.info(
            f"更新可见节点缓存: {len(self.visible_nodes_cache)} 个可见节点, {len(self.visible_edges_cache)} 条可见边")
        logger.debug(f"楼层过滤状态: {floor_filters}")

    def get_visible_nodes(self):
        """获取可见节点列表"""
        return self.visible_nodes_cache if self.visible_nodes_cache is not None else []

    def get_visible_edges(self):
        """获取可见边列表"""
        return self.visible_edges_cache if self.visible_edges_cache is not None else []

    def is_node_visible(self, node):
        """检查节点是否可见"""
        return self.node_visibility_cache.get(node, True)
