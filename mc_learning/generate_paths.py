from tkinter import messagebox, ttk
import tkinter as tk
import networkx as nx
import numpy as np
import random
from typing import List, Tuple, Optional, Set, Any
import math
import pandas as pd
from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()


def select_emergency_areas(graph, num_emergencies: int = 3) -> List[str]:
    """选择紧急区域
    Args:
        num_emergencies: 需要选择的紧急区域数量
    Returns:
        List[str]: 选中的紧急区域列表
    """
    if not graph:
        logger.warning("图为空，无法选择紧急区域")
        return []

    # 获取所有非出口节点
    areas = [
        node for node in graph.nodes
        if not graph.nodes[node].get('is_exit', False)
    ]

    if not areas:
        logger.warning("没有可用的非出口节点")
        return []

    # 计算安全等级权重
    weights = []
    for area in areas:
        try:
            safety_level = float(graph.nodes[area].get('safety_level', 1))
            weight = 1 / (safety_level + 1e-5)
        except (ValueError, TypeError):
            weight = 1.0
        weights.append(weight)

    # 选择紧急区域
    num_to_select = min(num_emergencies, len(areas))
    selected = []
    while len(selected) < num_to_select and areas:
        try:
            idx = possibility_select(range(len(areas)), weights)
            selected.append(areas.pop(idx))
            weights.pop(idx)
        except ValueError as e:
            logger.error(f"选择紧急区域时出错: {e}")
            break

    return selected


def possibility_select(options, weights):
    """带权重的随机选择"""
    logger.debug(f"带权重选择: options={options}, weights={weights}")

    if len(options) != len(weights):
        logger.error(f"选项列表 ({options}) 与权重列表 ({weights}) 长度不匹配")
        logger.error(f"选项数量: {len(options)}, 权重数量: {len(weights)}")

    # 检查 weights 是否为空
    if not weights:
        raise ValueError("权重不能为空")

    # 处理负数权重
    if any(w < 0 for w in weights):
        logger.warning("权重包含负数，将进行修正")
        min_weight = min(weights)  # 找到最小权重值
        if min_weight < 0:
            # 将所有权重加上最小权重的绝对值，确保非负
            offset = abs(min_weight)
            weights = [w + offset for w in weights]
            logger.debug(f"修正后的权重: {weights}")

    # 检查修正后的权重是否有效
    if any(w < 0 for w in weights):
        logger.error(f"修正后权重仍包含负数: weights={weights}")
        raise ValueError("权重修正失败，仍包含负数")

    total = sum(weights)
    if total <= 0:
        logger.error(f"权重总和无效: total={total}")
        raise ValueError("权重总和必须为正数")

    # 计算概率
    probabilities = [w / total for w in weights]
    logger.debug(f"计算概率: probabilities={probabilities}")

    # 检查概率是否有效
    if any(p < 0 or p > 1 for p in probabilities):
        logger.error(f"概率无效: probabilities={probabilities}")
        raise ValueError("概率必须在 [0, 1] 范围内")

    # 确保概率总和为 1（允许浮点数误差）
    if not np.isclose(sum(probabilities), 1.0, atol=1e-8):
        logger.error(f"概率总和不为 1: sum={sum(probabilities)}")
        raise ValueError("概率总和必须为 1")

    # 执行随机选择
    try:
        selected = np.random.choice(options, p=probabilities)
        logger.debug(f"选择结果: {selected}")
        return selected
    except ValueError as e:
        logger.error(f"随机选择失败: {e}")
        logger.error(f"选项: {options}")
        logger.error(f"概率: {probabilities}")
        raise ValueError(f"随机选择失败: {e}")


def set_emergency_area(graph, seed=42):
    """随机选择属性为'室内'的实体，1-50个，设置为紧急区域，并传播安全等级"""
    # 固定随机数种子
    random.seed(seed)

    # 安全等级列表，从高到低
    safety_levels = ['Very Safe', 'Safe', 'Normal', 'Dangerous', 'Very Dangerous']

    entites = graph.nodes(data=True) if graph else []
    # 获取所有属性为'室内'的实体
    indoor_entities = [
        node for node, data in entites
        if data.get('Entity_Type') == '室内'
    ]
    total_entities = len(indoor_entities)

    if total_entities == 0:
        # logger.warning("没有属性为'室内'的实体")
        return

    # 随机选择 1-50 个紧急区域
    num_emergencies = random.randint(1, min(50, total_entities))  # 确保不超过总实体数
    # logger.debug(f"总室内实体数: {total_entities}, 需要设置的紧急区域数: {num_emergencies}")

    # 随机选择实体
    emergency_entities = random.sample(indoor_entities, num_emergencies)
    # logger.debug(f"选择的紧急区域实体: {emergency_entities}")

    # 设置初始紧急区域的安全等级
    for entity in emergency_entities:
        safety_level = random.choice(['Dangerous', 'Very Dangerous'])
        graph.nodes[entity]['Safety_Level'] = safety_level

    # 传播安全等级
    propagate_safety_levels(graph, emergency_entities, safety_levels)

    # 将所有未传播到的节点设置为 'Very Safe'
    for node in graph.nodes:
        if 'Safety_Level' not in graph.nodes[node]:
            graph.nodes[node]['Safety_Level'] = 'Very Safe'


def propagate_safety_levels(graph, start_nodes, safety_levels):
    """传播安全等级
    Args:
        start_nodes: 初始紧急区域节点列表
        safety_levels: 安全等级列表，从高到低
    """
    from collections import deque

    # 初始化队列和访问记录
    queue = deque()
    visited = set()

    # 将初始节点加入队列
    for node in start_nodes:
        queue.append((node, 0))  # (节点, 传播距离)
        visited.add(node)

    while queue:
        current_node, distance = queue.popleft()

        # 获取当前节点的安全等级
        current_level = graph.nodes[current_node].get('Safety_Level')
        if current_level not in safety_levels:
            continue

        # 计算传播后的安全等级
        level_index = safety_levels.index(current_level)
        new_level_index = min(level_index + distance, len(safety_levels) - 1)
        new_level = safety_levels[new_level_index]

        # 设置当前节点的安全等级
        graph.nodes[current_node]['Safety_Level'] = new_level

        # 如果传播到 'Very Safe'，停止传播
        if new_level == 'Very Safe':
            continue

        # 将相邻节点加入队列
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))


def manhattan_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """计算曼哈顿距离
    Args:
        pos1: 第一个点的坐标 (x1, y1, z1)
        pos2: 第二个点的坐标 (x2, y2, z2)
    Returns:
        float: 曼哈顿距离
    """
    for pos in [pos1, pos2]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 3:
            raise ValueError(f"坐标格式错误，应为长度为3的元组或列表：{pos}")
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])


def euclidean_distance(pos1: Tuple[float, float, float], pos2: Tuple[float, float, float]) -> float:
    """计算欧式距离
    Args:
        pos1: 第一个点的坐标 (x1, y1, z1)
        pos2: 第二个点的坐标 (x2, y2, z2)
    Returns:
        float: 欧式距离
    """
    for pos in [pos1, pos2]:
        if not isinstance(pos, (tuple, list)) or len(pos) != 3:
            raise ValueError(f"坐标格式错误，应为长度为3的元组或列表：{pos}")
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2 + (pos1[2] - pos2[2]) ** 2)


class DataSetGenerate:
    def __init__(self, output_file_14, path_output_file, output_file, grapml_file):
        self.output_file_14 = output_file_14
        self.path_output_file = path_output_file
        self.output_file = output_file
        self.dialog = None
        # self.floor_height = 3.5
        self.graph = nx.read_graphml(grapml_file)
        for node, data in self.graph.nodes(data=True):
            data['Center'] = (float(data['x']), float(data['y']), float(data['lc']))

        self.entities = self.graph.nodes(data=True) if self.graph else []  # 实体
        self.topology = self.graph.edges() if self.graph else []  # 连接关系
        self.used_areas: Set[str] = set()  # 保证起点尽量选择那些没做为路径的
        self.MAX_PATH_LENGTH = 100  # 最大路径长度限制
        self.MAX_ITERATIONS = 1000  # 最大迭代次数

    def parse_center(self, center: Any) -> Optional[tuple[float, ...]]:
        """解析 Center 属性为标准格式
        Args:
            center: Center 属性值
        Returns:
            Tuple[float, float, float] 或 None
        """
        try:
            if isinstance(center, str):
                # 处理字符串格式 "(x,y,z)" 或 "x,y,z"
                center = center.strip("() ")
                return tuple(map(float, center.split(',')))
            elif isinstance(center, (tuple, list)) and len(center) >= 3:
                # 处理元组或列表格式
                return tuple(map(float, center[:3]))
            else:
                logger.error(f"无效的 Center 格式: {center}")
                return None
        except (ValueError, TypeError) as e:
            logger.error(f"解析 Center 属性失败: {e}")
            return None

    def get_node_center(self, node: str) -> Optional[Tuple[float, float, float]]:
        """获取节点的 Center 坐标
        Args:
            node: 节点ID
        Returns:
            Tuple[float, float, float] 或 None
        """
        if not isinstance(node, str):
            node = str(node)
        if not self.graph or node not in self.graph.nodes:
            return None
        # center = self.graph.nodes[node].get('Center')
        center = (float(self.graph.nodes[node]['x']),
                  float(self.graph.nodes[node]['y']),
                  float(self.graph.nodes[node]['lc']))

        return center

    def select_exit(self, start, candidates, p=0.5):
        """选择出口节点，最近的出口节点权重更高
        Args:
            start: 起点节点 ID
            candidates: 候选节点列表
            p: 最近出口节点的权重比例，默认 0.5
        Returns:
            str: 选择的节点 ID
        """
        n = len(candidates)
        start_pos = self.get_node_center(start)
        nearest_exit = self.get_nearest_exit(start, start_pos)

        # 初始化权重字典
        weights = {}

        for candidate in candidates:
            if candidate == nearest_exit:
                weights[candidate] = p + (1 - p) / n  # 最近的实体
            else:
                weights[candidate] = (1 - p) / n  # 其他实体

        # 调用 possibility_select 方法进行选择
        selected = possibility_select(candidates, list(weights.values()))
        return selected

    def generate_path(self, emergency_areas, specified_start=None) -> Optional[Tuple[List[str], float, float, int]]:
        """生成一条路径
        Args:
            emergency_areas: 紧急区域列表
            specified_start: 指定的起点，如果为None则随机选择
        Returns:
            Optional[Tuple[List[str], float, int]]: (路径, 路径长度, 迭代次数) 或 None
        """
        if not self.graph:
            logger.error("图为空，无法生成路径")
            return None

        # 选择或使用指定的起点
        if specified_start:
            if specified_start not in self.graph.nodes:
                logger.error(f"指定的起点 {specified_start} 不存在")
                return None
            start_area = specified_start
        else:
            start_area = self.select_start_area()
            if not start_area:
                logger.error("无法选择起点")
                return None

        # 选择出口
        exit_candidates = [
            node for node, data in self.graph.nodes(data=True)
            if data.get('is_exit', False)
        ]
        if not exit_candidates:
            logger.error("没有找到任何出口节点")
            return None

        try:
            exit = self.select_exit(start_area, exit_candidates)
        except Exception as e:
            logger.error(f"选择出口时出错: {e}")
            return None

        path = [start_area]
        current = start_area
        previous_step = None
        iterations = 0
        path_length = 0.0  # 初始化路径长度
        eu_path_length = 0.0  # 欧氏距离长度

        visited_nodes = {start_area}

        while not self.is_exit_node(current) and iterations < self.MAX_ITERATIONS:
            iterations += 1
            if len(path) > self.MAX_PATH_LENGTH:
                logger.warning(f"路径长度超过限制 ({self.MAX_PATH_LENGTH})")
                return None

            all_candidates = self.get_candidate_entities(current)
            if not all_candidates:
                logger.warning(f"节点 {current} 没有可用的邻居节点")
                return None

            # 排除已访问的节点
            filter_visited_candidates = [c for c in all_candidates if c not in visited_nodes]
            if not filter_visited_candidates:
                if len(path) <= 1:
                    logger.warning("无法生成有效路径，起点没有可用的候选节点")
                    return None
                path.pop()  # 回退
                current = path[-1]
                iterations -= 1
                continue

            # 获取当前节点
            current_pos = self.get_node_center(current)
            if not current_pos:
                logger.error(f"无法获取节点 {current} 的坐标")
                return None

            # **计算候选节点的权重**
            weights = []
            # 使用新的筛选函数过滤候选节点
            # logger.info(f"当前节点{current}的候选节点为{candidates}")
            filter_candidates = self.filter_candidates(filter_visited_candidates, current, previous_step)
            copy_candidates = filter_candidates
            for candidate in copy_candidates:
                if self.is_exit_node(candidate):
                    next_step = candidate
                    # 更新路径长度
                    next_pos = self.get_node_center(next_step)
                    if current_pos and next_pos:
                        step_length = manhattan_distance(current_pos, next_pos)
                        eu_step_length = euclidean_distance(current_pos, next_pos)
                        path_length += step_length
                        eu_path_length += eu_step_length

                    path.append(next_step)
                    return path, path_length, eu_path_length, iterations

                candidate_pos = self.get_node_center(candidate)
                if not candidate_pos:
                    logger.error(f"无法获取候选节点 {candidate} 的坐标")
                    continue

                if int(candidate_pos[2]) > int(current_pos[2]):
                    copy_candidates.remove(candidate)
                    continue
            candidates = filter_candidates
            for candidate in candidates:
                try:
                    score = self.calculate_direction_score(current, candidate, self.get_node_center(exit),
                                                           emergency_areas)
                    if score is not None:
                        # logger.info(f"候选节点{candidate}的方向分数为{score}")
                        weights.append(score)
                    else:
                        logger.warning(f"候选节点{candidate}的方向分数计算失败")
                        continue
                except Exception as e:
                    logger.error(f"计算方向分数时出错: {e}")
                    continue

            # logger.info(f"处理后当前节点{current}的候选节点为{candidates}")
            if not candidates or not weights:
                # logger.warning(f"节点 {current} 没有可用的候选节点（经过筛选）")
                if len(path) <= 1:
                    logger.warning("无法生成有效路径，起点没有可用的候选节点")
                    return None
                path.pop()  # 回退
                current = path[-1]
                iterations -= 1
                continue

            try:
                next_step = possibility_select(candidates, weights)
                visited_nodes.add(next_step)

                # 更新路径长度
                next_pos = self.get_node_center(next_step)
                if current_pos and next_pos:
                    step_length = manhattan_distance(current_pos, next_pos)
                    eu_step_length = euclidean_distance(current_pos, next_pos)
                    path_length += step_length
                    eu_path_length += eu_step_length

                previous_step = current
                path.append(next_step)
                current = next_step

            except ValueError as e:
                logger.error(f"选择下一步时出错: {e}")
                return None

        if iterations >= self.MAX_ITERATIONS:
            logger.warning("达到最大迭代次数，路径生成失败")
            return None

        return path, path_length, eu_path_length, iterations

    def calculate_direction_score(self, current: str, candidate: str, exit_pos: Tuple[float, float, float],
                                  emergency_areas) -> Optional[
        float]:
        """计算方向评分"""

        # 获取节点坐标
        current_pos = self.get_node_center(current)
        candidate_pos = self.get_node_center(candidate)

        if not all([current_pos, candidate_pos, exit_pos]):
            return None

        try:
            # **1. 计算方向评分 C_d**
            # 获取当前楼层的层数（z 坐标）
            if self.graph.nodes[current].get('Entity_Type') in ['楼梯口', '建筑物出口']:
                C_d = 60
            else:
                current_floor = int(current_pos[2])

                # 如果楼层大于 1，计算到最近楼梯口的方向评分
                if current_floor > 1 and self.graph.nodes[current].get('Entity_Type') != '楼梯口':
                    # 获取当前楼层的楼梯口
                    staircases = self.get_staircases_on_current_floor(current_floor)
                    if not staircases:
                        logger.warning("没有楼梯口")
                        return 0  # 如果没有楼梯口，返回 0

                    # 找到最近的楼梯口
                    nearest_staircase = self.get_nearest_staircase(current_pos, staircases, current)
                    if not nearest_staircase:
                        logger.warning("不存在最近的楼梯口")
                        return 0

                    # 获取最近楼梯口的坐标
                    staircase_pos = self.get_node_center(nearest_staircase)
                    if not staircase_pos:
                        logger.warning("楼梯口坐标不存在")
                        return 0

                    # 计算到楼梯口的方向向量
                    vec_exit = np.array(staircase_pos) - np.array(current_pos)
                else:
                    # 如果楼层为 1，直接计算到出口的方向向量
                    vec_exit = np.array(exit_pos) - np.array(current_pos)

                vec_candidate = np.array(candidate_pos) - np.array(current_pos)

                exit_norm = np.linalg.norm(vec_exit)
                candidate_norm = np.linalg.norm(vec_candidate)

                if exit_norm == 0 or candidate_norm == 0:
                    return 0

                cosine = np.dot(vec_exit, vec_candidate) / (exit_norm * candidate_norm)
                C_d = max(0, cosine) * 60  # C_d = 60 * cos(θ)

            # **2. 计算楼梯评分 C_s**
            candidate_data = self.graph.nodes[candidate]
            C_s = 100 if candidate_data.get('Entity_Type') == '楼梯口' else 0  # 如果是楼梯口，加 100 分

            # **3. 计算紧急区域评分 C_l**
            C_l = -40 if candidate in emergency_areas else 0  # 如果是紧急区域，减 40 分

            # **4. 计算安全等级评分 C_l (对于非紧急区域)**
            # 定义安全等级到数值的映射
            SAFETY_LEVEL_MAPPING = {
                'Very Safe': 1.0,
                'Safe': 0.8,
                'Normal': 0.6,
                'Dangerous': 0.4,
                'Very Dangerous': 0.2
            }
            if candidate not in emergency_areas:
                # 获取安全等级，默认值为 'Normal'（对应 0.6）
                safety_level = self.graph.nodes[candidate].get('Safety Level', 'Normal')
                # 映射为数值
                safety_value = SAFETY_LEVEL_MAPPING.get(safety_level, 0.6)  # 默认值为 0.6
                C_l += -(1 - safety_value) * 20  # 根据安全等级调整评分

            # **5. 计算最终评分 W(candidate)**
            logger.debug(f"候选节点{candidate}的C_s={C_s}，C_d={C_d}，C_l={C_l}")
            W_candidate = 100 + C_s + C_d + C_l  # 论文公式: W(can_i) = C_0 + C_s + C_d + C_l

            return W_candidate

        except Exception as e:
            logger.error(f"计算方向评分时出错: {e}")
            return None

    def get_staircases_on_current_floor(self, floor):
        """获取当前楼层的所有楼梯口节点
        Args:
            floor: 当前楼层（z 坐标）
        Returns:
            List[str]: 当前楼层的楼梯口节点列表
        """
        staircases = [
            node for node, data in self.entities
            if data.get('Entity_Type') == '楼梯口' and int(data['Center'][2]) == floor
        ]
        return staircases

    def get_nearest_staircase(self, current_pos, staircases, current):
        """找到当前节点最近的楼梯口
        Args:
            current_pos: 当前节点的坐标 (x, y, z)
            staircases: 当前楼层的楼梯口节点列表
        Returns:
            Optional[str]: 最近的楼梯口节点 ID，如果找不到则返回 None
        """
        if not staircases:
            return None

        nearest_staircase = None
        min_distance = float('inf')

        for staircase in staircases:
            staircase_pos = self.get_node_center(staircase)
            if not staircase_pos:
                continue

            # 计算曼哈顿距离
            distance = manhattan_distance(current_pos, staircase_pos)

            # p, distance, s = self.visualizer.graph_manager.calculate_shortest_path(current, staircase)

            if distance < min_distance:
                min_distance = distance
                nearest_staircase = staircase

        return nearest_staircase

    def show_dialog(self):
        """显示路径生成对话框"""
        logger.debug("显示路径生成对话框")
        if not self.graph:
            logger.error("没有图形数据")
            messagebox.showerror("错误", "没有图形数据，请先加载图形", parent=self.parent)
            return

        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("生成路径")
        self.dialog.geometry("300x200")
        self.dialog.grab_set()

        # 创建输入框架
        input_frame = ttk.Frame(self.dialog)
        input_frame.pack(pady=20, padx=10, fill='both', expand=True)

        # 路径数量输入
        ttk.Label(input_frame, text="生成路径数量:").grid(row=0, column=0, sticky='w', pady=5)
        count_entry = ttk.Entry(input_frame)
        count_entry.grid(row=0, column=1, sticky='ew', pady=5)
        count_entry.insert(0, "10")  # 默认值

        # 起点输入
        ttk.Label(input_frame, text="起点(可选):").grid(row=1, column=0, sticky='w', pady=5)
        start_entry = ttk.Entry(input_frame)
        start_entry.grid(row=1, column=1, sticky='ew', pady=5)

        # 按钮框架
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(pady=10)

        def validate_and_generate():
            count_str = count_entry.get().strip()  # 获取用户输入并去除前后空格
            if not count_str.isdigit():  # 确保输入是数字
                messagebox.showerror("错误", "请输入有效的正整数", parent=self.dialog)
                return  # 终止操作

            count = int(count_str)  # 转换为整数
            if count <= 0:  # 检查是否是正整数
                messagebox.showerror("错误", "请输入大于0的正整数", parent=self.dialog)
                return

            self.generate_paths(count, start_entry.get().strip())  # 传入正确的参数

        # 确认按钮，调用验证函数
        confirm_button = ttk.Button(button_frame, text="确认", command=validate_and_generate)

        confirm_button.pack(pady=5)

        # 取消按钮
        cancel_button = ttk.Button(button_frame, text="取消", command=self.dialog.destroy)
        cancel_button.pack(pady=5)

    def generate_paths(self, count, specified_start=None):
        """处理路径生成命令"""

        # 获取指定的起点
        # specified_start = start_entry.get().strip()
        if specified_start and specified_start not in self.graph.nodes:
            messagebox.showerror("错误", f"指定的起点 {specified_start} 不存在", parent=self.dialog)
            return

        # 设置紧急实体
        set_emergency_area(self.graph)
        emergency_areas = select_emergency_areas(self.graph, num_emergencies=20)
        # emergency_areas = {}
        # 生成路径并计算评分
        successful_paths = 0
        total_score = 0
        max_score = float('-inf')
        min_score = float('inf')
        best_path = None
        worst_path = None
        paths_above_threshold = 0
        threshold = 0.3

        # 初始化保存路径和得分的列表
        all_paths = []
        all_scores = []

        for i in range(count):
            try:
                result = self.generate_path(emergency_areas, specified_start)
                if result is None:
                    logger.warning(f"路径 {i + 1} 生成失败")
                    continue

                path, path_length, eu_path_length, iteration = result
                if iteration == self.MAX_ITERATIONS:
                    logger.warning(f"路径 {i + 1} 生成失败，超出最大迭代次数")
                    continue

                score = self.evaluate_path(path, path_length, eu_path_length, emergency_areas)
                if score is None:
                    logger.warning(f"路径 {i + 1} 评分失败")
                    continue

                # self.highlight_path(path, emergency_areas)
                #
                # 保存路径及其分数
                all_paths.append(path)
                all_scores.append(score)
                #
                # 显示路径信息并等待用户确认
                path_info = f"路径 {i + 1}: {' -> '.join(f'{node}({self.get_node_center(node)[2]})' for node in path)},评分: {score:.2f}, 长度: {path_length:.2f}"
                # messagebox.showinfo("成功", path_info, parent=self.dialog)

                # logger.info(path_info)
                successful_paths += 1
                total_score += score

                # 统计评分大于threshold的路径
                if score > threshold:
                    paths_above_threshold += 1

                # 更新最高分和最低分
                if score > max_score:
                    max_score = score
                    best_path = path
                if score < min_score:
                    min_score = score
                    worst_path = path

            except Exception as e:
                logger.error(f"处理路径 {i + 1} 时出错: {e}")
                continue

        # 显示结果
        if successful_paths > 0:
            avg_score = total_score / successful_paths
            message = (
                f"成功生成 {successful_paths} 条路径\n"
                f"评分大于{threshold}的路径数量: {paths_above_threshold}\n"
                f"平均评分: {avg_score:.2f}\n"
                f"最高评分: {max_score:.2f}\n"
                f"最低评分: {min_score:.2f}"
            )
            # logger.info(message)
            # logger.info(f"最佳路径: {' -> '.join(best_path)}")
            # logger.info(f"最差路径: {' -> '.join(worst_path)}")
            # logger.info(f"评分大于{threshold}的路径数量: {paths_above_threshold}")
            # messagebox.showinfo("成功", message, parent=self.dialog)

            # 将路径及其分数传递给 dataset_create 函数
            if all_paths and all_scores:
                output_file = self.output_file
                output_file_14 = self.output_file_14
                path_output_file = self.path_output_file
                try:
                    self.dataset_create(all_paths, all_scores, output_file, path_output_file,
                                        emergency_areas,
                                        threshold)  # output_file_14,
                    logger.info(f"路径数据集已保存到文件 {output_file}")
                except Exception as e:
                    logger.error(f"保存数据集时出错: {e}")
        else:
            logger.warning("未能生成任何有效路径")
            # messagebox.showwarning("警告", "未能生成任何有效路径", parent=self.dialog)

        return emergency_areas
        # self.dialog.destroy()

    def select_start_area(self):
        """选择路径起点"""
        available = [
            node for node, data in self.entities
            if not data.get('is_exit', False) and node not in self.used_areas
        ]

        if not available:
            logger.debug("所有可用起点已使用，重置 used_areas")
            self.used_areas.clear()
            available = [
                node for node, data in self.entities
                if not data.get('is_exit', False)
            ]

        choice = random.choice(available)
        self.used_areas.add(choice)
        logger.debug(f"选择起点: {choice}")
        return choice

    def get_candidate_entities(self, current):
        """获取相邻节点"""
        candidates = list(self.get_node_neighbors(current))
        # logger.debug(f"获取节点 {current} 的候选节点: {candidates}")
        return candidates

    def evaluate_path(self, path, path_length, eu_path_length, emergency_areas=None):
        """计算路径评分"""
        logger.debug(f"开始计算路径评分: {path}")

        if len(path) < 2:
            logger.warning("路径长度不足，评分为 0")
            return 0.0

        # 获取起点和出口坐标
        start = path[0]
        if isinstance(start, int):
            start = str(start)

        start_pos = self.graph.nodes[start]['Center']
        exit = self.get_nearest_exit(start, start_pos)
        exit_pos = self.graph.nodes[exit]['Center']

        # **1. 计算路径长度评分 S_L**
        manhattan_dist = manhattan_distance(start_pos, exit_pos)
        # shortest_path, eu_dist, suc = self.visualizer.graph_manager.calculate_shortest_path(start, exit)
        S_L = manhattan_dist / path_length if path_length > 0 else 0
        logger.debug(
            f"路径长度评分 S_L: {S_L:.2f} (曼哈顿距离: {manhattan_dist:.2f}, 曼哈顿路径长度: {path_length:.2f},欧式路径长度:{eu_path_length}")  # ,欧式距离:{eu_dist:.2f})

        # **2. 计算紧急情况评分 S_E**
        S_E = 1.0
        if emergency_areas:
            for p in path:
                if p in emergency_areas:
                    S_E -= 0.3  # 经过紧急区域扣 0.3
                    # logger.info(f"节点 {p} 是紧急区域，扣除 0.3 分")
                elif any(n in emergency_areas for n in self.get_candidate_entities(p)):
                    S_E -= 0.1  # 相邻紧急区域扣 0.1
                    logger.debug(f"节点 {p} 邻近紧急区域，扣除 0.1 分")
        logger.debug(f"紧急情况评分 S_E: {S_E:.2f}")

        # **3. 计算奖励分数 S_b**
        n_p = len(set(path))  # 经过的不同区域数
        n_e = 0
        if emergency_areas:
            n_e = sum([1 for p in path if p in emergency_areas])  # 经过的紧急区域数
        S_b = n_p + n_e
        logger.debug(f"奖励分数 S_b: {S_b} (不同区域数: {n_p}, 紧急区域数: {n_e})")

        # **4. 计算最终得分**
        score = min(1.0, 0.3 * (S_L + S_E) + (1 / 250) * S_b)
        logger.debug(f"最终路径评分=0.3 * ({S_L} + {S_E}) + (1 / 250) * {S_b}={score:.3f}")

        return score

    def get_nearest_exit(self, start_id: str, start_pos: Tuple[float, float, float]) -> Optional[str]:
        """找到距离起点最近的出口节点
        Args:
            start_id: 起点节点的 ID
            start_pos: 起点节点的坐标 (x, y, z)
        Returns:
            Optional[str]: 最近的出口节点 ID，如果找不到则返回 None
        """
        if not self.graph:
            logger.error("图为空，无法查找出口")
            return None

        # 获取所有出口节点
        exit_nodes = [
            (node, data['Center']) for node, data in self.graph.nodes(data=True)
            if data.get('is_exit', False) and node != start_id  # 排除起点自身
        ]

        if not exit_nodes:
            logger.warning("图中没有出口节点")
            return None

        # 计算曼哈顿距离并找到最近的出口
        nearest_exit_id = None
        min_distance = float('inf')

        for exit_id, exit_pos in exit_nodes:
            # 解析出口坐标
            exit_pos_parsed = self.parse_center(exit_pos)
            if not exit_pos_parsed:
                continue

            # 计算曼哈顿距离
            distance = manhattan_distance(start_pos, exit_pos_parsed)
            if distance < min_distance:
                min_distance = distance
                nearest_exit_id = exit_id

        if not nearest_exit_id:
            logger.warning("未找到有效的出口节点")

        return nearest_exit_id

    def filter_candidates(self, candidates, current_node, previous_node=None):
        """筛选候选节点，删除度数为1的节点，检查度数为2的节点是否为死胡同
        
        Args:
            candidates (List[str]): 候选节点列表
            current_node (str): 当前节点
            previous_node (str, optional): 前一个节点，用于判断方向。默认为None
            
        Returns:
            List[str]: 筛选后的候选节点列表
        """
        if not candidates:
            return []

        filtered_candidates = []

        for candidate in candidates:
            # 获取候选节点的邻居
            neighbors = self.get_candidate_entities(candidate)
            degree = len(neighbors)

            # 跳过度数为1的节点（除非是出口节点）
            if degree == 1 and not self.is_exit_node(candidate):
                # logger.info(f"跳过度数为1的节点: {candidate}")
                continue

            # 检查度数为2的节点是否为死胡同
            if degree == 2 and previous_node is not None:
                # 如果候选节点的两个邻居是当前节点和前一个节点，则形成了一个回路，应该跳过
                if current_node in neighbors and previous_node in neighbors:
                    # logger.info(f"跳过形成回路的节点: {candidate}")
                    continue

                # 检查是否是死胡同（除了当前节点外，另一个邻居是度数为1的节点）
                other_neighbors = [n for n in neighbors if n != current_node]
                if other_neighbors:
                    other_neighbor = other_neighbors[0]
                    other_neighbor_degree = len(self.get_candidate_entities(other_neighbor))
                    if other_neighbor_degree == 1 and not self.is_exit_node(other_neighbor):
                        # logger.info(f"跳过通向死胡同的节点: {candidate}")
                        continue

            # 通过筛选的节点添加到结果列表
            filtered_candidates.append(candidate)

        # logger.debug(f"筛选前候选节点: {candidates}")
        # logger.debug(f"筛选后候选节点: {filtered_candidates}")

        return filtered_candidates

    def is_exit_node(self, node):
        """判断节点是否为出口节点
        
        Args:
            node (str): 节点ID
            
        Returns:
            bool: 是否为出口节点
        """
        # 检查 is_exit 属性
        is_exit = self.graph.nodes[node].get('is_exit', False)
        if isinstance(is_exit, str):
            is_exit = is_exit.lower() == 'true'

        # 检查 Entity_Type 属性
        entity_type = self.graph.nodes[node].get('Entity_Type', '')
        is_exit_type = entity_type == '建筑物出口'

        return bool(is_exit or is_exit_type)

    def highlight_path(self, path, emergency_areas):
        """高亮显示最短路径：
        - 路径上的边和节点颜色设为红色（除终点外）
        - 紧急区域显示为黄色
        - 楼梯显示为棕色
        - 非路径上的边、节点和标签透明度降低
        - 只显示路径、楼梯和出口的标签
        Args:
            path: 节点路径列表
            emergency_areas: 紧急区域列表
        """
        if not path or len(path) < 2:
            return

        # **创建路径的集合**
        path_edges = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
        path_edges |= {(b, a) for a, b in path_edges}  # 添加反向边
        path_nodes = set(path)  # 路径上的节点

        self.visualizer.color_handler.reset_colors()
        # **获取当前所有节点颜色**
        node_colors = self.visualizer.nodes_scatter.get_facecolors()

        # **更新节点样式**
        for i, node in enumerate(self.visualizer.graph_manager.graph.nodes()):
            color = node_colors[i].copy()  # 复制颜色

            # 检查是否为终点（最后一个节点）、楼梯和出口
            is_end_node = node == path[-1]
            is_staircase = self.graph.nodes[node].get('Entity_Type') == '楼梯口'
            is_exit = self.is_exit_node(node)

            if is_staircase:
                color[:3] = [0.647, 0.165, 0.165]  # 楼梯节点变棕色 (RGB: 褐红色)
                color[3] = 1.0  # 确保完全可见
            elif node in path_nodes and not is_end_node:
                if node in emergency_areas:
                    color[:3] = [1.0, 1.0, 0.0]  # 路径上的紧急区域节点变黄 (RGB: 黄色)
                else:
                    color[:3] = [1.0, 0.0, 0.0]  # 路径节点变红 (RGB: 红色)
                color[3] = 1.0  # 确保完全可见
            elif node in emergency_areas and not is_end_node:
                color[:3] = [1.0, 1.0, 0.0]  # 紧急区域节点变黄
            elif not is_end_node:
                color[3] = 0.1  # 非路径非紧急区域节点透明度降低

            node_colors[i] = color  # 更新颜色数组

            # **调整节点标签可见性**
            if node in self.visualizer.node_labels:
                # 只显示路径上的节点、楼梯和出口的标签
                if node in path_nodes or is_staircase or is_exit:
                    self.visualizer.node_labels[node].set_alpha(1.0)
                else:
                    self.visualizer.node_labels[node].set_alpha(0.0)  # 完全隐藏其他标签

        # **应用颜色更新**
        self.visualizer.nodes_scatter.set_facecolors(node_colors)

        # **更新边的样式**
        for edge, line in zip(self.visualizer.graph_manager.graph.edges(), self.visualizer.edges_lines):
            if edge in path_edges:
                line.set_color('red')  # **路径边变红**
                line.set_alpha(1.0)  # 确保完全可见
            else:
                line.set_alpha(0.1)  # **非路径边透明度降低**

        # **刷新画布**
        self.visualizer.canvas.draw_idle()

    def dataset_create(self, paths: List[List[str]], scores: List[float], output_file: str,
                       sorted_paths_output_file: str, emergency_areas, threshold,
                       sorted: int = 1):  # , output_file_14: str
        """
        编码路径和分数，并保存或追加到文件中，可选择是否排序，仅保留前 50000 条数据。

        Args:
            paths (List[List[str]]): 路径列表，每条路径是节点序列。
            scores (List[float]): 每条路径的得分列表，长度与 paths 相同。
            output_file (str): 保存数据集的文件路径（输入/输出编码）。
            output_file_14 (str): 保存 4 分类编码的文件路径。
            sorted_paths_output_file (str): 保存排序后的路径文件路径。
            emergency_areas: 紧急区域列表。
            threshold (float): 过滤的最低分数阈值。
            sorted (int): 是否对数据排序并保留前 50000 条（1=排序，0=不排序）
        """
        # paths ->List[List[str]] 数据校验
        new_paths = []
        for i, path in enumerate(paths):
            if not isinstance(path, list):
                logger.warning(f"第{i}条路径不是列表，已跳过：{path}")
                continue
            if not path:
                logger.warning(f"第{i}条路径为空，已跳过")
                continue

            corrected_path = []
            for j, node in enumerate(path):
                if not isinstance(node, str):
                    logger.info(f"将第{i}条路径中第{j}个节点 {node}（类型: {type(node)}）转换为字符串")
                    node = str(node)
                corrected_path.append(node)

            new_paths.append(corrected_path)

        paths = new_paths

        data = []
        # data_14 = []
        path_data = {}  # 记录路径及其分数 {路径ID: (路径字符串, 分数,紧急实体位置)}

        if len(paths) != len(scores):
            raise ValueError("路径列表与分数列表的长度不一致")

        # 编码路径
        for path_id, (path, score) in enumerate(zip(paths, scores)):
            if score >= threshold:
                for i in range(len(path) - 1):
                    current_node = path[i]
                    next_node = path[i + 1]

                    input_vector = self.encode_input(current_node, emergency_areas)
                    # output_vector_14 = self.encode_output_14(current_node, next_node, score)
                    # if len(output_vector_14) != 4:
                    #     print("len > 4")
                    #     continue
                    output_vector = self.encode_output(next_node, score)

                    data.append({
                        "path_id": path_id,
                        "input": input_vector,
                        "output": output_vector,
                        "score": score
                    })

                    # data_14.append({
                    #     "path_id": path_id,
                    #     "input": input_vector,
                    #     "output": output_vector_14,
                    #     "score": score
                    # })

                path_data[path_id] = (" -> ".join(path), score, emergency_areas)

        # 转为 DataFrame
        # new_df_14 = pd.DataFrame(data_14)
        new_df_all = pd.DataFrame(data)
        path_df = pd.DataFrame.from_dict(path_data, orient="index", columns=["path", "score", "emergency_area"])

        try:
            existing_df = pd.read_csv(output_file)
            # combined_df_14 = pd.concat([existing_df, new_df_14], ignore_index=True)
            combined_df = pd.concat([existing_df, new_df_all], ignore_index=True)
        except FileNotFoundError:
            # combined_df_14 = new_df_14
            combined_df = new_df_all

        try:
            existing_path_df = pd.read_csv(sorted_paths_output_file)
            combined_path_df = pd.concat([existing_path_df, path_df], ignore_index=True)
        except FileNotFoundError:
            combined_path_df = path_df

        # 是否排序与筛选
        if sorted:
            combined_df = combined_df.sort_values(by='score', ascending=False).reset_index(drop=True)
            combined_df = combined_df.head(50000)

            # combined_df_14 = combined_df_14.sort_values(by='score', ascending=False).reset_index(drop=True)
            # combined_df_14 = combined_df_14.head(50000)

            # 只保留得分前5万的路径
            top_path_ids = set(combined_df["path_id"].unique())
            combined_path_df = combined_path_df[combined_path_df.index.isin(top_path_ids)]
            combined_path_df = combined_path_df.sort_values(by='score', ascending=False).reset_index(drop=True)

        # 保存结果
        # combined_df_14.to_csv(output_file_14, index=False)
        combined_df.to_csv(output_file, index=False)
        combined_path_df.to_csv(sorted_paths_output_file, index=False)

        logger.info(f"数据集已保存到 {output_file}，共 {len(combined_df)} 条记录。")  # 和 {output_file_14}
        logger.info(f"路径文件保存到 {sorted_paths_output_file}，共 {len(combined_path_df)} 条记录。")

    def encode_output_14(self, current_node: str, next_node: str, score: float) -> List[float]:
        """
        测试
        编码输出向量，大小固定为 1×4，按邻接节点 ID 升序排序，并在正确位置设置目标节点路径得分。

        Args:
            current_node (str): 当前节点 ID。
            next_node (str): 下一步目标节点 ID。
            score (float): 对应路径的分数。

        Returns:
            List[float]: 长度固定为 4 的输出特征向量。
        """
        output_vector = [0.0] * 4  # 1×4 向量，初始全为 0

        # 获取当前节点的邻接节点，并按 ID 升序排序
        neighbors = sorted(list(self.graph.neighbors(current_node)))

        # 仅保留前 4 个邻接节点（如果邻接节点少于 4 个，则不填充）
        neighbors = neighbors[:4]

        if next_node in neighbors:
            index = neighbors.index(next_node)  # 找到目标节点在邻接节点中的索引
            output_vector[index] = score  # 在相应位置填充路径得分

        return output_vector

    def get_node_neighbors(self, node):
        """获取节点的邻居
        node: str
        """
        # logger.debug(f"Attempting to get neighbors for node: {node}")
        # logger.debug(f"Graph exists: {self.graph is not None}")
        if not isinstance(node, str):
            # logger.warning(f"{node} is not str, converting to...")
            node = str(node)
        if self.graph:
            # logger.debug(f"Node type: {type(node)}")
            # logger.debug(f"Nodes in graph: {list(self.graph.nodes())[:5]}...")  # 只打印前5个节点避免日志过长
            # logger.debug(f"Node in graph: {node in self.graph}")
            if node in self.graph:
                neighbors = list(self.graph.neighbors(node))
                logger.debug(f"Found {len(neighbors)} neighbors for node {node}")
                return neighbors
            else:
                logger.warning(f"Node {node} not found in graph")
        return []

    def new_generate_paths(self, count=1000, specific_start=None):
        """
        :param count:
        :param specific_start:
        :return:
        """
        paths = []
        scores = []
        paths_scores_map = []
        danger_zones = []
        set_emergency_area(self.graph)
        for i in range(count + 1):
            # if i / 617 > 1: # 一些无紧急实体测试
            #
            danger_zones = select_emergency_areas(self.graph, num_emergencies=10)  # 增加紧急实体数量
            # 选择起点
            if specific_start:
                result = self.generate_path(emergency_areas=danger_zones, specified_start=str(specific_start))  # j
            else:
                j = i % len(self.graph.nodes)
                if j in [0, 27, 104, 105, 121, 122, 129, 151, 152, 469, 478] or self.is_exit_node(str(j)):
                    continue
                result = self.generate_path(emergency_areas=danger_zones, specified_start=str(j))  # j

            if result is None:
                logger.warning(f"路径 {i + 1} 生成失败")
                continue
            path, path_length, eu_path_length, iteration = result

            if iteration == self.MAX_ITERATIONS:
                logger.warning(f"路径 {i + 1} 生成失败，超出最大迭代次数")
                continue

            score = self.evaluate_path(path, path_length, eu_path_length, danger_zones)
            if path and score:
                paths.append(path)
                scores.append(score)
                paths_scores_map.append([path, score, danger_zones])
                print(f"轮次{i}:{path, score, danger_zones}")

            if i % (count / 3) == 0:
                paths = [item[0] for item in paths_scores_map]
                scores = [item[1] for item in paths_scores_map]
                danger_zones = paths_scores_map[-1][2] if paths_scores_map else []
                # if i > 627:
                self.dataset_create(paths, scores, self.output_file, self.path_output_file,
                                    danger_zones, threshold=0.3, sorted=1)  # self.output_file_14,
                # else:
                #     self.dataset_create(paths, scores, "output_all_start_paths.csv", self.output_file_14,
                #                         "output_path_all_start_paths.csv", danger_zones, threshold=0.1, sorted=0)
                paths_scores_map.clear()

        paths = [item[0] for item in paths_scores_map]
        scores = [item[1] for item in paths_scores_map]
        danger_zones = paths_scores_map[-1][2] if paths_scores_map else []
        self.dataset_create(paths, scores, self.output_file, self.path_output_file,
                            danger_zones,
                            threshold=0.3, sorted=1)  # self.output_file_14,


def encode_input(graph, current_node: str, emergency_areas) -> List[float]:
    """
    编码输入向量。

    Args:
        current_node (str): 当前节点 ID。

    Returns:
        List[float]: 输入特征向量（稀疏表示）。
    """
    if not isinstance(current_node, str):
        current_node = str(current_node)
    # 1. 当前节点的位置
    current_node_vector = [1.0 if node == current_node else 0.0 for node in graph.nodes]
    # 2. 紧急区域的状态
    emergency_vector = [1.0 if node in emergency_areas else 0.0 for node in graph.nodes]
    # 3. 合并输入向量并添加常数 1
    input_vector = current_node_vector + emergency_vector + [1.0]
    return input_vector


def encode_output(graph, next_node: str, score: float) -> List[float]:
    """
    编码输出向量。

    Args:
        next_node (str): 下一步目标节点 ID。
        score (float): 对应路径的分数。

    Returns:
        List[float]: 输出特征向量。
    """
    # 创建全为 0 的向量
    output_vector = [0.0 for _ in graph.nodes]

    # 将目标节点位置赋值为路径得分
    node_index = list(graph.nodes).index(next_node)
    output_vector[node_index] = score

    return output_vector


if __name__ == "__main__":
    grapml = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\test03.graphml"
    # output_file14 = "test_emergency.csv"  # "output_file_140407.csv"
    output_file = "test_emergency.csv"  # "output_file_0407.csv"
    output_path = "test_emergency_path.csv"  # "output_path.csv"
    generate_datasets = DataSetGenerate(output_file, output_path, output_file, grapml)
    generate_datasets.new_generate_paths(count=700)
