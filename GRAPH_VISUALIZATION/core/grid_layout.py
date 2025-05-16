import networkx as nx
import numpy as np
from collections import deque
from typing import Dict, Set, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GridCrossingLayout:
    def __init__(self, graph: nx.Graph, grid_spacing: float = 100.0):
        """
        初始化网格交叉布局算法
        
        Args:
            graph: NetworkX图对象
            grid_spacing: 网格间距
        """
        self.graph = graph
        self.grid_spacing = grid_spacing * 5
        self.positions = {}  # 存储节点位置
        self.occupied_positions = set()  # 已占用的网格点
        self.grid_points = {}  # 网格点到节点的映射

    def _get_node_degree_order(self) -> List[str]:
        """获取按度数排序的节点列表"""
        return sorted(self.graph.nodes(),
                      key=lambda x: (self.graph.degree[x],
                                     float(self.graph.nodes[x].get('lc', 0))),
                      reverse=True)

    def _is_position_available(self, pos: Tuple[float, float]) -> bool:
        """检查位置是否可用"""
        return pos not in self.occupied_positions

    def _get_available_neighbors(self, pos: Tuple[float, float], radius: int = 1) -> List[Tuple[float, float]]:
        """获取可用的相邻网格点
        
        Args:
            pos: 当前位置
            radius: 搜索半径
            
        Returns:
            可用的网格点列表
        """
        available = []
        x, y = pos

        # 按照"十字"形状扩展，优先考虑水平和垂直方向
        directions = [
            (1, 0), (-1, 0),  # 水平
            (0, 1), (0, -1),  # 垂直
            (1, 1), (-1, 1), (1, -1), (-1, -1)  # 对角线
        ]

        for dx, dy in directions:
            for r in range(1, radius + 1):
                new_pos = (x + dx * r * self.grid_spacing,
                           y + dy * r * self.grid_spacing)
                if self._is_position_available(new_pos):
                    available.append(new_pos)

        return available

    def _assign_position(self, node: str, pos: Tuple[float, float]):
        """为节点分配位置"""
        self.positions[node] = pos
        self.occupied_positions.add(pos)
        self.grid_points[pos] = node

    def _get_optimal_neighbor_position(self, node: str, neighbors: List[str]) -> Optional[Tuple[float, float]]:
        """为节点找到最优的邻居位置
        
        Args:
            node: 当前节点
            neighbors: 邻居节点列表
            
        Returns:
            最优位置坐标
        """
        if not neighbors:
            return None

        # 计算已放置邻居的平均位置
        placed_neighbors = [n for n in neighbors if n in self.positions]
        if not placed_neighbors:
            return None

        avg_x = sum(self.positions[n][0] for n in placed_neighbors) / len(placed_neighbors)
        avg_y = sum(self.positions[n][1] for n in placed_neighbors) / len(placed_neighbors)

        # 找到最近的可用网格点
        best_pos = None
        min_dist = float('inf')

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                grid_x = round(avg_x / self.grid_spacing + dx) * self.grid_spacing
                grid_y = round(avg_y / self.grid_spacing + dy) * self.grid_spacing
                pos = (grid_x, grid_y)

                if self._is_position_available(pos):
                    dist = ((grid_x - avg_x) ** 2 + (grid_y - avg_y) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_pos = pos

        return best_pos

    def compute_layout(self) -> Dict[str, Tuple[float, float]]:
        """计算网格交叉布局
        
        Returns:
            节点位置字典
        """
        # 获取按度数排序的节点
        nodes = self._get_node_degree_order()
        if not nodes:
            return {}

        # 从度数最大的节点开始
        start_node = nodes[0]
        self._assign_position(start_node, (0, 0))

        # 使用BFS遍历图
        queue = deque([(start_node, 1)])  # (节点, 层级)
        processed = {start_node}

        while queue:
            current_node, level = queue.popleft()
            neighbors = list(self.graph.neighbors(current_node))

            # 按度数排序邻居
            neighbors.sort(key=lambda x: self.graph.degree[x], reverse=True)

            for neighbor in neighbors:
                if neighbor in processed:
                    continue

                # 找到最优位置
                optimal_pos = self._get_optimal_neighbor_position(neighbor,
                                                                  [n for n in self.graph.neighbors(neighbor) if
                                                                   n in self.positions])

                if not optimal_pos:
                    # 如果找不到最优位置，在当前节点周围找一个可用位置
                    available = self._get_available_neighbors(self.positions[current_node], radius=level)
                    if available:
                        optimal_pos = min(available,
                                          key=lambda p: sum((p[0] - self.positions[n][0]) ** 2 +
                                                            (p[1] - self.positions[n][1]) ** 2
                                                            for n in self.graph.neighbors(neighbor)
                                                            if n in self.positions))

                if optimal_pos:
                    self._assign_position(neighbor, optimal_pos)
                    queue.append((neighbor, level + 1))
                    processed.add(neighbor)

        # 调整Z坐标（楼层）
        for node in self.positions:
            pos = list(self.positions[node])
            floor = float(self.graph.nodes[node].get('lc', 0))
            pos.append(floor)  # 添加Z坐标
            self.positions[node] = tuple(pos)

        return self.positions

    def get_layout_statistics(self) -> Dict:
        """获取布局统计信息"""
        if not self.positions:
            return {}

        # 计算边交叉数
        crossings = self._count_edge_crossings()

        # 计算节点间距统计
        distances = []
        for n1 in self.graph.nodes():
            for n2 in self.graph.nodes():
                if n1 < n2:  # 避免重复计算
                    if n1 in self.positions and n2 in self.positions:
                        dist = np.sqrt(sum((a - b) ** 2
                                           for a, b in zip(self.positions[n1][:2],
                                                           self.positions[n2][:2])))
                        distances.append(dist)

        return {
            'edge_crossings': crossings,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0,
            'avg_distance': sum(distances) / len(distances) if distances else 0,
            'total_nodes': len(self.positions),
            'grid_spacing': self.grid_spacing
        }

    def _count_edge_crossings(self) -> int:
        """计算边交叉数量"""
        crossings = 0
        edges = list(self.graph.edges())

        for i, (u1, v1) in enumerate(edges):
            for u2, v2 in edges[i + 1:]:
                # 检查是否有共同顶点
                if len(set([u1, v1, u2, v2])) < 4:
                    continue

                # 检查边是否相交
                if self._do_edges_intersect(
                        self.positions[u1][:2], self.positions[v1][:2],
                        self.positions[u2][:2], self.positions[v2][:2]
                ):
                    crossings += 1

        return crossings

    def _do_edges_intersect(self, p1: Tuple[float, float], p2: Tuple[float, float],
                            p3: Tuple[float, float], p4: Tuple[float, float]) -> bool:
        """检查两条边是否相交"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
