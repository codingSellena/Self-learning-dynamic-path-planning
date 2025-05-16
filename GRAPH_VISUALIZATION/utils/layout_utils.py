import numpy as np


def find_nearest_node(node_positions, x, y, max_distance=None):
    """查找最近的节点
    Args:
        node_positions: 节点位置字典
        x: x坐标
        y: y坐标
        max_distance: 最大搜索距离
    Returns:
        str: 最近节点的ID，如果没有找到则返回None
    """
    # 使用graph_manager的KD树查找最近节点
    if hasattr(node_positions, 'get_nearest_node'):
        nearest_node, distance = node_positions.get_nearest_node(x, y, max_distance)
        if nearest_node is not None:
            return nearest_node
        return None

    # 如果没有KD树，使用原来的方法
    if not node_positions:
        return None

    # 计算所有节点到点击位置的距离
    distances = {}
    for node_id, pos in node_positions.items():
        dx = pos[0] - x
        dy = pos[1] - y
        distance = np.sqrt(dx * dx + dy * dy)
        distances[node_id] = distance

    # 找到最近的节点
    if distances:
        nearest_node = min(distances.items(), key=lambda x: x[1])
        if max_distance is None or nearest_node[1] <= max_distance:
            return nearest_node[0]
    return None
