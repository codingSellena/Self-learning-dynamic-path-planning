from tkinter import colorchooser, messagebox
from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()


class ColorHandler:
    def __init__(self, visualizer):
        self.visualizer = visualizer
        self.node_colors = {}  # 存储自定义节点颜色
        self.default_node_color = '#ADD8E6' # 淡蓝
        self.highlight_node_color = '#FF0000' # red
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

    def choose_color(self):
        """打开颜色选择器并应用所选颜色到选中的节点"""
        if not self.visualizer.selected_node:
            messagebox.showwarning("警告", "请先选择一个节点")
            return

        logger.info(f"Opening color chooser for node: {self.visualizer.selected_node}")
        
        # 获取当前节点的颜色作为初始颜色
        initial_color = self.get_node_color(self.visualizer.selected_node)
        logger.debug(f"Initial color for node {self.visualizer.selected_node}: {initial_color}")

        # 打开颜色选择器
        color = colorchooser.askcolor(
            title="选择节点颜色",
            color=initial_color
        )

        if color[1]:  # color[1] 是十六进制颜色代码
            old_color = self.node_colors.get(self.visualizer.selected_node, "default")
            self.node_colors[self.visualizer.selected_node] = color[1]
            logger.info(f"Color changed for node {self.visualizer.selected_node}: {old_color} -> {color[1]}")
            # 清除样式缓存，因为颜色已更改
            self.visualizer.clear_style_cache()
            self.visualizer.update_styles()

    def reset_colors(self):
        """重置所有节点的颜色为默认值"""
        self.node_colors.clear()
        # 清除样式缓存，因为颜色已重置
        self.visualizer.clear_style_cache()
        self.visualizer.update_display()

    def get_node_color(self, node):
        """获取节点的颜色
        优先级：自定义颜色->高亮颜色->出口->楼层颜色->默认颜色
        """
        logger.debug(f"Getting color for node: {node}")
        
        # 如果节点有自定义颜色，返回自定义颜色
        if node in self.node_colors:
            color = self.node_colors[node]
            logger.debug(f"Using custom color for node {node}: {color}")
            return color
            
        # 如果是选中的节点，返回高亮颜色
        elif node == self.visualizer.selected_node:
            logger.debug(f"Using highlight color for selected node {node}: {self.highlight_node_color}")
            return self.highlight_node_color
            
        # 如果是出口节点，返回绿色
        elif self.visualizer.graph_manager.get_node_attribute(node, 'is_exit'):
            logger.debug(f"Using exit color (green) for node {node}")
            return '#008000' # green
            
        # 获取节点的楼层信息
        try:
            # 获取楼层值并确保它是一个整数
            floor_value = self.visualizer.graph_manager.graph.nodes[node].get('lc')
            if floor_value is not None:
                floor = str(int(float(floor_value)))  # 确保转换为整数再转字符串
                logger.debug(f"Node {node} is on floor {floor}")
            else:
                floor = 'default'
                logger.debug(f"Node {node} has no floor value, using default")
        except (ValueError, TypeError) as e:
            floor = 'default'
            logger.warning(f"Error getting floor value for node {node}: {e}")

        # 获取对应的颜色
        color = self.floor_colors.get(floor, self.floor_colors['default'])
        logger.debug(f"Using floor color for node {node}: {color}")
        return color

    def get_edge_color(self, edge):
        """根据边连接的节点楼层获取边的颜色"""
        if not self.visualizer.graph_manager.graph or edge not in self.visualizer.graph_manager.graph.edges:
            logger.warning(f"Invalid edge: {edge}")
            return self.floor_colors['default']

        # 获取边两端节点的楼层
        node1, node2 = edge
        floor1 = str(self.visualizer.graph_manager.graph.nodes[node1].get('lc', 'default'))
        floor2 = str(self.visualizer.graph_manager.graph.nodes[node2].get('lc', 'default'))

        logger.debug(f"Getting edge color for {edge}: floor1={floor1}, floor2={floor2}")

        # 如果两个节点在同一楼层，使用该楼层的颜色
        if floor1 == floor2:
            color = self.floor_colors.get(floor1, self.floor_colors['default'])
            logger.debug(f"Using same-floor color for edge {edge}: {color}")
            return color
        # 如果不在同一楼层，使用默认颜色
        logger.debug(f"Using default color for cross-floor edge {edge}")
        return self.floor_colors['default']
