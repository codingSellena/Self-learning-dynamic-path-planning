import tkinter as tk
import sys
import os
import matplotlib.pyplot as plt

# 在创建图形之前设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from GRAPH_VISUALIZATION.core.visualizer import GraphVisualizer
from GRAPH_VISUALIZATION.utils.style_utils import setup_matplotlib_style

def main():
    # 设置matplotlib样式
    setup_matplotlib_style()
    
    # 创建主窗口
    root = tk.Tk()
    
    # 创建可视化器
    visualizer = GraphVisualizer(root)
    
    # 运行主循环
    root.mainloop()

if __name__ == "__main__":
    main()