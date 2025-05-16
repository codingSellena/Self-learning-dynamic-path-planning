import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class CanvasManager:
    """管理matplotlib画布的类"""

    def __init__(self, parent, figsize=(8, 10)):
        self.parent = parent
        self.setup_canvas(figsize)

    def setup_canvas(self, figsize=(8,10)):
        """设置matplotlib画布"""
        # 创建图形和轴对象
        self.fig = plt.figure(figsize=figsize, dpi=100)

        # 添加子图，并设置位置和大小，增加可用空间
        self.ax = self.fig.add_axes([0.02, 0.02, 0.96, 0.96])  # 减小边距，增大可用空间

        # 创建canvas并配置
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.get_tk_widget().grid(
            row=1, column=0,
            sticky=(tk.N, tk.S, tk.E, tk.W),
            padx=5,  # 减小内边距
            pady=5   # 减小内边距
        )
        # 完全隐藏坐标轴和刻度
        self.ax.set_axis_off()  # 关闭整个坐标轴系统

        # 确保坐标轴和刻度完全隐藏
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])

        # 设置网格权重使画布可以自适应调整大小
        self.parent.grid_rowconfigure(1, weight=1)
        self.parent.grid_columnconfigure(0, weight=1)

    def set_background_color(self, fig_color, ax_color):
        """设置背景颜色"""
        self.fig.patch.set_facecolor(fig_color)
        self.ax.set_facecolor(ax_color)

    def set_spine_style(self, visible=True, color='gray', width=1):
        """设置轴线样式"""
        for spine in self.ax.spines.values():
            spine.set_visible(visible)
            if visible:
                spine.set_color(color)
                spine.set_linewidth(width)

    def clear(self):
        """清空画布"""
        self.ax.clear()
        self.ax.set_axis_off()

    def get_canvas(self):
        """获取canvas对象"""
        return self.canvas

    def get_figure(self):
        """获取figure对象"""
        return self.fig

    def get_axes(self):
        """获取axes对象"""
        return self.ax
