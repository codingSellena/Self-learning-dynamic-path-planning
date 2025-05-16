import tkinter as tk
from tkinter import ttk

class StatusBar:
    """状态栏组件，显示当前操作状态"""
    
    def __init__(self, parent):
        # 创建状态栏框架
        self.frame = ttk.Frame(parent)
        self.frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(5,2))
        
        # 配置框架的网格权重，使状态栏可以水平拉伸
        self.frame.grid_columnconfigure(0, weight=1)
        
        # 创建状态文本变量
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")  # 设置默认状态文本
        
        # 创建状态标签
        self.status_label = ttk.Label(
            self.frame,
            textvariable=self.status_var,
            anchor="w",  # 文本左对齐
            padding=(5, 2)  # 内边距 (x, y)
        )
        self.status_label.grid(row=0, column=0, sticky="ew")

    def set_status(self, text):
        """设置状态栏文本"""
        if not text:
            self.status_var.set("就绪")
        else:
            self.status_var.set(text)

    def clear(self):
        """清空状态栏"""
        self.status_var.set("就绪")

    def update_node_info(self, node=None, neighbors=None):
        """更新节点信息显示"""
        if node is None:
            self.clear()
        else:
            if neighbors is None:
                neighbors = []
            neighbor_text = ", ".join(str(n) for n in neighbors) if neighbors else "无"
            self.set_status(f"选中节点: {node} | 相邻节点: {neighbor_text}")

    def show_hover_info(self, node=None):
        """显示悬停节点信息"""
        if node:
            self.set_status(f"当前悬停节点: {node}")
        else:
            self.clear()