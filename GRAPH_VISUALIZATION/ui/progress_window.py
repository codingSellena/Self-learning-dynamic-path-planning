import tkinter as tk
from tkinter import ttk

class ProgressWindow:
    def __init__(self, parent, title="加载进度"):
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        
        # 设置窗口大小和位置
        window_width = 400
        window_height = 100
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.window.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 设置窗口属性
        self.window.transient(parent)  # 设置为父窗口的临时窗口
        self.window.grab_set()  # 模态窗口
        self.window.resizable(False, False)  # 禁止调整大小
        
        # 创建标签用于显示当前执行的函数名
        self.status_label = ttk.Label(self.window, text="准备加载...", font=('SimHei', 10))
        self.status_label.pack(pady=10)
        
        # 创建进度条
        self.progress = ttk.Progressbar(
            self.window,
            orient="horizontal",
            length=350,
            mode="determinate"
        )
        self.progress.pack(pady=5)
        
        # 创建百分比标签
        self.percent_label = ttk.Label(self.window, text="0%", font=('SimHei', 9))
        self.percent_label.pack(pady=5)
        
        # 初始化进度
        self.progress["value"] = 0
        self.window.update()
    
    def update_progress(self, value, status_text=None):
        """更新进度条和状态文本"""
        self.progress["value"] = value
        self.percent_label.config(text=f"{int(value)}%")
        if status_text:
            self.status_label.config(text=status_text)
        self.window.update()
    
    def close(self):
        """关闭进度条窗口"""
        self.window.destroy() 