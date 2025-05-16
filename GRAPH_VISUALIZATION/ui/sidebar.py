import tkinter as tk
from tkinter import ttk

class Sidebar:
    def __init__(self, parent, visualizer):
        self.visualizer = visualizer
        self.parent = parent
        
        # 创建侧栏框架
        self.sidebar_frame = ttk.Frame(parent, style='Sidebar.TFrame')
        self.sidebar_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建切换按钮框架
        self.toggle_frame = ttk.Frame(self.sidebar_frame)
        self.toggle_frame.pack(side=tk.TOP, fill=tk.X)
        
        # 创建切换按钮
        self.toggle_button = ttk.Button(
            self.toggle_frame,
            text='◀',  # 使用箭头符号
            width=2,
            command=self.toggle_sidebar
        )
        self.toggle_button.pack(side=tk.RIGHT)
        
        # 创建内容框架
        self.content_frame = ttk.Frame(self.sidebar_frame)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # 创建标题
        self.title_label = ttk.Label(
            self.content_frame,
            text='节点属性',
            font=('Arial', 12, 'bold')
        )
        self.title_label.pack(side=tk.TOP, pady=10)
        
        # 创建属性框架
        self.properties_frame = ttk.Frame(self.content_frame)
        self.properties_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10)
        
        # 创建滚动条
        self.scrollbar = ttk.Scrollbar(self.properties_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 创建画布用于滚动
        self.canvas = tk.Canvas(
            self.properties_frame,
            yscrollcommand=self.scrollbar.set,
            highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 配置滚动条
        self.scrollbar.config(command=self.canvas.yview)
        
        # 创建属性内容框架
        self.properties_content = ttk.Frame(self.canvas)
        self.canvas.create_window(
            (0, 0),
            window=self.properties_content,
            anchor='nw',
            width=self.canvas.winfo_reqwidth()
        )
        
        # 绑定事件
        self.properties_content.bind('<Configure>', self._on_frame_configure)
        self.canvas.bind('<Configure>', self._on_canvas_configure)
        
        # 设置样式
        self.setup_styles()
        
        # 初始状态：显示
        self.is_visible = True
        self.sidebar_width = 300  # 侧栏宽度
        
        # 存储属性控件
        self.property_widgets = {}
        
    def setup_styles(self):
        """设置自定义样式"""
        style = ttk.Style()
        
        # 侧栏框架样式
        style.configure('Sidebar.TFrame', background='#f5f5f5')
        
        # 属性标签样式
        style.configure('PropertyLabel.TLabel', padding=(5, 2))
        
        # 属性值样式
        style.configure('PropertyValue.TEntry', padding=(5, 2))
        
    def toggle_sidebar(self):
        """切换侧栏显示状态"""
        if self.is_visible:
            self.content_frame.pack_forget()
            self.toggle_button.configure(text='▶')
            self.sidebar_frame.configure(width=30)
        else:
            self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            self.toggle_button.configure(text='◀')
            self.sidebar_frame.configure(width=self.sidebar_width)
        self.is_visible = not self.is_visible
        
    def update_properties(self, node_id, properties):
        """更新节点属性显示"""
        # 清除现有属性
        for widget in self.property_widgets.values():
            for w in widget:
                w.destroy()
        self.property_widgets.clear()
        
        # 如果没有选中节点，清空显示
        if node_id is None:
            self.title_label.configure(text='节点属性')
            return
            
        # 更新标题
        self.title_label.configure(text=f'节点 {node_id} 的属性')
        
        # 添加新属性
        for i, (key, value) in enumerate(properties.items()):
            # 创建属性标签
            label = ttk.Label(
                self.properties_content,
                text=f'{key}:',
                style='PropertyLabel.TLabel'
            )
            label.grid(row=i, column=0, sticky='w', pady=2)
            
            # 创建属性值输入框
            value_var = tk.StringVar(value=str(value))
            entry = ttk.Entry(
                self.properties_content,
                textvariable=value_var,
                style='PropertyValue.TEntry'
            )
            entry.grid(row=i, column=1, sticky='ew', padx=5, pady=2)
            
            # 存储控件引用
            self.property_widgets[key] = (label, entry, value_var)
            
        # 配置网格列权重
        self.properties_content.grid_columnconfigure(1, weight=1)
        
    def _on_frame_configure(self, event=None):
        """处理内容框架大小变化"""
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        
    def _on_canvas_configure(self, event=None):
        """处理画布大小变化"""
        if event:
            # 更新内容框架宽度以匹配画布
            self.canvas.itemconfig(
                self.canvas.find_withtag('all')[0],
                width=event.width
            ) 