import tkinter as tk
from tkinter import ttk, messagebox

from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()

class ToolBar(ttk.Frame):
    """工具栏组件，包含各种操作按钮和控件"""

    def __init__(self, parent, callbacks):
        super().__init__(parent)
        self.callbacks = callbacks
        self.graph_manager = callbacks.get('graph_manager')
        self.visualizer = callbacks.get('visualizer')
        self.grid(row=0, column=0, sticky="ew")

        # 创建工具栏框架
        self.toolbar_frame = ttk.Frame(parent)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        # 文件操作（折叠菜单）
        self.file_menu_button = ttk.Menubutton(self.toolbar_frame, text="文件操作", direction="below")
        self.file_menu_button.pack(side=tk.LEFT, padx=2)

        # 创建 `Menu` 作为下拉菜单
        self.file_menu = tk.Menu(self.file_menu_button, tearoff=0)
        self.file_menu.add_command(label="加载文件", command=callbacks['load_file'])
        self.file_menu.add_command(label="保存文件", command=callbacks['save_file'])
        self.file_menu.add_command(label="保存图片", command=callbacks['save_image'])
        self.file_menu.add_command(label="Excel转换", command=callbacks['excel2graphml'])

        # 绑定 `Menu` 到 `Menubutton`
        self.file_menu_button["menu"] = self.file_menu

        # 布局控制
        self.layout_frame = ttk.LabelFrame(self.toolbar_frame, text="布局控制")
        self.layout_frame.pack(side=tk.LEFT, padx=2)

        self.layout_var = tk.StringVar(value="fruchterman_reingold")
        layout_options = ["fruchterman_reingold", "floor_based"] # , "grid_crossing"
        self.layout_combo = ttk.Combobox(self.layout_frame, textvariable=self.layout_var, values=layout_options,
                                         width=15)
        self.layout_combo.pack(side=tk.LEFT, padx=1)

        # 绑定布局变化事件
        self.layout_var.trace_add('write', lambda *args: callbacks['relayout']())

        # 缩放控制
        self.zoom_frame = ttk.LabelFrame(self.toolbar_frame, text="缩放控制")
        ttk.Button(self.zoom_frame, text="-", width=2,
                   command=lambda: self._adjust_zoom(-10)).pack(side=tk.LEFT, padx=1)
        self.zoom_frame.pack(side=tk.LEFT, padx=2)

        # 使用百分比表示
        self.zoom_var = tk.StringVar(value="100%")
        self.zoom_entry = ttk.Entry(self.zoom_frame, textvariable=self.zoom_var, width=6)
        self.zoom_entry.pack(side=tk.LEFT, padx=1)

        # 绑定回车键和验证
        self.zoom_entry.bind('<Return>', self._validate_and_apply_zoom)

        # 加减按钮使用百分比增量
        ttk.Button(self.zoom_frame, text="+", width=2,
                   command=lambda: self._adjust_zoom(10)).pack(side=tk.LEFT, padx=1)

        # 颜色控制
        self.color_frame = ttk.LabelFrame(self.toolbar_frame, text="颜色控制")
        self.color_frame.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.color_frame, text="选择颜色", command=callbacks['choose_color']).pack(side=tk.LEFT, padx=1)
        ttk.Button(self.color_frame, text="重置样式", command=callbacks['reset_colors']).pack(side=tk.LEFT, padx=1)

        # 标签控制
        self.label_frame = ttk.LabelFrame(self.toolbar_frame, text="标签控制")
        self.label_frame.pack(side=tk.LEFT, padx=2)

        self.label_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.label_frame, text="显示标签", variable=self.label_var,
                        command=callbacks['toggle_labels']).pack(side=tk.LEFT, padx=1)

        # 大小控制
        self.size_frame = ttk.LabelFrame(self.toolbar_frame, text="大小控制")
        self.size_frame.pack(side=tk.LEFT, padx=2)

        size_control_frame = ttk.Frame(self.size_frame)
        size_control_frame.pack(side=tk.LEFT, padx=1)

        ttk.Label(size_control_frame, text="边宽度:").grid(row=0, column=0, padx=1)
        self.edge_scale = ttk.Scale(size_control_frame, from_=0.1, to=5.0, orient=tk.HORIZONTAL, length=80)
        self.edge_scale.set(1.0)
        self.edge_scale.grid(row=0, column=1, padx=1)

        ttk.Label(size_control_frame, text="节点大小:").grid(row=1, column=0, padx=1)
        self.node_scale = ttk.Scale(size_control_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL, length=80)
        self.node_scale.set(1.0)
        self.node_scale.grid(row=1, column=1, padx=1)

        # 绑定大小控制的回调
        self.edge_scale.configure(command=lambda x: self._on_size_change())
        self.node_scale.configure(command=lambda x: self._on_size_change())

        # 节点编辑框架
        self.node_edit_frame = ttk.LabelFrame(self.toolbar_frame, text="节点编辑")
        self.node_edit_frame.pack(side=tk.LEFT, padx=2)

        ttk.Button(self.node_edit_frame, text="添加节点属性",
                   command=callbacks['add_custom_node_attribute']).pack(side=tk.LEFT, padx=1)

        # 添加最短路径按钮
        ttk.Button(self.node_edit_frame, text="最短路径",
                   command=callbacks['shortest_path']).pack(side=tk.LEFT, padx=1)

        ttk.Button(self.node_edit_frame, text="可视化路径",
                   command=callbacks['visualize_path']).pack(side=tk.LEFT, padx=1)

        # 添加生成路径按钮
        # ttk.Button(self.node_edit_frame, text="生成路径",
        #            command=callbacks['generate_paths']).pack(side=tk.LEFT, padx=1)

        # 添加性能测试按钮
        # ttk.Button(self.node_edit_frame, text="开始性能测试",
        #            command=callbacks['start_performance_test']).pack(side=tk.LEFT, padx=1)
        # ttk.Button(self.node_edit_frame, text="停止性能测试",
        #            command=callbacks['stop_performance_test']).pack(side=tk.LEFT, padx=1)

        # 搜索框架
        self.search_frame = ttk.LabelFrame(self.toolbar_frame, text="搜索")
        self.search_frame.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)

        # 创建搜索输入框和按钮
        search_input_frame = ttk.Frame(self.search_frame)
        search_input_frame.pack(side=tk.LEFT, padx=1, fill=tk.X, expand=True)

        # 搜索输入框
        self.search_var = tk.StringVar()
        self.search_entry = ttk.Entry(search_input_frame, textvariable=self.search_var, width=15)
        self.search_entry.pack(side=tk.LEFT, padx=1)

        # 搜索按钮
        ttk.Button(search_input_frame, text="搜索", width=6,
                   command=lambda: callbacks['search']() if 'search' in callbacks else None).pack(side=tk.LEFT, padx=1)

        # 清除按钮
        ttk.Button(search_input_frame, text="清除", width=6,
                   command=lambda: callbacks['clear_search']() if 'clear_search' in callbacks else None).pack(
            side=tk.LEFT, padx=1)

        # 绑定回车键到搜索功能
        self.search_entry.bind('<Return>', lambda e: callbacks['search']() if 'search' in callbacks else None)

    def _on_size_change(self):
        """大小改变时的回调"""
        if 'update_sizes' in self.callbacks:
            self.callbacks['update_sizes']()

    def get_layout(self):
        """获取当前布局类型"""
        return self.layout_var.get()

    def get_zoom_value(self):
        """获取缩放值（百分比）"""
        try:
            # 移除百分号并转换为整数
            value = self.zoom_var.get().strip('%')
            logger.debug(f"[ToolBar.get_zoom_value] 获取值: {value}%")
            return int(value)
        except ValueError as e:
            logger.debug(f"[ToolBar.get_zoom_value] 错误: {e}")
            self.zoom_var.set("100%")
            return 100

    def get_edge_width_scale(self):
        """获取边宽度缩放值"""
        return self.edge_scale.get()

    def get_node_size_scale(self):
        """获取节点大小缩放值"""
        return self.node_scale.get()

    def get_label_state(self):
        """获取标签状态"""
        return self.label_var.get()

    def get_search_text(self):
        """获取搜索文本"""
        return self.search_var.get().strip()

    def clear_search_text(self):
        """清除搜索文本"""
        self.search_var.set("")

    def _validate_and_apply_zoom(self, event=None):
        """验证并应用缩放值"""
        try:
            value = self.get_zoom_value()
            logger.debug(f"[ToolBar._validate_and_apply_zoom] 输入值: {value}%")
            if not (20 <= value <= 500):
                raise ValueError("缩放值必须在20%到500%之间")
            
            # 计算与当前值的差值
            current = self.get_zoom_value()
            delta = value - current
            logger.debug(f"[ToolBar._validate_and_apply_zoom] 计算差值: {delta}%")
            
            # 调用zoom回调
            if 'zoom' in self.callbacks:
                logger.debug(f"[ToolBar._validate_and_apply_zoom] 调用zoom回调，传入差值: {delta}")
                self.callbacks['zoom'](delta)
                
        except ValueError as e:
            logger.debug(f"[ToolBar._validate_and_apply_zoom] 错误: {e}")
            messagebox.showerror("错误", str(e))
            self.zoom_var.set("100%")

    def _adjust_zoom(self, delta):
        """调整缩放值"""
        try:
            current = self.get_zoom_value()
            logger.debug(f"[ToolBar._adjust_zoom] 当前缩放值: {current}%")
            new_value = max(20, min(500, current + delta))
            logger.debug(f"[ToolBar._adjust_zoom] 计算新值: {new_value}% (delta={delta})")
            
            # 先更新显示值
            self.zoom_var.set(f"{new_value}%")
            logger.debug(f"[ToolBar._adjust_zoom] 更新显示值: {self.zoom_var.get()}")
            
            # 直接调用zoom_entry回调，传入新值
            if 'zoom_entry' in self.callbacks:
                logger.debug(f"[ToolBar._adjust_zoom] 调用zoom_entry回调，传入值: {new_value}")
                self.callbacks['zoom_entry'](new_value)
                
        except ValueError as e:
            logger.debug(f"[ToolBar._adjust_zoom] 错误: {e}")
            self.zoom_var.set("100%")
