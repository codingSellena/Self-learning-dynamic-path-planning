import tkinter as tk
from tkinter import ttk, messagebox
from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()

class NodeAttributePanel:
    def __init__(self, parent, graph_manager):
        self.parent = parent
        self.graph_manager = graph_manager
        self.current_node = None # type:dict, ['Id']

        # 定义不可编辑的属性
        self.readonly_attributes = {'Id', 'Connect_Entity_Id_List'}  # 数据库ID不可编辑

        # 创建右侧面板框架
        self.frame = ttk.Frame(parent)
        self.frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.E, tk.W), padx=5, pady=5)

        # 创建标题标签
        self.title_label = ttk.Label(self.frame, text="节点属性", font=('SimHei', 12, 'bold'))
        self.title_label.grid(row=0, column=0, columnspan=2, pady=5)

        # 创建属性编辑区域
        self.create_attribute_editors()

        # 创建分隔线
        separator = ttk.Separator(self.frame, orient='horizontal')
        separator.grid(row=999, column=0, columnspan=2, sticky='ew', pady=10)

        # 创建节点过滤模块
        self.create_filter_module()

        # 存储楼层过滤状态
        self.floor_filters = {}

        # 配置网格列权重
        self.frame.grid_columnconfigure(1, weight=1)


        logger.info("NodeAttributePanel initialized successfully")

    def create_coordinate_editor(self, parent, default_value=(0.0, 0.0, 0.0)):
        """创建坐标编辑器"""
        frame = ttk.Frame(parent)

        # 左括号标签
        left_bracket = ttk.Label(frame, text="(")
        left_bracket.pack(side=tk.LEFT)

        # X坐标输入框
        x_entry = ttk.Entry(frame, width=6)
        x_entry.pack(side=tk.LEFT, padx=1)
        x_entry.insert(0, str(default_value[0]))

        # 第一个逗号
        comma1 = ttk.Label(frame, text=",")
        comma1.pack(side=tk.LEFT)

        # Y坐标输入框
        y_entry = ttk.Entry(frame, width=6)
        y_entry.pack(side=tk.LEFT, padx=1)
        y_entry.insert(0, str(default_value[1]))

        # 第二个逗号
        comma2 = ttk.Label(frame, text=",")
        comma2.pack(side=tk.LEFT)

        # Z坐标输入框
        z_entry = ttk.Entry(frame, width=6)
        z_entry.pack(side=tk.LEFT, padx=1)
        z_entry.insert(0, str(default_value[2]))

        # 右括号标签
        right_bracket = ttk.Label(frame, text=")")
        right_bracket.pack(side=tk.LEFT)

        return frame, (x_entry, y_entry, z_entry)

    def create_attribute_editors(self):
        """创建所有属性的编辑器"""
        # 清除现有的编辑器
        if hasattr(self, 'editors'):
            for widget in self.frame.winfo_children():
                if widget != self.title_label and (not hasattr(self, 'update_button') or widget != self.update_button):
                    widget.destroy()
        
        self.editors = {}
        row = 1

        # 创建基础属性的编辑器
        for attr, info in self.graph_manager.base_attributes.items():
            if attr == 'node_id':  # 跳过节点标识符
                continue

            # 创建属性标签
            label = ttk.Label(self.frame, text=f"{attr}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            # 根据属性类型创建不同的编辑器
            if attr in self.readonly_attributes:
                # 只读属性使用只读的Entry
                editor = ttk.Entry(self.frame)
                editor.insert(0, str(info['default']))
                editor.configure(state='readonly')
                editor.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = editor
            elif attr == 'Center':
                # 为Center属性创建特殊的坐标编辑器
                coord_frame, coord_entries = self.create_coordinate_editor(
                    self.frame,
                    info['default']
                )
                coord_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = coord_entries
            elif attr == 'is_exit':
                # 为is_exit创建复选框
                var = tk.BooleanVar(value=info['default'])
                editor = ttk.Checkbutton(
                    self.frame,
                    variable=var,
                    command=lambda: self.on_exit_changed(var)
                )
                editor.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.editors[attr] = (editor, var)  # 保存复选框和变量
            elif info['type'] == 'str' and 'options' in info:
                editor = ttk.Combobox(self.frame, values=info['options'])
                editor.set(info['default'])
                editor.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = editor
            else:
                editor = ttk.Entry(self.frame)
                editor.insert(0, str(info['default']))
                editor.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = editor

            row += 1

        # 创建自定义属性的编辑器
        for attr, info in self.graph_manager.custom_node_attributes.items():
            # 创建属性标签
            label = ttk.Label(self.frame, text=f"{attr}:")
            label.grid(row=row, column=0, sticky=tk.W, padx=5, pady=2)

            # 根据属性类型创建不同的编辑器
            if info['type'] == str and 'options' in info:
                # 如果属性类型为str且有options，创建下拉框
                editor = ttk.Combobox(self.frame, values=info['options'])
                editor.set(info['default'])
                editor.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = editor
            elif info['type'] == 'bool':
                # 为布尔值创建复选框
                var = tk.BooleanVar(value=info['default'])
                editor = ttk.Checkbutton(
                    self.frame,
                    variable=var
                )
                editor.grid(row=row, column=1, sticky=tk.W, padx=5, pady=2)
                self.editors[attr] = (editor, var)
            elif info['type'] == 'int' or info['type'] == 'double':
                # 整数或浮点数类型使用Entry
                editor = ttk.Entry(self.frame)
                editor.insert(0, str(info['default']))
                editor.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = editor
            else:
                editor = ttk.Entry(self.frame)
                editor.insert(0, str(info['default']))
                editor.grid(row=row, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
                self.editors[attr] = editor

            row += 1

        # 创建或重新定位更新按钮
        if hasattr(self, 'update_button'):
            self.update_button.grid(row=row, column=0, columnspan=2, pady=10)
        else:
            # 创建更新按钮
            self.update_button = ttk.Button(
                self.frame,
                text="更新属性",
                command=self.update_attributes
            )
            self.update_button.grid(row=row, column=0, columnspan=2, pady=10)
            # 初始状态下禁用所有输入

        self.set_editors_state('disabled')

    def create_filter_module(self):
        """创建节点过滤模块"""
        # 创建过滤模块框架
        filter_frame = ttk.LabelFrame(self.frame, text="节点过滤")
        filter_frame.grid(row=1000, column=0, columnspan=2, sticky='ew', padx=5, pady=5)

        # 创建标题
        filter_label = ttk.Label(filter_frame, text="按楼层过滤:")
        filter_label.pack(anchor='w', padx=5, pady=2)

        # 获取所有可能的楼层值
        floors = set()
        if self.graph_manager.graph:
            for node in self.graph_manager.graph.nodes():
                floor = self.graph_manager.graph.nodes[node].get('lc')
                if floor is not None:
                    try:
                        floor_num = int(float(floor))
                        floors.add(floor_num)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"节点 {node} 的楼层值 '{floor}' 无效: {e}")
                        continue

        # 如果没有找到有效的楼层，记录警告
        if not floors:
            logger.warning("没有找到有效的楼层值")
            return

        # 初始化楼层过滤状态
        if not hasattr(self, 'floor_filters'):
            self.floor_filters = {}
            # 为每个楼层创建默认的过滤状态（默认可见）
            for floor in sorted(floors):
                self.floor_filters[floor] = tk.BooleanVar(value=True)

        # 创建楼层过滤复选框
        for floor in sorted(floors):
            # 如果该楼层还没有过滤状态，创建一个新的
            if floor not in self.floor_filters:
                self.floor_filters[floor] = tk.BooleanVar(value=True)

            # 创建复选框
            cb = ttk.Checkbutton(
                filter_frame,
                text=f"第 {floor} 层",
                variable=self.floor_filters[floor],
                command=lambda f=floor: self.on_floor_filter_change(f)
            )
            cb.pack(anchor='w', padx=20, pady=2)

        logger.info(f"创建了 {len(floors)} 个楼层过滤器")

    def on_floor_filter_change(self, floor):
        """处理楼层过滤变化"""
        # 触发重绘事件
        self.parent.event_generate('<<NodeAttributesUpdated>>')
        logger.debug("Generated NodeAttributesUpdated event")
        
        # 通知主窗口更新显示
        if hasattr(self.graph_manager, 'visualizer') and self.graph_manager.visualizer:
            # 获取当前的楼层过滤状态
            floor_filters = getattr(self.graph_manager.visualizer.node_panel, 'floor_filters', {})
            self.graph_manager.update_visible_nodes_cache(floor_filters)
            self.graph_manager.visualizer.update_display()
        else:
            logger.warning("Visualizer not accessible, display update may be delayed")

    def refresh_filter_module(self):
        """刷新过滤模块"""
        # 清除现有的过滤控件
        for widget in self.frame.winfo_children():
            if isinstance(widget, ttk.LabelFrame) and widget.winfo_children()[0].cget("text") == "按楼层过滤:":
                widget.destroy()
        
        # 重新创建过滤模块
        self.create_filter_module()

    def get_unselected_floors(self):
        """获取当前未被勾选的楼层"""
        unselected_floors = [floor for floor, var in self.floor_filters.items() if not var.get()]
        return unselected_floors

    def on_exit_changed(self, var):
        """处理出口状态改变"""
        if not self.current_node:
            logger.warning("No node selected when exit status changed")
            return

        is_exit = var.get()
        logger.info(f"Exit status changed for node {self.current_node} to {is_exit}")

        # 更新属性
        new_attrs = {'is_exit': is_exit}
        if self.graph_manager.set_node_info(str(self.current_node['Id']), new_attrs):
            logger.info(f"Successfully updated exit status for node {self.current_node}")
            # 触发重绘事件
            self.parent.event_generate('<<NodeAttributesUpdated>>')
            logger.debug("Generated NodeAttributesUpdated event")
        else:
            logger.error(f"Failed to update exit status for node {self.current_node}")

    def set_editors_state(self, state):
        """设置所有编辑器的状态"""
        for attr, editor in self.editors.items():
            if attr in self.readonly_attributes:
                if isinstance(editor, ttk.Entry):
                    editor.configure(state='readonly')
            elif attr == 'Center':
                for entry in editor:
                    entry.configure(state=state)
            elif attr == 'is_exit':
                editor[0].configure(state=state)
            else:
                editor.configure(state=state)
        self.update_button.configure(state=state)

    def display_node_attributes(self, node):
        """显示节点的属性
        Args:
            node: 节点ID或节点属性字典
        """
        # logger.info(f"display_node_attributes called with node: {node}")
        
        self.current_node = node
        logger.debug(f"Set current_node to: {self.current_node}")
        
        if node is None:
            logger.warning("Node is None, disabling editors")
            self.set_editors_state('disabled')
            return

        # 获取节点信息
        if isinstance(node, dict):
            # 如果传入的是节点属性字典，直接使用
            # logger.info("Node is a dictionary, using it directly")
            node_info = node
            # 尝试从字典中获取节点ID
            node_id = node.get('Id', None)
        else:
            # 如果传入的是节点ID，从图中获取信息
            # logger.info(f"Node is an ID: {node}, getting info from graph")
            node_info = self.graph_manager.get_node_info(node)
            
        logger.debug(f"Node info: {node_info}")
        
        if not node_info:
            logger.warning("Failed to get node info, returning")
            return

        # 检查编辑器是否已创建
        if not hasattr(self, 'editors') or not self.editors:
            logger.warning("Editors not created yet, creating them now")
            self.create_attribute_editors()
            
        logger.debug(f"Available editors: {list(self.editors.keys())}")

        # 更新每个编辑器的值
        for attr, editor in self.editors.items():
            logger.debug(f"Updating editor for attribute: {attr}")
            
            # 检查属性是否在 base_attributes 或 custom_node_attributes 中
            if attr in self.graph_manager.base_attributes:
                default_value = self.graph_manager.base_attributes[attr]['default']
            elif attr in self.graph_manager.custom_node_attributes:
                default_value = self.graph_manager.custom_node_attributes[attr]['default']
            else:
                logger.warning(f"Attribute {attr} not found in base or custom attributes, skipping")
                continue

            value = node_info.get(attr, default_value)
            logger.debug(f"Value for {attr}: {value} (default: {default_value})")
            
            try:
                if attr == 'Center':
                    # 更新坐标输入框的值
                    if isinstance(value, str):
                        # 如果是字符串，解析为元组
                        try:
                            value = eval(value)
                            logger.debug(f"Parsed Center string to tuple: {value}")
                        except Exception as e:
                            logger.error(f"Error parsing Center value: {e}")
                            value = (0.0, 0.0, 0.0)
                    for i, entry in enumerate(editor):
                        entry.delete(0, tk.END)
                        entry.insert(0, str(value[i]))
                        logger.debug(f"Set Center[{i}] to {value[i]}")
                elif attr == 'is_exit':
                    editor[1].set(value)  # 设置复选框状态
                    logger.debug(f"Set is_exit checkbox to {value}")
                elif isinstance(editor, ttk.Combobox):
                    editor.set(value)
                    logger.debug(f"Set combobox {attr} to {value}")
                else:
                    editor.configure(state='normal')
                    editor.delete(0, tk.END)
                    editor.insert(0, str(value))
                    logger.debug(f"Set text entry {attr} to {value}")
                    if attr in self.readonly_attributes:
                        editor.configure(state='readonly')
                        logger.debug(f"Set {attr} to readonly")
            except Exception as e:
                logger.error(f"Error updating editor for {attr}: {e}")

        # 启用可编辑的编辑器
        # logger.info("Enabling editors")
        self.set_editors_state('normal')

    def update_attributes(self):
        """更新节点属性"""
        if not self.current_node:
            logger.warning("Cannot update attributes: no node selected")
            messagebox.showwarning("警告", "请先选择一个节点")
            return
            
        # 保存当前选中节点的颜色
        current_node_id = str(self.current_node['Id'])
        # 从 ColorHandler 中获取当前节点的颜色
        current_color = self.graph_manager.visualizer.color_handler.get_node_color(current_node_id)
        
        logger.info(f"Starting attribute update for node {self.current_node}")
        new_attrs = {}

        # 处理只读属性
        for attr in self.readonly_attributes:
            if attr in self.editors:
                logger.debug(f"Skipping readonly attribute: {attr}")
                continue

        # 处理可编辑属性
        for attr, editor in self.editors.items():
            if attr in self.readonly_attributes:
                continue

            try:
                if attr == 'Center':
                    try:
                        x = float(editor[0].get())
                        y = float(editor[1].get())
                        z = float(editor[2].get())
                        new_attrs['x'] = x
                        new_attrs['y'] = y
                        new_attrs['lc'] = z
                        new_attrs['Center'] = (x, y, z)
                        logger.debug(f"Updated coordinates: x={x}, y={y}, z={z}")
                    except ValueError:
                        logger.error("Invalid coordinate values")
                        messagebox.showerror("错误", "坐标值必须是数字")
                        return
                elif attr == 'is_exit':
                    value = editor[1].get()
                    new_attrs[attr] = value
                    logger.debug(f"Updated is_exit attribute: {value}")
                else:
                    if attr in self.graph_manager.base_attributes:
                        attr_info = self.graph_manager.base_attributes[attr]
                        logger.debug(f"Processing base attribute {attr} with info: {attr_info}")
                    elif attr in self.graph_manager.custom_node_attributes:
                        attr_info = self.graph_manager.custom_node_attributes[attr]
                        logger.debug(f"Processing custom attribute {attr} with info: {attr_info}")
                    else:
                        logger.warning(f"Attribute {attr} not found in base or custom attributes")
                        continue

                    value = editor.get()
                    logger.debug(f"Raw value for {attr}: {value}")

                    try:
                        if attr_info['type'] == 'int':
                            value = int(value)
                        elif attr_info['type'] == 'double':
                            value = float(value)
                        elif attr_info['type'] == 'list':
                            if isinstance(value, str):
                                value = value.split(',')
                        elif attr_info['type'] == 'tuple':
                            if isinstance(value, str):
                                value = eval(value)
                        elif attr_info['type'] == 'str' and 'options' in attr_info:
                            if value not in attr_info['options']:
                                logger.error(f"Invalid option '{value}' for {attr}. Valid options: {attr_info['options']}")
                                messagebox.showerror("错误", f"'{attr}' 的值必须是以下选项之一：{', '.join(attr_info['options'])}")
                                return
                        new_attrs[attr] = value
                        logger.debug(f"Processed value for {attr}: {value}")
                    except ValueError as e:
                        logger.error(f"Invalid value for attribute {attr}: {str(e)}")
                        messagebox.showerror("错误", f"属性 {attr} 的值格式不正确")
                        return
            except Exception as e:
                logger.error(f"Error processing attribute {attr}: {str(e)}")
                messagebox.showerror("错误", f"处理属性 {attr} 时出错")
                return

        # 更新属性
        logger.info(f"Attempting to update node {self.current_node} with attributes: {new_attrs}")
        if self.graph_manager.set_node_info(current_node_id, new_attrs):
            # 恢复节点颜色
            if current_color:
                self.graph_manager.visualizer.color_handler.node_colors[current_node_id] = current_color
                logger.debug(f"Restored node color: {current_color} for node {current_node_id}")
            
            logger.info("Node attributes updated successfully")
            messagebox.showinfo("成功", "属性已更新，请记得保存文件")
            
            # 清除样式缓存并更新显示
            self.graph_manager.visualizer.clear_style_cache()
            self.parent.event_generate('<<NodeAttributesUpdated>>')
            logger.debug("Generated NodeAttributesUpdated event")
        else:
            logger.error("Failed to update node attributes")
            messagebox.showerror("错误", "更新属性失败")

