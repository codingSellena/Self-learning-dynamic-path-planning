import tkinter as tk
from tkinter import ttk, messagebox


class AddAttributeDialog:
    """添加自定义属性的对话框"""

    def __init__(self, parent, graph_manager, node_panel):
        self.parent = parent
        self.graph_manager = graph_manager
        self.node_panel = node_panel
        self.list_items = []  # 存储 自定义选项 的值
        self.dialog = None

    def show_dialog(self):
        """弹出窗口添加自定义节点属性"""
        if not self.graph_manager.graph:
            messagebox.showerror("错误", "没有图信息，请先加载图形", parent=self.parent)
            return

        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("添加节点属性")
        self.dialog.geometry("400x300")
        self.dialog.grab_set()

        input_frame = ttk.Frame(self.dialog)
        input_frame.pack(pady=10, padx=10, fill='both', expand=True)

        input_frame.columnconfigure(0, weight=1)  # 第一列较窄
        input_frame.columnconfigure(1, weight=2)  # 第二列较宽

        # 属性名称输入
        ttk.Label(input_frame, text="属性名称:").grid(row=0, column=0, sticky='w', pady=5)
        self.name_entry = ttk.Entry(input_frame)
        self.name_entry.grid(row=0, column=1, sticky='ew', pady=5)

        # 类型选择
        ttk.Label(input_frame, text="属性类型:").grid(row=1, column=0, sticky='w', pady=5)
        self.type_combo = ttk.Combobox(
            input_frame,
            values=['int', 'double', 'str', 'list', 'bool', '自定义选项'],
            state='readonly'
        )
        self.type_combo.grid(row=1, column=1, sticky='ew', pady=5)
        self.type_combo.current(0)

        # 默认值输入（普通类型）
        ttk.Label(input_frame, text="默认值:").grid(row=2, column=0, sticky='w', pady=5)
        self.default_entry = ttk.Entry(input_frame)
        self.default_entry.grid(row=2, column=1, sticky='ew', pady=5)

        # 默认值下拉框
        self.default_combo = ttk.Combobox(input_frame, state="readonly")
        self.default_combo.grid(row=2, column=1, sticky='ew', pady=5)
        self.default_combo.grid_remove()  # 初始隐藏

        # 动态区域
        self.dynamic_frame = ttk.Frame(self.dialog)

        self.type_combo.bind("<<ComboboxSelected>>", self.update_dynamic_area)
        self.create_dynamic_list_ui = self.create_dynamic_list_ui

        # 按钮框架
        button_frame = ttk.Frame(self.dialog)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        ttk.Button(button_frame, text="确认", command=self.on_confirm).pack(side='left', padx=5)
        ttk.Button(button_frame, text="取消", command=self.dialog.destroy).pack(side='left', padx=5)

        self.name_entry.focus_set()
        self.dialog.bind("<Configure>", self.on_window_resize)
        self.on_window_resize()

    def on_window_resize(self, event=None):
        """窗口大小调整时的回调函数"""
        self.dialog.update_idletasks()
        self.dialog.geometry("")

    def update_dynamic_area(self, event=None):
        """更新动态输入区域"""
        attr_type = self.type_combo.get()
        if attr_type in '自定义选项':
            self.default_entry.grid_remove()  # 隐藏普通输入框
            self.default_combo.grid()  # 显示下拉框
            self.dynamic_frame.pack(pady=10, padx=10, fill='both', expand=True)
            self.create_dynamic_list_ui(self.dynamic_frame)
            self.on_window_resize()
        else:
            self.default_combo.grid_remove()  # 隐藏下拉框
            self.default_entry.grid()  # 显示普通输入框
            self.dynamic_frame.pack_forget()
            self.on_window_resize()

    def create_dynamic_list_ui(self, frame):
        """创建动态输入区域（表格）"""
        for widget in frame.winfo_children():
            widget.destroy()  # 清空旧内容

        # 列表显示区域
        listbox = tk.Listbox(frame, height=5)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        list_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        listbox.config(yscrollcommand=list_scroll.set)

        # 输入框和按钮
        entry_var = tk.StringVar()
        entry_box = ttk.Entry(frame, textvariable=entry_var)
        entry_box.pack(pady=5, padx=5, fill='x')

        def add_item():
            """添加元素"""
            item = entry_var.get().strip()
            if item:
                self.list_items.append(item)
                listbox.insert(tk.END, item)
                self.default_combo['values'] = self.list_items  # 更新下拉框选项
                if not self.default_combo.get():
                    self.default_combo.current(0)  # 默认选择第一个选项
                entry_var.set("")  # 清空输入框

        def remove_item():
            """删除选中的元素"""
            try:
                selected_index = listbox.curselection()[0]
                listbox.delete(selected_index)
                del self.list_items[selected_index]
                self.default_combo['values'] = self.list_items  # 更新下拉框选项
                if self.list_items:
                    self.default_combo.current(0)  # 默认选择第一个选项
                else:
                    self.default_combo.set("")  # 清空下拉框
            except IndexError:
                messagebox.showwarning("提示", "请先选择要删除的元素")

        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=5, fill='x')

        ttk.Button(button_frame, text="+", command=add_item).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="-", command=remove_item).pack(side=tk.LEFT, padx=5)

    def validate_inputs(self):
        """验证输入数据"""
        name = self.name_entry.get().strip()
        attr_type = self.type_combo.get()

        if not name:
            messagebox.showerror("错误", "属性名称不能为空", parent=self.dialog)
            return False

        if attr_type == '自定义选项':
            if not self.list_items:
                messagebox.showerror("错误", "请至少添加一个选项", parent=self.dialog)
                return False
            default_value = self.default_combo.get()  # 从下拉框获取默认值
            if not default_value:
                default_value = self.list_items[0]  # 默认选择第一个选项
            # default_value = tuple(self.list_items) if attr_type == 'tuple' else self.list_items
        else:
            default_value = self.default_entry.get().strip()

        try:
            converters = {
                'int': int,
                'double': float,
                'str': str,
                'bool': lambda x: x.lower() in ('true', '1'),
                'list': list,
                '自定义选项': str
            }
            converted_value = converters[attr_type](default_value)
            return name, attr_type, converted_value
        except Exception as e:
            messagebox.showerror("错误", f"默认值无法转换为 {attr_type} 类型: {str(e)}", parent=self.dialog)
            return False

    def on_confirm(self):
        """确认添加属性"""
        validated = self.validate_inputs()
        if validated:
            name, attr_type, default = validated
            if self.graph_manager.add_custom_node_attribute(name, attr_type, default, self.list_items):
                messagebox.showinfo("成功", f"属性 {name} 已添加，记得保存文件", parent=self.dialog)
                self.node_panel.create_attribute_editors()
                self.dialog.destroy()
            else:
                messagebox.showerror("错误", f"属性 {name} 已存在", parent=self.dialog)


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口


    # 模拟 GraphManager 和 NodePanel
    class GraphManager:
        def __init__(self):
            self.graph = True  # 模拟图存在

        def add_custom_node_attribute(self, name, attr_type, default, list_items=None):
            print(f"添加属性: {name}, 类型: {attr_type}, 默认值: {default}, 选项: {list_items}")
            return True


    class NodePanel:
        def create_attribute_editors(self):
            print("更新节点属性编辑器")


    graph_manager = GraphManager()
    node_panel = NodePanel()

    dialog = AddAttributeDialog(root, graph_manager, node_panel)
    dialog.show_dialog()

    root.mainloop()
