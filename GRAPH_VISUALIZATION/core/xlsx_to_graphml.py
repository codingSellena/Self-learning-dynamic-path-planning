import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import networkx as nx


class Excel2GraphML:
    def __init__(self, root):
        self.root = root
        self.root.title("Excel 转 GraphML")
        self.create_widgets()

    def create_widgets(self):
        # 输入文件选择框
        ttk.Label(self.root, text="输入文件路径:").grid(row=0, column=0, padx=10, pady=10, sticky='w')
        self.input_entry = ttk.Entry(self.root, width=50)
        self.input_entry.grid(row=0, column=1, padx=10, pady=10)
        input_button = ttk.Button(self.root, text="选择文件", command=self.select_input_file)
        input_button.grid(row=0, column=2, padx=10, pady=10)

        # 输出文件选择框
        ttk.Label(self.root, text="输出文件路径:").grid(row=1, column=0, padx=10, pady=10, sticky='w')
        self.output_entry = ttk.Entry(self.root, width=50)
        self.output_entry.grid(row=1, column=1, padx=10, pady=10)
        output_button = ttk.Button(self.root, text="选择文件", command=self.select_output_file)
        output_button.grid(row=1, column=2, padx=10, pady=10)

        # 转换按钮
        convert_button = ttk.Button(self.root, text="转换", command=self.convert_to_graphml)
        convert_button.grid(row=2, column=1, padx=10, pady=20)

    def select_input_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")])
        self.input_entry.delete(0, tk.END)
        self.input_entry.insert(0, file_path)

    def select_output_file(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".graphml",
                                                 filetypes=[("GraphML files", "*.graphml"), ("All files", "*.*")])
        self.output_entry.delete(0, tk.END)
        self.output_entry.insert(0, file_path)

    def read_excel(self, file_path):
        """读取Excel文件并返回数据框"""
        df = pd.read_excel(file_path,
                           dtype={'glgx': str, 'pointtype': str, 'length': float, 'area': float, 'lc': int, 'x': float,
                                  'y': float, 'id': int})
        df.replace('<空>', '0', inplace=True)  # 将内容为"0"的单元格设置为字符串"0"
        df = df.fillna(0)  # 将内容为空的单元格设置为0
        return df

    def parse_connections(self, glgx):
        """解析glgx列并返回连接关系列表"""
        connections = []
        if glgx and glgx != '0':
            # 使用正则表达式来处理逗号或点号分隔的连接关系
            import re
            connections = re.split(r'[,.]', glgx)
            connections = [conn.strip() for conn in connections if conn.strip()]  # 移除空格和空字符串

        return connections

    def convert_to_graphml(self):
        """将Excel文件转换为GraphML文件"""
        input_file = self.input_entry.get()
        output_file = self.output_entry.get()

        if not input_file or not output_file:
            messagebox.showerror("错误", "请提供输入和输出文件路径")
            return

        df = self.read_excel(input_file)
        G = nx.Graph()  # 使用无向图

        # 遍历数据框的每一行
        for idx, row in df.iterrows():
            node_id = row['id']  # 使用id列作为节点ID
            node_attrs = {
                'Elem_Id': row['fjguid'],
                'Entity_Type': row['pointtype'],
                'x': row['x坐标'],
                'y': row['y坐标'],
                'lc': row['lc'],
                'length': row['length'],
                'Capacity': row['area'],
                'Id': row['id'],
                'is_exit': True if row['pointtype'] == '建筑物出口' else False
                # 'Connect_Entity_Id_List': row['glgx']
            }
            G.add_node(node_id, **node_attrs)

            # 解析glgx列，添加边（连接关系）
            connections = self.parse_connections(row['glgx'])
            for target_id in connections:
                # try:
                target_id_int = int(target_id)
                if target_id_int in df['id'].values:  # 确保目标节点存在
                    G.add_edge(node_id, target_id_int)  # 在无向图中，边会自动双向连接
                    # print(f"添加边: {node_id} <-> {target_id_int}")  # 修改输出格式以表示双向连接
            #     else:
            #         # print(f"警告: 目标节点 {target_id_int} 不存在")
            # except ValueError as e:
            #     # print(f"错误: 无法将 {target_id} 转换为整数: {e}")

        # 写入GraphML文件
        nx.write_graphml(G, output_file)
        messagebox.showinfo("成功", f"GraphML文件已保存到 {output_file}")
        self.root.destroy()  # 转换完成后关闭窗口


def main():
    root = tk.Tk()
    app = Excel2GraphML(root)
    root.mainloop()


if __name__ == "__main__":
    main()
