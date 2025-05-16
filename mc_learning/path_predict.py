import os

import networkx as nx
import torch
from mc_learning.train4 import Net
from generate_paths import select_emergency_areas, set_emergency_area, encode_input


class PathPredictor:
    def __init__(self, model, threshold=0, graph=None, device=None, weight_path=None):
        """
        初始化路径预测器
        Args:
            model (Net): 训练好的BP神经网络模型
            threshold (float): 输出值阈值，用于选择最佳下一步
            graph (nx.Graph): 图数据
            device (torch.device): 指定设备
        """
        self.weights_path = weight_path
        self.model = model
        self.threshold = threshold
        self.graph = graph
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.node_list = list(self.graph.nodes)  # 保持顺序一致
        self.node_to_index = {node: i for i, node in enumerate(self.node_list)}
        self.index_to_node = {i: node for i, node in enumerate(self.node_list)}

    def predict_path(self, input_vector, exit_entities, max_steps=100, use_neighbors=True):
        """
        Predict the evacuation path using the trained model.
        :param input_vector: torch.Tensor or list, the input feature vector
        :param exit_entities: list of exit node IDs
        :param max_steps: int, max steps allowed
        :param use_neighbors: bool, whether to only consider neighbor nodes
        :return: (path, is_reach_exit)
        """
        # 确保输入向量在正确的设备上
        if isinstance(input_vector, torch.Tensor):
            input_vector = input_vector.to(self.device)
        else:
            input_vector = torch.tensor(input_vector, device=self.device)

        path = []
        visited = set()
        is_reach_exit = 0

        N = (len(input_vector) - 1) // 2
        current_index = torch.argmax(input_vector[:N]).item()
        current_node = self.index_to_node[current_index]
        path.append(str(current_node))
        visited.add(current_node)

        exit_entities = set(map(str, exit_entities))

        print(f"\n起始节点: {current_node}")
        print(f"目标出口: {exit_entities}")

        with torch.no_grad():
            for step in range(max_steps):
                if current_node not in self.graph.nodes:
                    print(f"警告：节点 {current_node} 不在图中，停止预测")
                    break

                _, output_vector = self.model.forward_propagation(input_vector)
                probabilities = output_vector.cpu().numpy().flatten()

                next_node = None
                max_prob = -1

                if use_neighbors:
                    neighbors = sorted(list(self.graph.neighbors(current_node)))
                    if not neighbors:
                        print(f"警告：节点 {current_node} 没有邻居，停止预测")
                        break

                    for node_id in neighbors:
                        try:
                            idx = self.node_to_index[node_id]
                            prob = probabilities[idx]
                            if prob > max_prob and node_id not in visited:
                                max_prob = prob
                                next_node = node_id
                        except KeyError:
                            continue

                else:
                    max_index = probabilities.argmax()
                    max_prob = probabilities[max_index]
                    next_node = self.index_to_node[max_index]

                if next_node is None:
                    print("没有可选的下一个节点，停止预测")
                    break

                if max_prob < self.threshold:
                    print(f"概率 {max_prob:.6f} 低于阈值 {self.threshold}，预测停止")
                    break

                path.append(str(next_node))
                visited.add(next_node)

                if next_node in exit_entities:
                    print(f"到达出口实体: {next_node}")
                    is_reach_exit = 1
                    break

                input_vector = self.update_input_vector(input_vector, next_node, N=N, device=self.device)
                current_node = next_node

            print("\n最终路径:")
            print(" -> ".join(map(str, path)))
            return path, is_reach_exit

    def update_input_vector(self, prev_input, current_node, N, device):
        """
        更新输入向量（input_vector），用于模型的下一步预测。
        输入向量结构说明（长度为 2N+1）：
        - 前 N 维：表示当前节点的 one-hot 编码（哪个位置是当前节点）
        - 后 N 维：紧急区域
        - 最后 1 维：偏置项1

        参数:
        - prev_input: 上一步的输入向量（tensor）
        - current_node: 也就是模型选的下一个结点
        - N: 节点总数
        - device: 模型所在设备 (CPU / CUDA)

        返回:
        - 更新后的输入向量，类型为 torch.Tensor
        """
        # 创建一个与原输入相同形状的全零向量
        input_vector = torch.zeros_like(prev_input, device=device)

        # 获取当前节点在节点列表中的索引位置
        idx = self.node_to_index[current_node]

        # 设置 one-hot 编码：当前节点位置置为 1
        input_vector[idx] = 1.0

        # 保留目标出口信息（或上下文）：将原向量的后半部分复制过来
        input_vector[N:] = prev_input[N:]

        input_vector[-1] = 1

        return input_vector

    def load_weights(self):
        """如果存在已训练权重，则加载"""
        if self.weights_path and os.path.exists(self.weights_path):
            print("检测到已保存的权重文件，正在加载权重...")
            checkpoint = torch.load(self.weights_path)
            self.H = checkpoint['H']
            self.U = checkpoint['U']
            print("✅ 权重加载成功，继续训练")

    def compare_two_models(self, path1: list[str], path2: list[str]):
        if (len(path1) > len(path2)):
            print(f"path1: {path1}, path2: {path2}")
            return
        elif len(path1) < len(path2):
            print(f"path1: {path1}, path2: {path2}")
            return
        lens = len(path1)
        has_difference = False

        for i in range(lens):
            if path1[i] != path2[i]:
                print(f"path1:{path1} vs path2:{path2}")
                return

        if not has_difference:
            print("两个路径完全一致。")

        return


if __name__ == "__main__":
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载图数据
    graph = nx.read_graphml("D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\test03.graphml")
    print(f"图节点数量: {len(graph.nodes)}")

    # 加载训练好的模型权重
    weight_path = "self_iters_5.pt"  # "full_vector_best_model_0415.pt"
    checkpoint = torch.load(weight_path, map_location=device)

    # 打印模型参数信息
    print("\n模型参数信息:")
    print(f"隐藏层权重 H 的形状: {checkpoint['H'].shape}")
    print(f"输出层权重 U 的形状: {checkpoint['U'].shape}")

    # 根据权重矩阵形状确定模型参数
    input_size = checkpoint['H'].shape[1]  # 输入大小 = H的列数
    hidden_size = checkpoint['H'].shape[0]  # 隐藏层大小 = H的行数
    output_size = checkpoint['U'].shape[0]  # 输出大小 = U的行数

    # 初始化模型
    learning_rate = 0.01
    momentum = 0.5
    trained_model = Net(input_size, hidden_size, output_size, learning_rate,
                        momentum)  # ,weights_path="full_vector_best_model_0408.pt"
    trained_model.H = checkpoint['H'].to(device)
    trained_model.U = checkpoint['U'].to(device)

    # 创建输入向量
    n = len(graph.nodes)

    node_list = list(graph.nodes)

    # 定义出口和要避开的数字
    exit_entities = {"63", "102", "155", "135", "123"}
    avoid_numbers = {"27", "104", "105", "121", "122", "129", "151", "152", "469", "478", "542"}

    # 将出口和要避开的数字合并为一个集合，提高查询效率
    all_excluded_numbers = exit_entities.union(avoid_numbers)

    # 初始化路径预测器
    path_predictor = PathPredictor(trained_model, threshold=0.01, graph=graph, device=device)

    #
    # 加载训练好的模型权重
    weight_path = "self_iters_4.pt"  # "full_vector_best_model_0415.pt"
    checkpoint = torch.load(weight_path, map_location=device)

    # 打印模型参数信息
    print("\n模型参数信息:")
    print(f"隐藏层权重 H 的形状: {checkpoint['H'].shape}")
    print(f"输出层权重 U 的形状: {checkpoint['U'].shape}")

    # 根据权重矩阵形状确定模型参数
    input_size = checkpoint['H'].shape[1]  # 输入大小 = H的列数
    hidden_size = checkpoint['H'].shape[0]  # 隐藏层大小 = H的行数
    output_size = checkpoint['U'].shape[0]  # 输出大小 = U的行数

    # 初始化模型
    learning_rate = 0.01
    momentum = 0.5
    trained_model = Net(input_size, hidden_size, output_size, learning_rate,
                        momentum)  # ,weights_path="full_vector_best_model_0408.pt"
    trained_model.H = checkpoint['H'].to(device)
    trained_model.U = checkpoint['U'].to(device)
    path_predictor2 = PathPredictor(trained_model, threshold=0.01, graph=graph, device=device)

    # 清空CUDA缓存
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    set_emergency_area(graph)
    emergency_area = ['31', '361', '610', '496', '25', '18', '477', '465', '254', '386']
    print(emergency_area)
    for start_node in node_list:
        if str(start_node) in all_excluded_numbers:
            print(f"no available {start_node}")
            continue

        if start_node not in graph.nodes:
            print(f"警告：节点 {start_node} 不在图中，停止预测")
            continue

        input_vector = encode_input(graph, start_node, emergency_area)

        # emergency_area = node_list.index('13')
        # input_vector2[n + emergency_area] = 1.0
        # emergency_area = node_list.index('602')
        # input_vector2[n + emergency_area] = 1.0
        # emergency_area = node_list.index('213')
        # input_vector2[n + emergency_area] = 1.0
        # 执行预测
        predicted_path, is_reach_exit = path_predictor.predict_path(input_vector, exit_entities)
        predicted_path2, is_reach_exit2 = path_predictor2.predict_path(input_vector, exit_entities)
        path_predictor2.compare_two_models(predicted_path, predicted_path2)
        if is_reach_exit2 == 0:
            print(f"节点 {start_node} 无法到达出口")
