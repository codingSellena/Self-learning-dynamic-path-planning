import networkx as nx
import torch
from mc_learning.train2 import BPNeuralNetwork, load_dataset
from GRAPH_VISUALIZATION.utils.logger_config import setup_logger

# 配置日志记录器
logger = setup_logger()


class PathPredictor:
    def __init__(self, model, threshold=0.01, graph=None):
        """
        初始化路径预测器
        Args:
            model (BPNeuralNetwork): 训练好的BP神经网络模型
            threshold (float): 输出值阈值，用于选择最佳下一步
        """
        self.model = model
        self.threshold = threshold
        self.graph = graph

    def predict_path(self, input_vector, exit_entities, max_steps=100):
        """
        使用训练好的模型进行路径预测（基于 1×4 的输出向量）。

        Args:
            input_vector (ndarray/tensor): 初始输入向量
            exit_entities (list): 所有出口实体的ID
            max_steps (int): 最大允许的预测步骤数，防止死循环
        Returns:
            list: 预测路径 (按实体ID顺序)
        """
        path = []
        visited = set()

        # 获取初始节点
        N = (len(input_vector) - 1) // 2  # 计算节点总数（减去常数项）
        current_node = torch.argmax(input_vector[:N]).item()
        path.append(current_node)
        visited.add(str(current_node))

        # 确保出口 ID 是 str
        exit_entities = set(map(str, exit_entities))
        print(f"\n起始节点: {current_node}")
        print(f"目标出口: {exit_entities}")

        with torch.no_grad():  # 预测时不计算梯度
            for step in range(max_steps):
                print(f"\n步骤 {step + 1}: 当前节点 {current_node}")

                if self.graph is None:
                    print("graph is none in self_learning_iter.predict")

                # 获取当前节点的邻居（按 ID 升序排序）
                if str(current_node) in self.graph.nodes():
                    neighbors = sorted(list(self.graph.neighbors(str(current_node))))
                else:
                    print("当前节点不在图中，停止预测")
                    break

                if not neighbors:
                    print("当前节点没有邻居，停止预测")
                    break

                # 执行前向传播
                _, output_vector = self.model.forward_propagation(input_vector)

                # 转换输出向量为 numpy 数组
                probabilities = output_vector.numpy().flatten()

                # **对度数为 1 的邻接节点，设其概率为 0**
                adjusted_probs = []
                for neighbor, prob in zip(neighbors, probabilities):
                    if self.graph.degree(str(neighbor)) == 1:
                        adjusted_probs.append(0.0)  # 设为 0
                    else:
                        adjusted_probs.append(prob)

                # **筛选有效节点**
                valid_nodes = [(n, p) for n, p in zip(neighbors, adjusted_probs) if p > 0]

                if not valid_nodes:
                    # **打印邻接节点信息**：
                    for neighbor, orig_prob, adj_prob in zip(neighbors, probabilities, adjusted_probs):
                        print(f"节点: {neighbor}, 原始概率: {orig_prob}, 调整后概率: {adj_prob}")
                    print("无有效的邻接节点，停止预测")
                    break

                # **按照调整后的概率降序排序**
                sorted_neighbors_probs = sorted(valid_nodes, key=lambda x: x[1], reverse=True)

                # 打印选择的节点
                print("\n预测概率分布:")
                print("节点ID  调整后概率  原始预测概率 是否访问 是否出口")
                print("-" * 50)
                for node_id, adj_prob in sorted_neighbors_probs[:4]:
                    is_visited = "是" if node_id in visited else "否"
                    is_exit = "是" if node_id in exit_entities else "否"
                    print(
                        f"{node_id:<6} {adj_prob:<10} {probabilities[neighbors.index(node_id)]:<10} {is_visited:<8} {is_exit:<6}")

                # 选择最大概率的邻接节点
                next_node, max_prob = sorted_neighbors_probs[0]

                print(f"\n选择下一节点: {next_node} (概率: {max_prob:.6f})")

                # 如果最大概率低于阈值，终止搜索
                if max_prob < self.threshold:
                    print(f"概率 {max_prob:.6f} 低于阈值 {self.threshold}，预测停止")
                    break

                # 记录路径
                path.append(next_node)
                visited.add(next_node)

                # **如果到达出口实体，终止搜索，并记录路径**
                if next_node in exit_entities:
                    print(f" 到达出口实体: {next_node}")

                    # **写入到 txt 文件**
                    with open("evacuation_paths.txt", "a", encoding="utf-8") as file:
                        file.write(" -> ".join(map(str, path)) + "\n")

                    print(f"路径已保存到 evacuation_paths.txt: {' -> '.join(map(str, path))}")
                    break

                # 更新当前输入向量
                input_vector = self.update_input_vector(input_vector, next_node, N)
                current_node = next_node

                # 如果所有可用邻居都已访问，终止搜索
                if all(n in visited for n, _ in valid_nodes):
                    print("所有可用邻居节点都已访问，无法继续前进")
                    break

                print(f"当前路径: {' -> '.join(map(str, path))}")
                print("-" * 50)

        print("\n最终路径:")
        print(" -> ".join(map(str, path)))
        return path

    @staticmethod
    def update_input_vector(original_vector, next_node, N):
        """
        更新输入向量以表示当前实体
        Args:
            original_vector (tensor): 原始输入向量
            next_node (int): 下一步的实体ID
            N (int): 节点数量
        Returns:
            tensor: 更新后的输入向量
        """
        # 创建新的输入向量，保持紧急区域信息和常数项不变
        new_vector = torch.zeros_like(original_vector)
        # 确保索引不会超出范围
        if int(next_node) >= len(new_vector):
            raise ValueError(f"next_node {next_node} 超出 new_vector 长度 {len(new_vector)}")

        # 复制紧急区域信息（从N到2N-1的部分）
        new_vector[N:2 * N] = original_vector[N:2 * N]
        # 设置当前节点
        new_vector[int(next_node)] = 1.0
        # 保持常数项（最后一个元素）
        new_vector[-1] = 1.0
        return new_vector


if __name__ == "__main__":
    # 加载模型
    N = 616  # 节点数量
    input_size = 2 * N + 1  # 输入层大小（N个节点状态 + N个紧急区域状态 + 1个常数项）
    hidden_size = 1500
    output_size = 4
    trained_model = BPNeuralNetwork(input_size, hidden_size, output_size)

    graph = nx.read_graphml("D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\test03.graphml")

    try:
        # 加载训练好的权重（PyTorch格式）
        model_path = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\GRAPH_VISUALIZATION\\best_model.pt"
        checkpoint = torch.load(model_path)
        trained_model.H = checkpoint['H']
        trained_model.U = checkpoint['U']
        print("成功加载模型权重")
    except Exception as e:
        print(f"加载模型权重失败: {e}")
        exit(1)

    # 创建初始输入向量
    input_vector = torch.zeros(input_size)
    start_node = 61  # 起始节点ID
    input_vector[start_node] = 1.0  # 设置起始节点
    input_vector[-1] = 1.0  # 设置常数项

    # 定义出口实体列表
    exit_entities = ["63", "102", "155", "135", "123"]
    print(f"出口节点: {exit_entities}")

    # 初始化路径预测器并执行预测
    path_predictor = PathPredictor(trained_model, threshold=0.01, graph=graph)
    predicted_path = path_predictor.predict_path(input_vector, exit_entities)
