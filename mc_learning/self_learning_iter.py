import os
import random
from tkinter import messagebox

import networkx as nx
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from mc_learning.generate_paths import DataSetGenerate, manhattan_distance, euclidean_distance
from mc_learning.train4 import PathDataset, Net
from mc_learning.path_predict import PathPredictor
import argparse


class SelfLearningTrainer:
    def __init__(self, csv_file, weights_path, graph_path, device, batch_size=128, hidden_size=1500):
        """
        初始化自学习训练器。

        Args:
            csv_file (str): 数据集路径
            weights_path (str): 隐藏层权重路径
            batch_size (int): 训练的批量大小
            hidden_size (int): 隐藏层大小
        """
        self.emergency_areas = None
        self.evaluator = DataSetGenerate(output_file_14="output_file_140407.csv", path_output_file="output_path.csv",
                                         output_file="output_file_0407.csv", grapml_file=graph_path)
        self.evaluator.set_emergency_area()
        self.graph = nx.read_graphml(graph_path)
        self.csv_file = csv_file
        self.weights_path = weights_path
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        # 初始化数据集
        if not os.path.exists(self.csv_file):
            print("数据集不存在...")
            return

        self.dataset = PathDataset([self.csv_file])
        self.input_size = self.dataset.input_size
        self.output_size = self.dataset.output_size

        # 初始化BP神经网络
        self.model = Net(self.input_size, self.hidden_size, self.output_size, learning_rate=0.01,
                         momentum=0.5, weights_path=self.weights_path)

        self.path_predictor = PathPredictor(self.model, threshold=0.6, graph=self.graph, device=self.device)

    def self_learning_train(self, num_iterations=5, epochs=200):
        """
        执行自学习迭代训练。

        Args:
            num_iterations (int): 训练迭代次数
            epochs (int): 每次迭代的训练轮数
        """
        for iteration in range(num_iterations):
            print(f"\n自学习迭代 {iteration + 1}/{num_iterations} 开始...")
            # 由模型生成新数据->评分 -->更新数据集-->train
            paths_scores_map = []
            for paths in range(1, 100):
                predict_path, is_reach_exit = self.predict()
                if is_reach_exit == 0:
                    print("路径预测失败，使用数据集生成")
                    print(predict_path)
                    for i in range(5):
                        result = self.evaluator.generate_path(self.emergency_areas,
                                                              specified_start=str(predict_path[0]))
                        if result:
                            path, path_length, eu_path_length, iterations = result
                            score = self.evaluator.evaluate_path(path, path_length, eu_path_length,
                                                                 self.emergency_areas)
                            if score and path:
                                print(f"path:{path} score:{score}  danger:{self.emergency_areas}")
                                paths_scores_map.append([path, score, self.emergency_areas])
                        else:
                            print(f"生成路径{predict_path}失败")
                else:
                    current_pos_id = predict_path[0]
                    current_pos = self.evaluator.get_node_center(current_pos_id)
                    path_length = 0
                    eu_path_length = 0
                    for step in predict_path[1:]:
                        next_pos = self.evaluator.get_node_center(step)
                        if current_pos and next_pos:
                            step_length = manhattan_distance(current_pos, next_pos)
                            eu_step_length = euclidean_distance(current_pos, next_pos)
                            path_length += step_length
                            eu_path_length += eu_step_length
                            current_pos = next_pos
                    score = self.evaluator.evaluate_path(predict_path, path_length, eu_path_length,
                                                         self.emergency_areas)
                    if score and predict_path:
                        paths_scores_map.append([predict_path, score, self.emergency_areas])

            paths = [item[0] for item in paths_scores_map]
            scores = [item[1] for item in paths_scores_map]
            danger_zones = paths_scores_map[-1][2] if paths_scores_map else []
            self.evaluator.dataset_create(paths, scores, self.evaluator.output_file,
                                          self.evaluator.path_output_file,
                                          danger_zones,
                                          threshold=0.3, sorted=1)  # self.evaluator.output_file_14,

            # 重新划分训练集和数据集
            # self.dataset_generate.generate_paths(count=100)
            self.dataset = PathDataset([self.csv_file,"47_test_14.csv","61_test_14.csv","552_test_14.csv"])
            train_dataset, val_dataset = train_test_split(self.dataset, test_size=0.3, random_state=42)  # 7:3

            # 创建 DataLoader
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            # 训练神经网络
            self.model.train(train_loader, val_loader, epochs=epochs,
                             save_model_path=f"self_iters_{iteration + 1}.pt", patience=10)

    def predict(self):
        """使用训练好的模型进行路径预测"""
        self.emergency_areas = self.evaluator.select_emergency_areas()
        # 定义出口和要避开的数字
        exit_entities = {"63", "102", "155", "135", "123"}
        # 缺失的数字
        avoid_numbers = {"27", "104", "105", "121", "122", "129", "151", "152", "469", "478", "542"}

        # 将出口和要避开的数字合并为一个集合，提高查询效率
        all_excluded_numbers = exit_entities.union(avoid_numbers)

        # 生成随机起始节点，避开出口和指定的数字
        while True:
            start_node = random.randint(1, 627)  # 生成 1 到 627 之间的随机数
            if str(start_node) not in all_excluded_numbers:
                break

        # 设置起始节点
        n = len(self.graph.nodes)
        input_vector = torch.zeros(2 * n + 1, device=self.device, dtype=torch.float32)
        node_list = list(self.graph.nodes)

        index = node_list.index(str(start_node))
        input_vector[index] = 1.0  # 将第一个 n 位中对应节点设置为 1

        input_vector[-1] = 1.0

        # 预测路径
        path_predictor = PathPredictor(self.model, threshold=0.01, graph=self.graph)
        predicted_path, success = path_predictor.predict_path(input_vector, exit_entities)

        print("\n预测路径:")
        print(" -> ".join(map(str, predicted_path)))
        return predicted_path, success


def get_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练神经网络并进行路径预测")

    parser.add_argument("--iterations", type=int, default=5, help="自学习迭代次数")
    parser.add_argument("--epochs", type=int, default=100, help="一次自学习训练次数")
    parser.add_argument("--csv_file", type=str,
                        default="D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\output_file_0407.csv",
                        help="数据集文件路径")
    parser.add_argument("--weights_path", type=str,
                        default="D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\full_vector_best_model_0415"
                                ".pt",
                        help="权重文件路径")
    parser.add_argument("--device", type=str, default=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        help="cpu还是gpu")
    parser.add_argument("--graph_path", type=str,
                        default="D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\test03.graphml")
    # parser.add_argument("--save_model_path",type=str,default="")

    return parser.parse_args()


def main(args):
    # 设置迭代次数
    self_learning_trainer = SelfLearningTrainer(args.csv_file, args.weights_path, args.graph_path, args.device)

    # 开始训练（自学习 5 轮，每轮 100 代）
    self_learning_trainer.self_learning_train(num_iterations=args.iterations, epochs=args.epochs)


if __name__ == "__main__":
    main(get_arguments())
    # todo 两个问题：1. 路径探索对危险区域不敏感-->增加紧急实体的数量，再次训练，2.高分路径不一定更好（实体加分较多）-->改变自迭代的评分细则

