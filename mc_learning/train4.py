import json
import os

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import json
import pandas as pd
import torch
from torch.utils.data import Dataset

import json
import pandas as pd
import torch
from torch.utils.data import Dataset

class PathDataset(Dataset):
    def __init__(self, data_files):
        self.inputs = []
        self.outputs = []

        for data_file in data_files:
            try:
                data = pd.read_csv(data_file)
            except Exception as e:
                print(f"[读取失败] {data_file} 读取失败：{e}")
                continue

            for idx, row in data.iterrows():
                try:
                    input_str = row["input"]
                    output_str = row["output"]

                    input_vec = json.loads(input_str.replace("'", "\""))
                    output_vec = json.loads(output_str.replace("'", "\""))

                    if not isinstance(input_vec, list) or not isinstance(output_vec, list):
                        raise ValueError("解析结果不是 list！")

                    self.inputs.append(input_vec)
                    self.outputs.append(output_vec)

                except Exception as e:
                    print(f"\n解析失败 | 文件: {data_file} | 行号: {idx}")
                    print(f"input: {row['input']}")
                    print(f"output: {row['output']}")
                    print(f"错误信息: {e}")
                    continue

        if not self.inputs:
            raise ValueError("没有合法的输入样本！")

        self.input_size = len(self.inputs[0])
        self.output_size = len(self.outputs[0])
        for vec in self.inputs:
            if len(vec) != self.input_size:
                raise ValueError("输入维度不一致！")
        for vec in self.outputs:
            if len(vec) != self.output_size:
                raise ValueError("输出维度不一致！")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_vector = torch.tensor(self.inputs[idx], dtype=torch.float32).to(device)
        output_vector = torch.tensor(self.outputs[idx], dtype=torch.float32).to(device)
        return input_vector, output_vector



class Net:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.5, weights_path=None):
        self.weights_path = weights_path
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # 权重矩阵
        bound = 1 / (4 * output_size + 2) ** 0.5  # N = output_size
        self.H = torch.nn.Parameter(torch.empty(hidden_size, input_size).uniform_(-bound, bound).to(device))
        self.U = torch.nn.Parameter(torch.empty(output_size, hidden_size).uniform_(-bound, bound).to(device))

        # 初始化动量项
        self.pU = torch.zeros_like(self.U, requires_grad=False).to(device)
        self.pH = torch.zeros_like(self.H, requires_grad=False).to(device)
        self.load_weights()

    def forward_propagation(self, x):
        """
        forward propagation
        :param x: tensor type
        :return: d(use relu to simulate prevent from 梯度爆炸),y
        """
        if not isinstance(x, torch.Tensor):  # type check
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        # # 计算隐藏层输出
        # self.d = self.sigmoid(torch.matmul(x, self.H.T))
        # # 计算输出层输出
        # self.y = self.sigmoid(torch.matmul(self.d, self.U.T))
        d = torch.nn.functional.leaky_relu(torch.matmul(x, self.H.T), negative_slope=0.1)
        y = torch.matmul(d, self.U.T)
        return d, y

    def backward_propagation(self, x, y_true, d, y):
        """
        backward propagation:manual backward propagation with momentum
        :param x: tensor
        :param y_true: tensor
        :param d:
        :param y:
        :return:
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32, device=device)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(y_true.shape) == 1:
            y_true = y_true.unsqueeze(0)

        # ====== 检查 NaN 和 Inf ======
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("x (input):", x)
            print("y_true (label):", y_true)
            print("y_pred (output):", y)
            torch.save({'x': x, 'y_true': y_true}, f"bad_sample.pt")
            raise ValueError("停止训练：发现 NaN 或 Inf")

        # === 计算误差项 === #
        delta_U = y - y_true  # 输出层误差
        delta_N = torch.matmul(delta_U, self.U)  # 输入层误差项 δN

        # === 梯度计算 === #
        grad_y = delta_U  # 因为输出层没有激活函数（线性）
        grad_d = delta_N * ((d > 0).float() + 0.1 * (d <= 0).float())  # leaky ReLU 导数

        # === 原始梯度 === #
        grad_U = torch.matmul(grad_y.T, d)
        grad_H = torch.matmul(grad_d.T, x)

        # === 动量更新 === #
        with torch.no_grad():
            self.pU = self.momentum * self.pU + self.learning_rate * grad_U
            self.pH = self.momentum * self.pH + self.learning_rate * grad_H
            self.U -= self.pU
            self.H -= self.pH

    def compute_relative_l2_error(self, y_pred, y_true):
        """
        relative_l2_error: compute # todo
        :param y_pred: tensor
        :param y_true: tensor
        :return: l2_error
        """
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32, device=device)

        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32, device=device)

        l2_norm_error = torch.norm(y_pred - y_true, p=2, dim=1)
        l2_norm_target = torch.norm(y_true, p=2, dim=1)
        relative_error = l2_norm_error / (l2_norm_target + 1e-8)
        return torch.mean(relative_error).item()

    def compute_max_error(self, y_pred, y_true):
        """
        maximum_error:
        :param y_pred:
        :param y_true:
        :return: max_error
        """
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32, device=device)
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32, device=device)

        max_errors, _ = torch.max(torch.abs(y_pred - y_true), dim=1)
        return torch.mean(max_errors).item()

    def train(self, train_loader, val_loader, epochs, save_model_path, patience=None):
        """
        train
        :param train_loader: set of train
        :param val_loader: set of validation
        :param epochs: epoch to train
        :param save_model_path: save_path
        :param patience: fill when you hope to early stop
        """
        best_val_loss = float('Inf')
        best_val_max = float('Inf')
        no_improve = 0

        # 初始化TensorBoard SummaryWriter
        writer = SummaryWriter(log_dir='runs/bp_model')  # 创建SummaryWriter实例
        optimizer = torch.optim.Adam([self.H, self.U], lr=0.001)
        # optimizer = torch.optim.SGD([self.H, self.U], lr=self.learning_rate, momentum=self.momentum)
        scheduler = StepLR(optimizer, step_size=1000, gamma=0.7)  # 每10个epoch，学习率乘以0.7

        for epoch in range(epochs):
            scheduler.step()  # 每个epoch调用
            self.train_epoch(train_loader, epoch, epochs, writer)
            avg_val_rel, avg_val_max = self.validate(val_loader)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Rel Err: {avg_val_rel:.4f}, Max Err: {avg_val_max:.4f}")

            # 记录到TensorBoard
            writer.add_scalar('Error/Rel', avg_val_rel, epoch)
            writer.add_scalar('Error/Max', avg_val_max, epoch)

            if avg_val_rel < best_val_loss and avg_val_max < best_val_max:
                best_val_loss = avg_val_rel
                best_val_max = avg_val_max
                no_improve = 0
                torch.save({'H': self.H, 'U': self.U}, save_model_path)
            elif patience is not None:  # 自行决定是否早停
                no_improve += 1
                if no_improve >= patience:
                    print(f"早停：验证损失连续 {patience} 轮未改善")
                    break

        torch.save({'H': self.H, 'U': self.U}, "final_model.pt")
        writer.close()

    def train_epoch(self, train_loader, epoch, total_epochs, writer):
        """
        one train epoch
        :param train_loader:
        :param epoch:
        :param total_epochs:
        :param writer:
        :return: None
        """
        total_relative_loss = 0.0
        total_max_error = 0.0
        total_samples = 0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}") as pbar:
            for i, (x, y_true) in enumerate(pbar):
                # 前向传播
                d, y_pred = self.forward_propagation(x)
                # 反向传播
                self.backward_propagation(x, y_true, d, y_pred)

                batch_size = x.size(0)
                total_samples += batch_size

                # 每个样本的相对误差
                l2_error = torch.norm(y_pred - y_true, p=2, dim=1)
                l2_target = torch.norm(y_true, p=2, dim=1)
                relative_error = l2_error / (l2_target + 1e-8)
                total_relative_loss += relative_error.sum().item()

                # 每个样本的最大误差
                max_errors = torch.max(torch.abs(y_pred - y_true), dim=1)[0]
                total_max_error += max_errors.sum().item()

                # 平均误差
                avg_relative_loss = total_relative_loss / total_samples
                avg_max_error = total_max_error / total_samples

                # 更新进度条
                pbar.set_postfix({
                    "Rel Err": f"{avg_relative_loss:.4f}",
                    "Max Err": f"{avg_max_error:.4f}"
                })

        # 记录训练损失到 TensorBoard
        writer.add_scalar('Loss/Train_Relative', avg_relative_loss, epoch)
        writer.add_scalar('Loss/Train_Max', avg_max_error, epoch)

    def validate(self, val_loader):
        """
        :param val_loader:
        :return:
        """
        total_val_relative_error = 0.0
        total_val_max_error = 0.0
        total_samples = 0

        with torch.no_grad():
            for x_val, y_val in tqdm(val_loader, desc="Validating", leave=False):
                x_val = x_val.to(device).float()
                y_val = y_val.to(device).float()

                # 前向传播
                _, y_pred = self.forward_propagation(x_val)

                # 当前 batch 的样本数量
                batch_size = x_val.size(0)
                total_samples += batch_size

                # 计算每个样本的误差（返回的是 shape: [batch_size]）
                l2_norm_error = torch.norm(y_pred - y_val, p=2, dim=1)
                l2_norm_target = torch.norm(y_val, p=2, dim=1)
                relative_error = l2_norm_error / (l2_norm_target + 1e-8)
                total_val_relative_error += relative_error.sum().item()  # 累加所有样本的误差

                max_errors = torch.max(torch.abs(y_pred - y_val), dim=1)[0]
                total_val_max_error += max_errors.sum().item()  # 累加所有样本的最大误差

        # 统一平均
        avg_val_relative_error = total_val_relative_error / total_samples
        avg_val_max_error = total_val_max_error / total_samples

        return avg_val_relative_error, avg_val_max_error

    def load_weights(self):
        """如果存在已训练权重，则加载"""
        if self.weights_path and os.path.exists(self.weights_path):
            print("检测到已保存的权重文件，正在加载权重...")
            checkpoint = torch.load(self.weights_path)
            self.H = checkpoint['H']
            self.U = checkpoint['U']
            print("✅ 权重加载成功，继续训练")


if __name__ == '__main__':
    # dataset_path = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\test_14.csv"
    dataset_47 = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\47_test_14.csv"
    dataset_64 = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\61_test_14.csv"
    dataset_552 = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\552_test_14.csv"
    dataset_path2 = "D:\\A大三课内资料\\系统能力培养\\junior\\pythonProject\\mc_learning\\output_file_0407.csv"
    dataset_test = "test_emergency.csv"
    batch_size = 128  # 调整批次大小
    # dataset = PathDataset([dataset_path2, dataset_47, dataset_64, dataset_552])  # dataset_path,dataset_path2,
    dataset = PathDataset([dataset_test])
    train_dataset, val_dataset = train_test_split(dataset, test_size=0.3, random_state=42)  # 7:3

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    input_size = dataset.input_size
    output_size = dataset.output_size
    hidden_size = 1500
    learning_rate = 1e-4
    momentum = 0.5
    nn = Net(input_size, hidden_size, output_size, learning_rate,
             momentum, weights_path="full_vector_best_model_0415.pt")  #
    nn.train(train_loader, val_loader, epochs=200,
             save_model_path="full_vector_best_model_0416.pt")
