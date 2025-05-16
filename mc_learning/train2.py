import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm  # 用于显示训练和验证的进度条


# 自定义数据集类，用于处理输入数据文件（CSV格式）
class PathDataset(Dataset):
    """
    自定义路径数据集，用于加载输入特征和输出标签。
    """

    def __init__(self, data_file):
        """
        初始化数据集。
        Args:
            data_file (str): 数据集文件路径（CSV格式）。
        """
        # 读取CSV数据
        self.data = pd.read_csv(data_file)
        # 将输入和输出列的字符串数据转换为数组形式
        self.inputs = np.array([eval(x) for x in self.data["input"].values])  # 输入特征
        self.outputs = np.array([eval(y) for y in self.data["output"].values])  # 输出标签

    def __len__(self):
        """
        返回数据集的样本数量。
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本数据。
        Args:
            idx (int): 样本索引。
        Returns:
            input_vector (tensor): 输入特征向量。
            output_vector (tensor): 输出标签向量。
        """
        input_vector = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output_vector = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return input_vector, output_vector


# 定义一个函数，用于从CSV文件加载数据集并返回数据加载器
def load_dataset(csv_file, batch_size=32):
    """
    从CSV文件中加载数据集并返回DataLoader。
    Args:
        csv_file (str): 数据集的CSV文件路径。
        batch_size (int): 每个批次的数据大小。
    Returns:
        DataLoader: 数据加载器，用于批量获取数据。
    """
    dataset = PathDataset(csv_file)  # 创建数据集对象
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 创建数据加载器
    return dataloader


# 定义BP神经网络模型
class BPNeuralNetwork:
    """
    定义BP神经网络，用于路径规划任务。
    """

    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, momentum=0.5):
        """
        初始化BP神经网络。
        Args:
            input_size (int): 输入层神经元的数量。
            hidden_size (int): 隐藏层神经元的数量。
            output_size (int): 输出层神经元的数量。
            learning_rate (float): 学习率。
            momentum (float): 动量因子。
        """
        # 模型参数初始化
        self.y = None
        self.d = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # 初始化权重矩阵，权重值随机分布
        self.H = torch.randn(hidden_size, input_size) / np.sqrt(input_size)  # 输入到隐藏层权重
        self.U = torch.randn(output_size, hidden_size) / np.sqrt(hidden_size)  # 隐藏到输出层权重

        # 初始化用于动量更新的权重矩阵
        self.delta_H = torch.zeros_like(self.H)  # 输入到隐藏层动量
        self.delta_U = torch.zeros_like(self.U)  # 隐藏到输出层动量

    def sigmoid(self, x):
        """
        Sigmoid激活函数。
        Args:
            x (tensor): 输入张量。
        Returns:
            tensor: 激活后的张量。
        """
        return torch.sigmoid(x)

    def sigmoid_derivative(self, x):
        """
        Sigmoid函数的导数，用于计算梯度。
        Args:
            x (tensor): 输入张量。
        Returns:
            tensor: Sigmoid导数。
        """
        return x * (1 - x)

    def forward_propagation(self, x):
        """
        前向传播，计算隐藏层和输出层的激活值。
        Args:
            x (tensor): 输入向量或批次数据。
        Returns:
            d (tensor): 隐藏层激活值。
            y (tensor): 输出层激活值。
        """
        # 转换输入为Tensor格式
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if len(x.shape) == 1:  # 如果是单个样本，添加批次维度
            x = x.unsqueeze(0)

        # 计算隐藏层输出
        self.d = self.sigmoid(torch.matmul(x, self.H.T))  # [batch_size, hidden_size]
        # 计算输出层输出
        self.y = self.sigmoid(torch.matmul(self.d, self.U.T))  # [batch_size, output_size]
        return self.d, self.y

    def backward_propagation(self, x, y_true):
        """
        反向传播，计算梯度并更新权重。
        Args:
            x (tensor): 输入向量或批次数据。
            y_true (tensor): 实际的输出目标。
        """
        # 计算输出层误差
        delta_U = (self.y - y_true) * self.sigmoid_derivative(self.y)
        # 计算隐藏层误差
        delta_H = torch.matmul(delta_U, self.U) * self.sigmoid_derivative(self.d)

        # 计算梯度
        grad_U = torch.matmul(delta_U.T, self.d) / x.shape[0]
        grad_H = torch.matmul(delta_H.T, x) / x.shape[0]

        # 更新权重（包含动量因子）
        self.delta_U = self.momentum * self.delta_U - self.learning_rate * grad_U
        self.delta_H = self.momentum * self.delta_H - self.learning_rate * grad_H
        self.U += self.delta_U
        self.H += self.delta_H

    def train(self, train_loader, val_loader, epochs):
        """
        训练神经网络，并保存最佳模型权重。
        Args:
            train_loader (DataLoader): 训练数据加载器。
            val_loader (DataLoader): 验证数据加载器。
            epochs (int): 训练的总轮数。
        """
        best_val_loss = float('inf')  # 保存最优验证损失

        for epoch in range(epochs):
            # 训练阶段
            self.train_epoch(train_loader, epoch, epochs)

            # 验证阶段
            val_loss = self.validate(val_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.6f}")

            # 如果当前验证损失更低，则保存模型权重
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({'H': self.H, 'U': self.U}, "best_model.pt")
                print(f"New best model saved with val loss: {best_val_loss:.6f}")

    def train_epoch(self, train_loader, epoch, total_epochs):
        """
        单个训练轮次。
        Args:
            train_loader (DataLoader): 训练数据加载器。
            epoch (int): 当前轮次编号。
            total_epochs (int): 总轮次数。
        """
        total_loss = 0

        # 使用进度条显示训练进度
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}") as pbar:
            for x, y_true in pbar:
                self.forward_propagation(x)
                loss = torch.mean((self.y - y_true) ** 2)
                total_loss += loss.item()
                self.backward_propagation(x, y_true)
                pbar.set_postfix({'loss': f'{total_loss:.6f}'})

    def validate(self, val_loader):
        """
        验证模型性能。
        Args:
            val_loader (DataLoader): 验证数据加载器。
        Returns:
            float: 验证损失。
        """
        total_val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                _, y_pred = self.forward_propagation(x_val)
                val_loss = torch.mean((y_pred - y_val) ** 2)
                total_val_loss += val_loss.item()
        return total_val_loss / len(val_loader)
