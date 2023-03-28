import torch
import torch.nn as nn

# 于构建一个多层感知器（MLP）的神经网络模型。
# sizes包含每一层神经元数目。例如，sizes=[10, 20, 5] 表示该 MLP 包含 3 层，第一层有 10 个神经元，第二层有 20 个神经元，第三层有 5 个神经元。
# activation是激活函数，它将在模型的所有隐藏层上使用。它是一个 PyTorch 中的函数对象，例如 nn.ReLU。
# output_activation是一个激活函数，它将在模型的输出层上使用。默认值为 nn.Identity，即没有激活函数。

#对于每一层（除了输出层），它使用 nn.Linear 函数添加一个线性层（即全连接层），并将其添加到 layers 列表中。线性层的输入和输出大小分别为 sizes[j] 和 sizes[j+1]，即从列表 sizes 中获取。
#对于隐藏层，它还使用 activation 函数添加一个激活函数，并将其添加到 layers 列表中。对于输出层，它使用 output_activation 函数添加一个激活函数，并将其添加到 layers 列表中。
def mlp(sizes, activation, output_activation=nn.Identity): #第三个参数是默认值
    layers = []
    for j in range(len(sizes) - 1): #除了输出层之外
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers) #Sequential是容器类
#全连接层（Fully Connected Layer），也称为线性层（Linear Layer）或稠密层（Dense Layer），是深度学习中最基本的一种层类型。在全连接层中，每个输入神经元都与每个输出神经元相连，因此全连接层的输出可以看作是输入的线性变换。
class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)
    #obs_dim 表示输入数据的维度，即状态的特征数。                             ->输入
    #hidden_sizes 是一个列表，包含了 MLP 模型中隐藏层的神经元数目。           ->隐藏层
    #activation 是一个激活函数，用于在 MLP 的隐藏层中对输出进行非线性变换。     ->激活函数
    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


obs_dim=3
observation=torch.as_tensor([0.5, 0.1, 0],dtype=torch.float32)
hidden_sizes=[64,64]
activation=nn.Tanh    #激活函数，它将输入张量的每个元素通过双曲正切函数进行转换。nn.Tanh 的作用是将输入张量中的每个元素映射到范围在 -1 到 1 之间的值。类似于Sigmoid

critic=MLPCritic(obs_dim,hidden_sizes,activation)
print('v_net={}'.format(critic.v_net))
print('v_net(obs)={}'.format(critic.v_net(observation)))
print('v_net forward={}'.format(critic.forward(observation)))
