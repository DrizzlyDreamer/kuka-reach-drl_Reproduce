import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

# obs_dim 表示输入状态的维度，act_dim 表示输出动作的维度。
class MLPActor(nn.Module):
    def __init__(self,obs,obs_dim,hidden_sizes,act_dim,activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)  # 高斯分布的标准差 ，用.ones创建了数组，长度为act_dim，每个元素都是-0.5，每个动作维度的标准差都是-0.5
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std)) #as_tensor转换为张量，Par.封装为可学习PyTorch参数，可以被更新  *****************
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)  # self.mu_net 是一个 MLP 模型，用于估计在给定状态下，动作分布的均值
        self.obs=obs

    # 首先是定义std，也就是高斯分布的标准差，关于为什么要用log而不是直接定义标准差呢，
    # 这是因为std也是作为网络的一个参数来优化，在网络的反向传播中，更倾向于log函数
    # 关于具体原因，可以参考一下这篇文章，基本原理是相同的
    # https://blog.csdn.net/bornfree5511/article/details/115017192?spm=1001.2014.3001.5501

#这三个函数结合起来，可以用于在给定状态下生成一个动作，并计算该动作在动作分布中的对数概率密度。
    def _distribution(self):
        mu = self.mu_net(self.obs)
        std = torch.exp(self.log_std)
        self.pi=Normal(mu,std)   #利用神经网络输出的  动作分布均值'mu'和  标准差'std'，然后用Normal转换为一个高斯分布self.pi
        return self.pi
    
    def _get_action(self):
        self.act=self.pi.sample()    #从self.pi中抽样一个动作，并返回
        return self.act

    # 该函数计算当前动作 self.act 在动作分布 self.pi 中的对数概率密度。 然后对这些概率密度进行求和得到 总的对数概率密度 logp_a 。最后返回 logp_a。
    def _log_prob_of_act_from_distribution(self):
        logp_a=pi.log_prob(self.act).sum(axis=-1)     #求概率密度   ->   求和
        return logp_a

obs_dim=3
act_dim=3
observation=torch.as_tensor([0.5, 0.1, 0],dtype=torch.float32)
hidden_sizes=[64,64]
activation=nn.Tanh

actor=MLPActor(observation,obs_dim,hidden_sizes,act_dim,activation)
pi=actor._distribution()
act=actor._get_action()
logp_a=actor._log_prob_of_act_from_distribution()

print('actor={},\npi={},\nact={},\nlogp_a={}'.format(actor,pi,act,logp_a))

