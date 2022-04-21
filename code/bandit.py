# coding=utf-8
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pylab import mpl
np.random.seed(sum(map(ord, "aesthetics")))
mpl.rcParams['font.sans-serif'] = ['SimHei']


class Bandit:
    def __init__(self, k_arm=10, initial=0., step_size=0.1, sample_averages=0, true_reward=0.):
        self.k = k_arm  # 设定Bandit臂数
        self.step_size = step_size  # 设定更新步长
        self.sample_averages = sample_averages  # int变量，1表示使用简单增量式算法, 0表示不使用简单增量式算法，2表示使用gradient
        self.indices = np.arange(self.k)  # 创建动作索引（和Bandit臂数长度相同）
        self.initial = initial  # 设定初始价值估计，如果调高就是乐观初始值
        self.true_reward = true_reward  # 存储一个Bandit的真实收益均值，为下面制作一个Bandit的真实价值函数做准备
        self.average_reward = 0  # 用于存储baseline所需的平均收益

    # 初始化训练状态，设定真实收益和最佳行动状态
    def reset(self):
        # #设定真实价值函数，在一个标准高斯分布上抬高一个外部设定的true_reward，true_reward设置为非0数字以和不使用baseline的方法相区分
        self.q_true = np.random.randn(self.k) + self.true_reward

        # 设定初始估计值，在全0的基础上用initial垫高，表现乐观初始值
        self.q_estimation = np.zeros(self.k) + self.initial

        # 计算每个动作被选取的数量，为UCB做准备
        self.action_count = np.zeros(self.k)
        # 根据真实价值函数选择最佳策略，为之后做准备
        self.best_action = np.argmax(self.q_true)
        # 设定时间步t为0
        self.time = 0

    # take an action, update estimation for this action
    # 选择动作并进行价值更新
    def step(self, action):
        # generate the reward under N(real reward, 1)
        # 按照之前产生的真实价值作为均值，产生一个遵从正态分布的随机奖励
        reward = np.random.randn() + self.q_true[action]
        # 执行动作后时间步加一
        self.time += 1
        # 动作选择数量计数
        self.action_count[action] += 1
        # 计算平均回报（return），为baseline做好准备
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages == 1:
            # update estimation using sample averages
            # 增量式实现（非固定步长1/n）
            # Qn+1 = Qn + [Rn-Qn]/n
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]

        elif self.sample_averages == 0:
            # update estimation with constant step size
            # 如过上面两种方法都没有选取，就选用常数步长更新
            # Qn+1 = Qn + å(Rn-Qn)
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])

        elif self.sample_averages == 2:
            # 将k臂做onehot编码
            one_hot = np.zeros(self.k)
            # 将选择的动作位置置为1.
            one_hot[action] = 1
            # 在梯度Bandit上使用baseline
            if self.gradient_baseline:
                # 平均收益作为baseline
                baseline = self.average_reward
            else:
                # 不使用baseline的情况
                baseline = 0
            # 梯度式偏好函数更新：
            # 对action At：Ht+1(At) = Ht(At) + å(Rt-R_avg)(1-π(At))
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        return reward


class ETC(Bandit):
    def __init__(self, k_arm=10, M=0., initial=0., step_size=0.1, sample_averages=0):
        Bandit.__init__(self, k_arm, initial, step_size, sample_averages)
        self.m = M  # 每个arm都执行m次

    def act(self):
        if self.time < self.m * self.k:
            return self.time % self.k
        else:
            q_best = np.max(self.q_estimation)
            return np.random.choice(np.where(self.q_estimation == q_best)[0])


class Epsilon_Greedy(Bandit):
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=0):
        Bandit.__init__(self, k_arm, initial, step_size, sample_averages)
        self.epsilon = epsilon  # 设定epsilon-贪婪算法的参数

    def act(self):
        # epsilon概率随机选取一个动作下标
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)
        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])


class UCB1(Bandit):
    def __init__(self, k_arm=10, initial=0., step_size=0.1, sample_averages=0, UCB_param=None):
        Bandit.__init__(self, k_arm, initial, step_size, sample_averages)
        # UCB参数
        self.UCB_param = UCB_param  # 如果设定了UCB的参数c，就会在下面使用UCB算法
        self.time = 0  # 计算所有选取动作的数量，为UCB做准备
        
    def act(self):
        # 按照公式计算UCB估计值
        UCB_estimation = self.q_estimation + \
                         self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
        # 选择不同动作导致的预测值中最大的
        q_best = np.max(UCB_estimation)
        # 返回基于UCB的动作选择下值最大的动作
        return np.random.choice(np.where(UCB_estimation == q_best)[0])


class Gradient(Bandit):
    def __init__(self, k_arm=10, initial=0., step_size=0.1, sample_averages=2,
                 gradient_baseline=False, true_reward=0.):
        Bandit.__init__(self, k_arm, initial, step_size, sample_averages, true_reward)

        # gradient 公式参数
        self.gradient_baseline = gradient_baseline # 设定梯度Bandit算法中的基准项（baseline），通常都用时刻t内的平均收益表示
        self.average_reward = 0  # 用于存储baseline所需的平均收益

    def act(self):
        # 基于梯度Bandit算法
        exp_est = np.exp(self.q_estimation)
        self.action_prob = exp_est / np.sum(exp_est)
        return np.random.choice(self.indices, p=self.action_prob)


class UCB2(Bandit):
    def __init__(self, k_arm=10, initial=0., step_size=0.1, sample_averages=0, alpha=0.1):
        Bandit.__init__(self, k_arm, initial, step_size, sample_averages)
        # UCB2参数
        self.alpha = alpha  # UCB2公式中的alpha
        self.r = [0 for col in range(self.k)]
        self.__current_arm = 0
        self.__next_update = 0

    def __bonus(self, n, rs):
        tau = self.__tau(rs)
        bonus = [math.sqrt((1. + self.alpha) * math.log(math.e * float(n) / t) / (2 * t)) for t in tau]
        return bonus

    def __tau(self, rs):
        return [int(math.ceil((1 + self.alpha) ** r)) for r in rs]

    def __tau_i(self, r):
        return int(math.ceil((1 + self.alpha) ** r))

    def __set_arm(self, arm):
        self.__current_arm = arm
        self.__next_update += max(1, self.__tau_i(self.r[arm] + 1) - self.__tau_i(self.r[arm]))
        # 确保玩tau(ri+1)-tau(ri)次
        self.r[arm] += 1

    def act(self):
        # play each arm once
        for arm in range(self.k):
            if self.action_count[arm] == 0:
                self.__set_arm(arm)
                return arm

        # 执行次数限制
        if self.__next_update > sum(self.action_count):
            return self.__current_arm
        total_counts = sum(self.action_count)
        bonus = self.__bonus(total_counts, self.r)
        UCB2_estimation = self.q_estimation + bonus

        # 选择不同动作导致的预测值中最大的
        q_best = np.max(UCB2_estimation)
        self.__set_arm(np.random.choice(np.where(UCB2_estimation == q_best)[0]))
        return np.random.choice(np.where(UCB2_estimation == q_best)[0])
