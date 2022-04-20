# coding=utf-8
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from bandit import ETC, Epsilon_Greedy, UCB1, Gradient
from simulation import simulate


# 作奖励分布的图
def figure_reward_distribution():
    plt.violinplot(dataset=np.random.randn(200, 10) + np.random.randn(10))
    plt.xlabel("action")
    plt.ylabel("reward distribution")
    plt.show()


# ETC调参实验
def figure_ETC(runs=2000, time=1000):
    ms = [5, 20, 50]
    bandits = [ETC(k_arm=10, M=m, initial=0., step_size=0.1, sample_averages=0) for m in ms]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for m, rewards in zip(ms, rewards):
        plt.plot(rewards, label='m = %.02f' % (m))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for m, counts in zip(ms, best_action_counts):
        plt.plot(counts, label='m = %.02f' % (m))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.show()


# epsilon调参实验
def figure_epsilon_greedy(runs=2000, time=1000):
    epsilons = [0, 0.1, 0.01]
    bandits = [Epsilon_Greedy(epsilon=eps, sample_averages=1) for eps in epsilons]
    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, rewards in zip(epsilons, rewards):
        plt.plot(rewards, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.show()


# 乐观初始值测试
# 乐观值设定会让每个动作在收敛之前都被尝试很多次
def figure_positive_init(runs=2000, time=1000):
    bandits = [Epsilon_Greedy(epsilon=0, initial=5, step_size=0.1), Epsilon_Greedy(epsilon=0.1, initial=0, step_size=0.1)]
    # 实验对照组
    best_action_counts, _ = simulate(runs, time, bandits)

    plt.plot(best_action_counts[0], label='epsilon = 0, q = 5')
    plt.plot(best_action_counts[1], label='epsilon = 0.1, q = 0')
    plt.xlabel('Steps')
    plt.ylabel('% optimal action')
    plt.legend()
    plt.show()


# UCB测试
def figure_UCB1(runs=2000, time=1000):
    bandits = [UCB1(UCB_param=2, sample_averages=1), \
               Epsilon_Greedy(epsilon=0.1, sample_averages=1)]
    _, average_rewards = simulate(runs, time, bandits)

    plt.plot(average_rewards[0], label='UCB c = 2')
    plt.plot(average_rewards[1], label='epsilon greedy epsilon = 0.1')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.show()


# 梯度Bandit
def figure_gradient_bandit(runs=2000, time=1000):
    bandits = []
    bandits.append(Gradient(sample_averages=2, step_size=0.1, gradient_baseline=True, true_reward=4))
    bandits.append(Gradient(sample_averages=2, step_size=0.1, gradient_baseline=False, true_reward=4))
    bandits.append(Gradient(sample_averages=2, step_size=0.4, gradient_baseline=True, true_reward=4))
    bandits.append(Gradient(sample_averages=2, step_size=0.4, gradient_baseline=False, true_reward=4))
    best_action_counts, _ = simulate(runs, time, bandits)
    labels = ['alpha = 0.1, with baseline',
              'alpha = 0.1, without baseline',
              'alpha = 0.4, with baseline',
              'alpha = 0.4, without baseline']

    for i in range(len(bandits)):
        plt.plot(best_action_counts[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    figure_epsilon_greedy()