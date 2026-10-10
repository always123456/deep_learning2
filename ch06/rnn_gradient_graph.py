# 简单 RNN 层反向传播时，通过 MatMul 节点，梯度大小的变化：
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体（Windows 常用 SimHei，Mac 可用 Arial Unicode MS 或 PingFang SC）
plt.rcParams['font.sans-serif'] = ['SimHei']   # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

N = 2  # mini-batch的大小
H = 3  # 隐藏状态向量的维数
T = 20  # 时序数据的长度

dh = np.ones((N, H))

np.random.seed(3)   # 固定随机数种子，以完成复现

#Wh = np.random.randn(H, H)     # 梯度爆炸
Wh = np.random.randn(H, H) * 0.5    # 梯度消失

norm_list = []
for t in range(T):
    dh = np.dot(dh, Wh.T)
    norm = np.sqrt(np.sum(dh**2)) / N
    norm_list.append(norm)

print(norm_list)

# 绘制图形
plt.plot(np.arange(len(norm_list)), norm_list)
plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])   # type:ignore
plt.xlabel('time step')
plt.ylabel('梯度范数')
plt.show()
