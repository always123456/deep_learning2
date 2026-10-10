# 基于 SVD 的降维：
# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)

# 共现矩阵：
C = create_co_matrix(corpus, vocab_size, window_size=1)

# PPMI 矩阵：
W = ppmi(C)

# SVD 降维
U, S, V = np.linalg.svd(W)

np.set_printoptions(precision=3)  # 有效位数为3位
print(C[0])
print(W[0])
print(U[0])

# 再次对密集向量矩阵 U 做降维，比如降成二维向量，取出前两个元素即可
# 密集向量矩阵 U 的列对应的奇异值降序排列的，选最高奇异值的两列即可将行向量降维成二维向量
# plot：画二维单词向量的表示
for word, word_id in word_to_id.items():
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))

plt.scatter(U[:,0], U[:,1], alpha=0.5)
plt.show()
