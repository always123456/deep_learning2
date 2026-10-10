import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
import numpy as np
from common.util import preprocess, create_co_matrix

text = "You say hello and I say goodbye."       # 实验语料库
corpus, word_to_id, id_to_word = preprocess(text)

# C = np.array(           # 共现矩阵
#     [
#         [0, 1, 0, 0, 0, 0, 0],
#         [1, 0, 1, 0, 1, 1, 0],
#         [0, 1, 0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 1, 0, 0],
#         [0, 1, 0, 1, 0, 0, 0],
#         [0, 1, 0, 0, 0, 0, 1],
#     ], dtype=np.int32
# )

C = create_co_matrix(corpus, len(word_to_id), 1)

print(C[0])     # 单词 ID 为0的向量
print(C[4])     # 单词 ID 为4的向量