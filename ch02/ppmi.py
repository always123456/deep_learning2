# 基于 Python 的语料库预处理： 共现矩阵和 PPMI 矩阵
# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi, create_contexts_target


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
# contexts, target = create_contexts_target(corpus, 1)
# print(contexts)
# print(target)
# print(word_to_id)
# print(id_to_word)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)  # 有效位数为3位
print('covariance matrix')
print(C)
print('-'*50)
print('PPMI')
print(W)

