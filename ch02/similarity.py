# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
from common.util import preprocess, create_co_matrix, cos_similarity


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']]  #you的单词向量
c1 = C[word_to_id['i']]  #i的单词向量
c2 = C[word_to_id['goodbye']] #goodbye的单词向量

print(cos_similarity(c0, c1))   # 0.7071...
print(cos_similarity(c1, c2))   # 0.9999...

