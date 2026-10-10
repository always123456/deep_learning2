# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
from common.util import most_similar, analogy
import pickle


pkl_file = 'ch04/cbow_params.pkl'
# pkl_file = 'skipgram_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']

# most similar task
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# analogy task (类推问题: king - man + woman = ?)
print('-'*50)
analogy('king', 'man', 'queen',  word_to_id, id_to_word, word_vecs)
analogy('take', 'took', 'go',  word_to_id, id_to_word, word_vecs)    # 表示编码了时态相关的信息
analogy('car', 'cars', 'child',  word_to_id, id_to_word, word_vecs)  # 单词的单复数模式也被捕获
analogy('good', 'better', 'bad',  word_to_id, id_to_word, word_vecs) # 比较级模式也被捕获 

# 语料库偏小 -> 类推问题处理不够良好
