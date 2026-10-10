# 学习数据的准备， 基本流程：
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
from common.util import preprocess, create_contexts_target, convert_one_hot
import numpy as np

text = "You say hello and I say goodbye"
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)    # 词汇数量
target = convert_one_hot(target, vocab_size=vocab_size)
contexts = convert_one_hot(contexts, vocab_size=vocab_size)

print(target)
print(contexts)