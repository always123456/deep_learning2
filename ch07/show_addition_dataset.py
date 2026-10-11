# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
from dataset import sequence


(x_train, t_train), (x_test, t_test) = sequence.load_data('dataset/addition.txt', seed=1984) # type:ignore
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)
# (45000, 7) (45000, 5)
# (5000, 7) (5000, 5)

print(x_train[0])
print(t_train[0])       # 存放字符 ID
# [ 3  0  2  0  0 11  5]
# [ 6  0 11  7  5]

print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))     # 字符 ID 与字符之间存在映射关系
# 71+118
# _189
