# coding: utf-8
# 使用 seq2seq 模型进行简单加法运算的学习：
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq


# 读入数据集(简单的加法数据集)
(x_train, t_train), (x_test, t_test) = sequence.load_data('dataset/addition.txt') # type:ignore
char_to_id, id_to_char = sequence.get_vocab()

# Reverse input? =================================================  反转输入数据，有效提高了模型正确率
is_reverse = False  # True
if is_reverse:
    x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
# ================================================================

# 设定超参数
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# 生成模型、优化器、训练器：
# Normal or Peeky? ==============================================
# 在 max_epoch=25的情况下，Normal seq2seq 模型正确率大约只有 10% 左右：
#model = Seq2seq(vocab_size, wordvec_size, hidden_size)
model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)
# ================================================================
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)    # 采用正确率作为评价指标

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('val acc %.3f%%' % (acc * 100))

# 绘制图形
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker='o')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()


# 模型正确率评价结果：
# Normal:     10%
# Normal + Reversed:      50%
# PeekySeq2seq:      81%
# PeekySeq2seq + Reversed:       97%
