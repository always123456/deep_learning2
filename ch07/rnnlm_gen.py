# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm


# 继承自 Rnnlm 类的 RnnlmGen 类： 新添加了生成文本的方法：
class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        """
        文本生成
        :param start_id: 第 1 个单词 ID
        :param skip_ids: 一个单词 ID 列表，它指定的单词不被采样，用于排除 PTB 数据集中的 <unk>/N 等被预处理过的单词
        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            # 根据概率分布 p 采样下一个单词：
            sampled = np.random.choice(len(p), size=1, p=p)
            # size = 1 ：意味着 sampled 是一个包含一个元素的一维数组(Array)，而非一个标量。
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x[0]))  # 这里应当取 x 的索引为 0 的元素再 int()

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)


class BetterRnnlmGen(BetterRnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x[0]))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)