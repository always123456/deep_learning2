# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)


model = BetterRnnlmGen()
model.load_params('ch06/BetterRnnlm.pkl')   # 这里权重文件丢失！！！！！   已通过租用的 GPU 计算出来

# 设定start字符和skip字符
# 给该模型输入"you"，并生成后续的单词：
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 文本生成
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print(txt)

model.reset_state()

# 给该模型输入"the meaning of life is"，并生成后续的单词：
start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)
