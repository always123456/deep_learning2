# coding: utf-8
import sys, os
# sys.path.append('..')     # 相对路径是相对于“当前工作目录”， 而不是相对于“当前脚本文件所在的目录”
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # 直接添加绝对路径
from common.time_layers import *
from common.base_model import BaseModel


# 编码器类：
class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]     # 传向解码器的固定长度向量 h 

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


# 解码器类：
class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


# 组成 seq2seq 模型类：
class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts): # type:ignore
        # https://chat.deepseek.com/a/chat/s/bdca704e-28be-4858-9129-cc85e50a70ed

        # 直觉上理解： 在解码器中，上一个时间步的输出预测结果，要作为下一个时间步的输入。 (自回归的生成方式)
        # 但在训练过程中，可以采用另一种常见技术： Teacher Forcing (教师强制) 来实现并行计算。
        # 假设 ts(目标序列) 是 <start> A B C <END>
        # 则 decoder_xs(解码器输入) ： <START> A B C (去掉最后一个字符)
        #    decoder_ts(解码器目标) ： A B C <END> (去掉第一个字符)
        # 在时间步上对齐：
        # 时间步1： 输入 <START>，目标 A
        # 时间步2： 输入 A，目标 B
        # 时间步3： 输入 B，目标 C
        # 时间步4： 输入 C，目标 <END>
        # 如果，严格按照“上一个时间的输出预测结果，作为当前时间步的输入”，则模型必须一个词一个词地计算(串行)，算完第1个，再算第2个；
        # 使用 Teacher Forcing 后，无论序列多长，decoder_xs 在训练前已知，整个序列可以一次性输入进 RNN，并行计算所有时间步的 Loss

        # 再来解释 Autoregressive(自回归生成)：
        # 该模型，通常出现在推理（测试）阶段；
        # 在推理阶段，没有真实的 decoder_ts（正是我们需要预测的东西），所以代码逻辑就会变成自回归生成模式
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)  # 根据 decoder_ts ，来计算 decoder_xs 预测结果的损失函数
        return loss

    def backward(self, dout=1): # type:ignore
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
