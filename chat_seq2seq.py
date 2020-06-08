# @Author:sunshine
# @Time  : 2020/6/3 下午1:49
import codecs
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model
from random import shuffle

# 基本参数
maxlen = 256
batch_size = 8
epochs = 10
model_path = 'best_model_v1.weight'
# bert配置
config_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/vocab.txt'

data_file = '/home/chenbing/datasets/diagnosis/train.txt'


def load_data(file):
    data = []
    tmp = ""
    with codecs.open(file, 'r', encoding='utf-8') as rd:
        for line in rd:
            line = line.strip('\n')
            if line:
                if tmp:
                    data.append([tmp.strip('@'), line])
                tmp += line + '@'
            else:
                tmp = ''
    return data


def load_data1(file):
    data = []
    tmp = []
    with codecs.open(file, 'r', encoding='utf-8') as rd:
        for line in rd:
            line = line.strip('\n')
            if line:
                tmp.append(line)
            else:
                data.append(['@'.join(tmp[:-1]), tmp[-1]])
                tmp = []
    return data


data = load_data1(data_file)
shuffle(data)

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, (q, a) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(
                q, a, max_length=maxlen
            )
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """

    @AutoRegressiveDecoder.set_rtype('probas')
    def predict(self, inputs, output_ids, step):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, max_length=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


class Evaluate(keras.callbacks.Callback):
    def __init__(self):
        self.loss = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.loss:
            self.loss = logs['loss']
            model.save_weights(model_path)


def chatting():
    text = []
    while True:
        Q = input('Q:')
        if text == 'quit':
            break
        text.append(Q)
        if len(text) > 3:
            text = text[-3:]
        A = autotitle.generate('@'.join(text))
        print('A:', A)
        text.append(A)


if __name__ == '__main__':
    # 训练
    # evaluator = Evaluate()
    # train_generator = data_generator(data, batch_size)
    # model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=epochs,
    #     callbacks=[evaluator]
    # )

    # 测试
    model.load_weights(model_path)
    chatting()
