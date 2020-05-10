import pandas as pd  # 用来读取csv文件
import pickle  # 将字转化为下标。
import numpy as np
import math
import random
from tqdm import tqdm
import os


def get_data_with_windows(name='train_small'):
    """
    函数作用：拼接（两个句子，三个句子拼接）、将所有字换成对应的下标
    :param name:
    :return:
    """
    with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)  # {'word':([i2w], {w21}), 'label':([i2w], {w21}), 'bound':([i2w], {w21}),...}

    def item2id(data, w2i):
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]

    results = []
    root = os.path.join('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/' + name)
    # 小数据
    # root = os.path.join('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/' + "train_small")
    # print(os.listdir(root))
    # print('**************'* 20)
    # print(list(os.listdir(root)))
    # exit()
    files = list(os.listdir(root))  # 将'train'文件夹下面的所有文件的名字读到一个列表中。['0.csv', '1.csv', '10.csv',...]
    for file in tqdm(files):  # tqdm 进度条来显示遍历files列表的进度。
        result = []
        path = os.path.join(root, file)
        samples = pd.read_csv(path, sep=',')
        num_samples = len(samples)  # 每个文件中数据的行数
        # samples['word'] == 'sep'判断samples的每一行中的'word'是否是'sep'，返回的是布尔类型。
        # samples[samples['word'] == 'sep'] 将返回True的每一行输出。
        # samples[samples['word'] == 'sep'].index.tolist() 将判断为True的所有行的行号读下来存进列表。
        sep_index = [-1] + samples[samples['word'] == 'sep'].index.tolist() + [num_samples]  # 获取句子间隔的id
        # --------------------------获取句子并将句子全部转换成ID--------------------------
        for i in range(len(sep_index)-1):
            start = sep_index[i] + 1
            end =sep_index[i + 1]
            data = []
            # 对每一个特征进行处理
            for feature in samples.columns:
                data.append(item2id(list(samples[feature])[start:end], map_dict[feature][1]))
            # result列表存放的是句子列表，每个句子列表中存放的是6个特征列表  result = [[[],[],[],[],[],[]],[...],...,[...]]
            result.append(data)

        # -----------------------------数据增强----------------------------------------------
        # 进行拼接，也可以不拼接。这样的话，学习的句子有长有短，可以提高处理长短句子的能力（即数据增强）

        # 两个句子拼接在一起（其实就是短句子变长句子）
        two = []
        for i in range(len(result)-1):
            first = result[i]
            second = result[i + 1]
            # 对应的特征进行拼接。如第一句话和第二句话的'word','label',...,进行拼接
            two.append(first[k] + second[k] for k in range(len(first)))

        # 三个句子拼接在一起（其实就是短句子变长句子）
        three = []
        for i in range(len(result) - 2):
            # first = ['word','label','bound','flag','radical','pinyin']
            first = result[i]  # result = [[],[],[],[],.......[]]
            second = result[i + 1]
            third = result[i + 2]
            # 对应的特征进行拼接
            three.append(first[k] + second[k] + third[k] for k in range(len(first)))
        results.extend(result + two + three)
    with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/' + name + '.pkl', 'wb') as f:
         pickle.dump(results, f)
    # return results


class BatchManager(object):
    def __init__(self, batch_size, name='train_small'):
        with open('./data/data_preparation/' + name + '.pkl', 'rb') as f:
        # with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/' + name + '.pkl', 'rb') as f:
            data = pickle.load(f)
        self.batch_data = self.sort_and_pad(data, batch_size)
        self.len_data = len(self.batch_data)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))  # ceil() 函数返回数字的上入整数。总共有多少批次
        # print(data)
        # data = [[[2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4]],
        #         [[2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4]],
        #         [[2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4]],
        #         [[2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4]],
        #         [[2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4]],
        #         [[2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4], [2,3,4]]]
        sorted_data = sorted(data, key=lambda x: len(x[0]), reverse=False) # 按照句子长度排序
        batch_data = list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size): (i+1)*int(batch_size)]))
        return batch_data

    @staticmethod
    def pad_data(data):
        words = []
        labels = []
        bounds = []
        flags = []
        radicals = []
        pinyins = []
        max_length = max([len(sentence[0]) for sentence in data])  # len(data[-1][0])
        for line in data:
            word, label, bound, flag, radical, pinyin = line
            padding = [0] * (max_length - len(word))
            words.append(word + padding)
            labels.append(label + padding)
            bounds.append(bound + padding)
            flags.append(flag + padding)
            radicals.append(radical + padding)
            pinyins.append(pinyin + padding)

        return [words, labels, bounds, flags, radicals, pinyins]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]  # 每次拿一个批次处理。


if __name__ == '__main__':
    # get_data_with_windows('test_small')
    # get_data_with_windows('train_small')
    train_data = BatchManager(10)
    with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/dict.pkl', 'rb') as f:
        map_dict = pickle.load(f)
    # print(map_dict)
