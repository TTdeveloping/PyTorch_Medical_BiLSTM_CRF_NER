import os
import pandas as pd
import shutil
from random import shuffle
from collections import Counter
from data_processing import split_text
from tqdm import tqdm
from cnradical import RunOption, Radical  # 开源一个高效获得汉字偏旁部首、拼音的python库_
import jieba.posseg as psg  # 结巴的词性标注（把一个词的词性标注也作为特征）
import pickle

train_dir = "./wt_pytorch_Medical_BiLSTM_CRF_NER/datas/ruijin_round1_train_small"


def process_text(file_id, split_method=None, split_name='train_small'):
    """
    本函数作用：用于读取文本、切割、打标记  并提取词边界、词性、偏旁部首和拼音等文本特征。
    :param file_name: 文件的名字，不含扩展名（.txt 和 .ann）
    :param split_method:  切割文本的方法（是一个函数）
    :param split_name:  最终保存的文件夹的名字
    :return:
    """
    data = {}  # 用来存所有的字，标记，边界，偏旁部首，拼音等。

    # ****************************************获取句子********************************************

    if split_method == None:  # 如果没有切割文本的方法，就一行当做一个句子处理。
        with open(f'{train_dir}/{file_id}.txt', encoding='utf-8') as f:
            texts = f.readlines()
    else:
        with open(f'{train_dir}/{file_id}.txt', encoding='utf-8') as f:
            texts = f.read()
            texts = split_method(texts)
            # print(texts)
    data['word'] = texts

    # ****************************************获取标签*********************************************
    tag_list = ['O' for sent in texts for word in sent]
    tag = pd.read_csv(f'{train_dir}/{file_id}.ann', header=None, sep='\t')
    for row in range(tag.shape[0]):
        tag_item = tag.iloc[row][1].split(' ')  # 获取实体和实体在文本中的起始位置。 iloc 主要是通过行号获取行数据
        entity_cls, tag_begin, tag_end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        tag_list[tag_begin] = 'B-'+ entity_cls  # 起始位置写上'B-'
        for i in range(tag_begin+1, tag_end):  # 同一个实体类的后面写上'I-'
            tag_list[i] = 'I-' + entity_cls
    assert len([word for sent in texts for word in sent]) == len(tag_list)  # 保证每个字都对应一个标签。

    tag_split_list = []
    sen_b = 0
    sen_e = 0
    for s in texts:
        leng = len(s)
        sen_e += leng
        tag_split_list.append(tag_list[sen_b:sen_e])
        sen_b += leng
    data['label'] = tag_split_list

    #**************************************提取词边界和词性特征*************************************
    word_bounds = ['M' for s_tag in tag_list]  # 保存词的边界(B, E, M, S)，刚初始化时，为每一个字都打上'M'的标记。
    word_flags = []  # 保存每个字的词性特征。
    for sent in texts:
        for word, flag in psg.cut(sent):  # 中国,ns    成人,n     2,m    型,k     糖尿病,n    HBA1C,eng
            if len(word) == 1:
                start = len(word_flags)
                word_bounds[start] = 'S'
                word_flags.append(flag)
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'
                word_flags += [flag] * len(word)
                end = len(word_flags) - 1
                word_bounds[end] = 'E'

    bounds = []
    flags = []
    sen_b = 0
    sen_e = 0
    for s in texts:
        leng = len(s)
        sen_e += leng
        bounds.append(word_bounds[sen_b:sen_e])
        flags.append(word_flags[sen_b: sen_e])
        sen_b += leng
    data['bound'] = bounds
    data['flag'] = flags
    # return texts[0], tag_split_list[0], bounds[0], flags[0]

    # **************************************获取拼音特征*************************************
    radical = Radical(RunOption.Radical)
    pinyin = Radical(RunOption.Pinyin)

    # 提取每个字的偏旁部首.对于没有偏旁部首的字标上PAD
    data['radical'] = radical_out = [[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    # 提取每个字的拼音.对于没有拼音的字标上PAD
    data['pinyin'] = pinyin_out = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]
    # return texts[0], tag_split_list[0], bounds[0], flags[0], data['radical'][0], data['pinyin'][0]

    # ****************************************存储数据****************************************
    num_samples = len(texts)  # 获取总共有多少句话
    num_col = len(data.keys())  # 获取特征项的个数（列数）
    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i])for v in data.values()]))
        dataset += records + [['sep'] * num_col]  # 每存完一句话，需要一行sep进行隔离。
    dataset = dataset[: -1]  # 最后一行的分隔符sep不要
    dataset = pd.DataFrame(dataset, columns=data.keys())  # 转换成dataframe（一个表格型的数据结构）

    # f-string用大括号 {} 表示被替换字段，其中直接填入替换内容：
    save_path = f'./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/{split_name}/{file_id}.csv'


    def clean_word(word):
        if word == '\n':
            return 'LB'
        if word in [' ', '\t', '\u2003']:
            return 'SPACE'
        # 把所有数字变成一种符号（命名实体识别的任务不关心是什么数字，只知道是数字就可以了）
        # 提高泛化能力。
        if word.isdigit():
            return 'num'
        return word

    dataset['word'] = dataset['word'].apply(clean_word)
    # dataset.to_csv(save_path, index=False, encoding='utf-8')
    dataset.to_csv(save_path, index=False, encoding='utf-8')


def multi_file_process(split_method=None, train_rate=0.8):
    if os.path.exists('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/'):
        shutil.rmtree('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/')
    if not os.path.exists('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train/'):
        os.makedirs('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train_small')
        os.makedirs('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/test_small')
        # os.makedirs('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train')
        # os.makedirs('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/test')


    idxs = list(set([file.split('.')[0] for file in os.listdir(train_dir)]))
    shuffle(idxs)
    index = int(len(idxs)* train_rate)
    train_ids = idxs[:index]  # 训练集文件
    test_ids = idxs[index:]  # 测试集文件

    # python中使用multiprocessing模块实现多进程
    import multiprocessing as mp
    cpu_nums = mp.cpu_count()  # 获取机器cpu的个数
    pool = mp.Pool(cpu_nums)  # 几个cpu就做几个并行。
    results = []
    for file_id in train_ids:
        # result = pool.apply_async(process_text, args=(file_id, split_method, 'train'))
        result = pool.apply_async(process_text, args=(file_id, split_method, 'train_small'))
        results.append(result)
        results.append(result)
    for file_id in test_ids:
        # result = pool.apply_async(process_text, args=(file_id, split_method, 'test'))
        result = pool.apply_async(process_text, args=(file_id, split_method, 'test_small'))
        results.append(result)
    pool.close()
    pool.join()
    [r.get() for r in results]


def mapping(data, fre=10, is_word=False, sep='sep', is_label=False):
    count = Counter(data)  # Counter参数是一个列表，返回一个字典，字典中包含列表中所有的字，和对应的词频。
    if sep is not None:
        count.pop(sep)
    if is_word:
        # PAD是为了填充一个批次中短句子。
        # UNK是表示字典中没有的字。
        count['PAD'] = 100000001
        count['UNK'] = 100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        # for couple in data:  # 去掉词频小于fre的单词
        #     if couple[1] > fre:
        #         final_word.append(couple[0])
        data = [couple[0] for couple in data if couple[1] > fre]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [couple[0] for couple in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}

    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
        data = [couple[0] for couple in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    return id2item, item2id


def get_dict():
    map_dict = {}
    from glob import glob  # 遍历文件的工具
    all_word = []
    all_label = []
    all_bound = []
    all_flag = []
    all_radical = []
    all_pinyin = []

    # 遍历train和test下面所有的csv文件
    # for file in glob('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train/*.csv') + \
    #     glob('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/test/*.csv'):
    for file in glob('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train_small/*.csv') + \
                glob('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/test_small/*.csv'):
        df = pd.read_csv(file, sep=',')
        all_word += df['word'].tolist()
        all_label += df['label'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()
        all_radical += df['radical'].tolist()
        all_pinyin += df['pinyin'].tolist()
    # map_dict['word'] = mapping(all_word, fre=20, is_word=True)
    map_dict['word'] = mapping(all_word, fre=0, is_word=True)
    map_dict['label'] = mapping(all_label, is_label=True)
    map_dict['bound'] = mapping(all_bound)
    map_dict['flag'] = mapping(all_flag)
    map_dict['radical'] = mapping(all_radical)
    map_dict['pinyin'] = mapping(all_pinyin)

    with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/dict.pkl', 'wb') as f:
        pickle.dump(map_dict, f)


if __name__ == '__main__':
    process_text('0', split_method=split_text, split_name='train_small')
    multi_file_process(split_text)
    get_dict()
    # process_text('0', split_method=split_text)
    # multi_file_process()
    # print(os.listdir(train_dir))
    # import multiprocessing as mp
    # cpu_num = mp.cpu_count()
    # print(cpu_num)
    # from glob import glob
    # # for file in glob('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train/*.csv') + \
    # #     glob('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/test/*.csv'):
    # all_word = []
    # file = './wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/train/1.csv'
    # df = pd.read_csv(file, sep=',')
    # all_word += df['word'].tolist()
    # count = Counter(all_word)
    # data = sorted(count.items(), key=lambda x: x[1], reverse=True)  # key 具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
    # print(data)
    # with open('./wt_pytorch_Medical_BiLSTM_CRF_NER/data/data_preparation/dict.pkl', 'rb') as f:
    #    data = pickle.load(f)
    # print(data['bound'])





