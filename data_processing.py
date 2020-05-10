import os
import re


def get_entities(dir):
    entities = {}
    files = os.listdir(dir)
    files = list(set([file.split('.')[0] for file in files]))  # set()去除重复的数字，无序
    for file in files:
        path = os.path.join(dir, file + ".ann")
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                entity = line.split('\t')[1].split(' ')[0]
                if entity in entities:
                    entities[entity] += 1
                else:
                    entities[entity] = 1
    return entities


def get_id2label_label2id(entities):
    """
    # 得到标签和下标的映射
    """
    # reverse = True 降序 ， reverse = False 升序
    # key 指定可迭代对象中的一个元素来进行排序
    # lambda是一个隐函数，是固定写法；x表示列表中的一个元素，在这里，表示一个元组
    # x只是临时起的一个名字，x[1]表示元组中的第二个元素。
    entities = sorted(entities.items(), key=lambda x:x[1], reverse=True)
    entities = [x[0] for x in entities]
    id2label = []
    id2label.append('O')
    for entity in entities:
        id2label.append('B-'+entity)
        id2label.append('I-' + entity)

    label2id = {id2label[i]: i for i in range(len(id2label))}
    return id2label, label2id


def IsChinese(char):  # 怎么判断一个字符是否为中文
    # 根据汉字编码符号
    if '\u4e00' <= char <= '\u9fff':
        return True
    return False


def split_text(text):
    pattern = '。|，|,|;|；|\.|\？'
    split_index = []  # 用来保存标点符号后被切分的字的id
    for m in re.finditer(pattern, text):
        idx = m.span()[0]
        if text[idx - 1] == '\n':
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isdigit():
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isspace() and text[idx+2].isdigit():
            continue
        if text[idx - 1].islower() and text[idx + 1].islower():
            continue
        if text[idx - 1].isupper() and text[idx + 1].isupper():
            continue
        if text[idx - 1].islower() and text[idx + 1].isdigit():
            continue
        if text[idx - 1].isupper() and text[idx + 1].isdigit():
            continue
        if text[idx - 1].isdigit() and text[idx + 1].islower():
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isupper():
            continue
        if text[idx + 1] in set('.。,，;；'):
            continue
        if text[idx - 1].isspace() and text[idx - 2].isspace() and text[idx - 3] == 'C':
          continue
        if text[idx - 1].isspace() and text[idx - 2] == 'C':
          continue
        if text[idx] == '.' and text[idx + 1:idx + 4] == 'com':
            continue
        split_index.append(idx + 1)

    pattern2 = '\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、'
    pattern2 += '注:|附录|表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,:]+?\n|图 \d|Fig \d'
    pattern2 += '\[Abstract\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论'
    pattern2 += 'and |or |with |by |because of |as well as'
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if (text[idx:idx + 2] in ['or', 'by'] or text[idx:idx + 3] == 'and' or text[idx:idx + 4] == 'with')\
                and (text[idx - 1].islower() or text[idx - 1].isupper()):
                continue
        split_index.append(idx)

    pattern3 = '\n\d\.'
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        if IsChinese(text[idx + 3]):
            split_index.append(idx + 1)
    for m in re.finditer('\n\(\d\)', text):
        idx = m.span()[0]
        split_index.append(idx + 1)
    split_index = list(sorted(set([0, len(text)] + split_index)))  # 保存每一句话的起始位置和终止位置。


    other_index = []
    for i in range(len(split_index) - 1):
        begin = split_index[i]
        end  = split_index[i + 1]
        if text[begin] in '一二三四五六七八九零十' or \
                (text[begin] == '(' and text[begin + 1] in '一二三四五六七八九零十'):
            for j in range(begin, end):
                if text[j] == '\n':\
                    other_index.append(j + 1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))



    other_index = []
    for i in range(len(split_index) - 1):
        b = split_index[i]
        e = split_index[i + 1]
        other_index.append(b)
        if e - b > 150:
            for j in range(b, e):
                if (j+1 - other_index[-1]) > 15:  # other_index[-1]是上一个拆分地方的下标。保证句子长度在15以上。
                    if text[j] == '\n':
                        other_index.append(j + 1)
                    if text[j] == ' ' and text[j - 1].isnumeric() and text[j + 1].isnumeric():
                        other_index.append(j + 1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    for i in range(1, len(split_index) - 1):  # [10, 20] 变相的干掉全部是空格的句子(其实只是合并了句子)
        idx = split_index[i]
        while idx > split_index[i - 1] -1 and text[idx - 1].isspace():
            idx -= 1
        split_index[i] = idx
    split_index = list(sorted(set([0, len(text)] + split_index)))

    # 处理短句子
    temp_idx = []
    i = 0
    while i < len(split_index) - 1:  # i 是一个句子一个句子去走
        b = split_index[i]
        e = split_index[i + 1]

        num_Chinese = 0
        num_English = 0
        if b - e < 15:
            for ch in text[b:e]:
                if IsChinese(ch):
                    num_Chinese += 1
                elif ch.islower() or ch.isupper():
                    num_English += 1
                if num_Chinese + 0.5*num_English > 5:  # 如果汉字加英文超过5个,则单独成为句子
                    temp_idx.append(b)
                    i += 1
                    break
            if num_Chinese + 0.5*num_English <= 5:
                temp_idx.append(b)
                # 解释 i+ 2。如果是0,  10,  20,  30,  45
                # 当前i =1，则i+1=2,split_index[i+1]-split_index[i]=10<15
                # 中英文加起来的数小于5的话，则10还作为b,20这个位置就跳过，直接遍历30的位置\
                # 相当于两句话进行了合并。
                i += 2
        else:
            temp_idx.append(b)
            i += 1
    plit_index = list(sorted(set([0, len(text)] + temp_idx)))
    result = []  # 将分割的句子添加进result列表。

    for i in range(len(split_index)-1):
        result.append(text[split_index[i]:split_index[i+1]])

    # 检查：通过前面的一系列处理，字符有没有丢掉？有没有乱添加？
    s = ''
    for sent in result:
        s += sent
    assert len(s) == len(text)
    return result


    # lens=[split_index[i+1]-split_index[i] for i in range(len(split_index)-1)][:-1]
    # print(max(lens) ,'*********', min(lens))
    # for i in range(len(split_index)-1):
    #     print(i,'||||',text[split_index[i]:split_index[i+1]])












        # lens = [split_index[i + 1] - split_index[i] for i in range(len(split_index) -1)]
        # print(max(lens), min(lens))
        #

if __name__ == '__main__':
    dir = "./wt_pytorch_Medical_BiLSTM_CRF_NER/datas/ruijin_round1_train_small"
    # entities = get_entities(dir)
    # label = get_id2label_label2id(entities)
    files = os.listdir(dir)
    files = list(set(file.split('.')[0] for file in files))
    path = os.path.join(dir, files[1] + '.txt')
    with open(path, 'r', encoding='utf8') as f:
        text = f.read()
        split_text(text)






