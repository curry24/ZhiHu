# coding: utf-8

import pandas as pd
from tqdm import tqdm  # pip install tqdm
# from six.moves import xrange


# 导入question_train_set
reader = pd.read_table('data/question_train_set.txt', sep='\t', header=None)
print reader.iloc[0:5]


# 导入question_topic_eval_set
topic_reader = pd.read_table('data/question_topic_train_set.txt', sep='\t', header=None)
print topic_reader.iloc[0:5]


# 合并title 的词语编号序列和话题 id
data_topic = pd.concat([reader.ix[:, 2], topic_reader.ix[:, 1]], axis=1, ignore_index=True)
print data_topic.iloc[0:5]


# 导入topic_info
label_reader = pd.read_table('data/topic_info.txt', sep='\t', header=None)
print label_reader.iloc[0:5]


# 把标签转为0-1998的编号
labels = list(label_reader.iloc[:, 0])
my_labels = []
for label in labels:
    my_labels.append(label)

# 建立topic字典
topic_dict = {}
for i, label in enumerate(my_labels):
    topic_dict[label] = i

print topic_dict[7739004195693774975]



for i in tqdm(xrange(data_topic.shape[0])):
    new_label = ''
    # 根据“,”切分话题id
    temp_topic = data_topic.iloc[i][1].split(',')
    for topic in temp_topic:
        # 判断该label是否在label文件中，并得到该行
        label_num = topic_dict[int(topic)]
        new_label = new_label + str(label_num) + ','
    data_topic.iloc[i][1] = new_label[:-1]
print data_topic.iloc[:5]



# 保存处理过后的文件
data_topic.to_csv("new_data/data_topic.txt", header=None, index=None, sep='\t')

# 切分成10块保存
for i in xrange(5):
    data_topic_filename = 'new_data/data_topic_5_' + str(i) + '.txt'
    if (i + 1) * 600000 < data_topic.shape[0]:
        data_topic.iloc[i * 600000:(i + 1) * 600000].to_csv(
            data_topic_filename, header=None, index=None, sep='\t')
    else:
        data_topic.iloc[i * 600000:data_topic.shape[0]].to_csv(
            data_topic_filename, header=None, index=None, sep='\t')








