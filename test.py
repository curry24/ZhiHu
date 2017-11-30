# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

label_reader = pd.read_table('data/topic_info.txt',sep='\t',header=None)
labels = list(label_reader.iloc[:, 0])
my_labels = []
for label in labels:
    my_labels.append(label)
# 建立topic字典
topic_dict = {}
for i, label in enumerate(my_labels):
    topic_dict[label] = i

label_input = pd.read_csv('data/label_example.txt', delimiter='\t', names=['0', '1'])
y=[]
for i in xrange(10):
    # 根据“,”切分话题id
    temp_topic = label_input.iloc[i][1].split(',')
    if (len(temp_topic)>5):
        temp_topic = temp_topic[0:5]
    label = np.zeros(1999)
    for topic in temp_topic:
        # 判断该label是否在label文件中，并得到该行
        label_num = topic_dict[int(topic)]
        label[int(label_num)] = 1
    y.append(list(label))






