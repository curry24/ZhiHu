# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import math
def read_data(data_file_name, max_i =-1):
    x = []
    y = []
    j = 0

    label_reader = pd.read_table('data/topic_info.txt', sep='\t', header=None)
    labels = list(label_reader.iloc[:, 0])
    my_labels = []
    for label in labels:
        my_labels.append(label)
    # 建立topic字典
    topic_dict = {}
    for i, label in enumerate(my_labels):
        topic_dict[label] = i

    label_input = pd.read_csv('data/question_topic_train_set.txt', delimiter='\t', names=['0', '1'], nrows=10)
    for i in xrange(label_input.shape[0]):
        # 根据“,”切分话题id
        temp_topic = label_input.iloc[i][1].split(',')
        if (len(temp_topic) > 5):
            temp_topic = temp_topic[0:5]
        label = np.zeros(1999)
        for topic in temp_topic:
            # 判断该label是否在label文件中，并得到该行
            label_num = topic_dict[int(topic)]
            label[int(label_num)] = 1
        y.append(list(label))
    # print y[0]
    # print y[3]
    # print y[4]


    with open(data_file_name) as f:
        for line in f:
            features = line
            xi = []
            for feature in features.split():
                xi.append(float(feature))
            x.append(xi)
            if j % 5 == 0:
                print 'load data %d' % j
            j+=1
            if max_i > 0 and j > max_i:
                break


    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)

    print x.shape
    print x[0].shape
    print x[1].shape
    print x[9].shape



    return (x, y)


x, y = read_data('cnn_vec/train.txt')


def eval(predict_label_and_marked_label_list):
    """
    :param predict_label_and_marked_label_list: 一个元组列表。例如
    [ ([1, 2, 3, 4, 5], [4, 5, 6, 7]),
      ([3, 2, 1, 4, 7], [5, 7, 3])
     ]
    需要注意这里 predict_label 是去重复的，例如 [1,2,3,2,4,1,6]，去重后变成[1,2,3,4,6]

    marked_label_list 本身没有顺序性，但提交结果有，例如上例的命中情况分别为
    [0，0，0，1，1]   (4，5命中)
    [1，0，0，0，1]   (3，7命中)
    """
    right_label_num = 0  # 总命中标签数量
    right_label_at_pos_num = [0, 0, 0, 0, 0]  # 在各个位置上总命中数量
    sample_num = 0  # 总问题数量
    all_marked_label_num = 0  # 总标签数量
    for predict_labels, marked_labels in predict_label_and_marked_label_list:
        sample_num += 1
        marked_label_set = set(marked_labels)
        all_marked_label_num += len(marked_label_set)
        for pos, label in zip(range(0, min(len(predict_labels), 5)), predict_labels):
            if label in marked_label_set:  # 命中
                right_label_num += 1
                right_label_at_pos_num[pos] += 1
    precision = 0.0
    for pos, right_num in zip(range(0, 5), right_label_at_pos_num):
        precision += ((right_num / float(sample_num))) / math.log(2.0 + pos)  # 下标0-4 映射到 pos1-5 + 1，所以最终+2
    recall = float(right_label_num) / all_marked_label_num
    return 2 * (precision * recall) / (precision + recall)
