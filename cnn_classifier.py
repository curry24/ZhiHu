#! /usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from TextCNN import *
import data_helpers
import utils
import pandas as pd
import math

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
    print 'precision: %f' % precision
    print 'recall: %f' % recall
    if (precision+recall)== 0:
        return 0
    else: return  2*(precision * recall)/(precision + recall)


# def get_measure(predict_y, acture_y):
#     tp = 0
#     fp = 0
#     tn = 0
#     fn = 0
#     length = len(acture_y)
#     for i in xrange(length):
#         if acture_y[i] > 0:
#             if predict_y[i] > 0:
#                 tp += 1
#             else:
#                 fn += 1
#         else:
#             if predict_y[i] > 0:
#                 fp += 1
#             else:
#                 tn += 1
#     acc = (tp + tn) / float(tp + tn + fp + fn)
#     pre = (tp) / float(tp + fp)
#     rec = (tp) / float(tp + fn)
#     return tp, fp, tn, fn, acc, pre, rec, float(tp * 2) / (tp * 2 + fp + fn)


# def read_data(data_file_name, max_i =-1):
#     x = []
#     y = []
#     i = 0
#     with open(data_file_name) as f:
#         for line in f:
#             label_1, features = line.strip().split(None, 1)
#             label_2 = float(label_1)
#             # if label == 1:
#             #     yi = [0.0, 1.0]
#             # else:
#             #     yi = [1.0, 0.0]
#             # label = list(label)
#             label_3 = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,label_2]
#             y.append(label_3)
#             xi = []
#             for feature in features.split():
#                 xi.append(float(feature))
#             x.append(xi)
#             if i % 5 == 0:
#                 print 'load data %d' % i
#             i+=1
#             if max_i > 0 and i > max_i:
#                 break
#     x = np.array(x).astype(np.float32)
#     y = np.array(y).astype(np.float32)
#     return (x, y)

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

    # 加载训练的label
    label_input = pd.read_csv('data/question_topic_train_set.txt', delimiter='\t', names=['0', '1'], nrows=10000)
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

    # 加载训练的矩阵向量
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
    return (x, y)


def read_test(data_file_name, max_i =-1):
    x = []
    j = 0
    # 加载训练的矩阵向量
    with open(data_file_name) as f:
        for line in f:
            features = line
            xi = []
            for feature in features.split():
                xi.append(float(feature))
            x.append(xi)
            if j % 2000 == 0:
                print 'load data %d' % j
            j+=1
            if max_i > 0 and j > max_i:
                break
    x = np.array(x).astype(np.float32)
    print x.shape
    return x

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 256, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 1024, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
need_train = False
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("need train = " + str(need_train))

if need_train:
    x, y = read_data('cnn_vec/train.txt')
    print 'load training data is ok'

    x_train = []
    y_train = []
    x_dev = []
    y_dev = []
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x[:dev_sample_index], x[dev_sample_index:]
    y_train, y_dev = y[:dev_sample_index], y[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

else:
    x_test = read_test('cnn_vec/test_0.txt')
    print 'load test data is ok'

# Training
# ==================================================

max_acc = 0.0

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options=gpu_options)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=66, # train 或者test 切换的时候要记得修改
            num_classes=1999,
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "cnn_runs"))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_prefix = "cnn_runs/cnn_model"
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
        # saver = tf.train.Saver()
        predict_top_5 = tf.nn.top_k(cnn.scores, k=5)
        label_top_5 = tf.nn.top_k(cnn.input_y, k=5)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        if not need_train:
            saver.restore(sess, checkpoint_prefix)


        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _, step, summaries, predict_5, label_5, loss = sess.run(
                [train_op, global_step, train_summary_op, predict_top_5, label_top_5, cnn.loss],
                feed_dict)

            # if step % 20 == 0:
            #     # time_str = datetime.datetime.now().isoformat()
            #     # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            #     print ("label:", label_5[1][:5])
            #     print ("predict:", predict_5[1][:5])
            #     print ("predict:", predict_5[0][:5])
            #     print ("loss:", loss)
            #     predict_label_and_marked_label_list = []
            #     for predict, label in zip(predict_5[1], label_5[1]):
            #         predict_label_and_marked_label_list.append((list(predict), list(label)))
            #     score = eval(predict_label_and_marked_label_list)
            #     print("score:", score)
            train_summary_writer.add_summary(summaries, step)


        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 1.0
            }
            step, summaries, predict_5, label_5, loss = sess.run(
                [global_step, dev_summary_op, predict_top_5, label_top_5, cnn.loss],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            print ("label:", label_5[1][:5])
            print ("predict:", predict_5[1][:5])
            print ("predict:", predict_5[0][:5])
            print ("loss:", loss)
            predict_label_and_marked_label_list = []
            for predict, label in zip(predict_5[1], label_5[1]):
                predict_label_and_marked_label_list.append((list(predict), list(label)))
            score = eval(predict_label_and_marked_label_list)
            print("score:", score)

            if writer:
                writer.add_summary(summaries, step)

            # measure = get_measure(np.array(predict).reshape(-1), y_dev[:, 1].reshape(-1))
            # print 'test:tp=%6d\tfp=%6d\ttn=%6d\tfn=%d\tacc=%f\tpre=%f\trec=%f\tF=%f\n\n' % measure
            # with open('result/cnn.txt', 'a') as f:
            #     f.write('test:tp=%6d\tfp=%6d\ttn=%6d\tfn=%d\tacc=%f\tpre=%f\trec=%f\tF=%f\n\n' % measure)



        # Generate batches
        if need_train:
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix)
                        print("Saved model checkpoint to {}\n".format(path))
        else:
            i = 0
            batches = data_helpers.batch_iter(list(x_test), 1000, 1)
            for batch in batches:
                i = i + 1
                feed_dict = {
                    cnn.input_x: batch,
                    cnn.dropout_keep_prob: 1.0
                }
                predict_5 = sess.run(predict_top_5, feed_dict=feed_dict)
                if i == 1:
                    predict = predict_5[1]
                else:
                    predict = np.concatenate((predict, predict_5[1]))
                if (i % 5 == 0):
                    print ("Evaluation:step", i)

            np.savetxt("predict_0.txt", predict, fmt='%d')


            # predict, acc = sess.run([cnn.predictions, cnn.accuracy], feed_dict=feed_dict)
            # measure = get_measure(np.array(predict).reshape(-1), y_dev[:, 1].reshape(-1))
            # print acc
            # print 'test:tp=%6d\tfp=%6d\ttn=%6d\tfn=%d\tacc=%f\tpre=%f\trec=%f\tF=%f\n\n' % measure
            # with open('result/cnn_result.txt', 'a') as f:
            #     f.write('test:tp=%6d\tfp=%6d\ttn=%6d\tfn=%d\tacc=%f\tpre=%f\trec=%f\tF=%f\n\n' % measure)
