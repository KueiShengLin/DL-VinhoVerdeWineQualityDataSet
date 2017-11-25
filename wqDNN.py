import csv
import os
import tensorflow as tf
import math
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


f = open('winequality-white.csv', 'r')
FEATURE = []    # list of the member and it's feature
FEATURE_NAME = []   # feature name ex:ph, density, alcohol ...
# ATTRIBUTE = []  # list of the feature
# ATTRIBUTE_ID = []   # cluster id of the each feature
# INPUT_DATA = []     # convert the feature which is str to the cluster id
CV = 10         # cross validation amount


def catch_feature():
    global FEATURE, FEATURE_NAME
    # read csv
    for row_id, row in enumerate(csv.reader(f, delimiter=';')):
        FEATURE.append(row)
        if row_id != 0:
            for fr_id, fr in enumerate(FEATURE[row_id]):
                FEATURE[row_id][fr_id] = float(fr)

        # if row_id == 0:
        #     for i in range(len(FEATURE[0])):
        #         ATTRIBUTE.append([])
        #         # ATTRIBUTE_ID.append({})
        #     continue
        # for ele_id, element in enumerate(row):
        #     ATTRIBUTE[ele_id].append(element)
    f.close()
    FEATURE_NAME = FEATURE[0]
    del FEATURE[0]
    # attribute cluster
    # for i in range(len(FEATURE[0])):
    #     try:
    #         for ele_id, element in enumerate(ATTRIBUTE[i]):
    #             ATTRIBUTE[i][ele_id] = int(element)
    #         ATTRIBUTE_ID[i]['int'] = 0
    #     except Exception:
    #         a_id = 0
    #         for v, k in enumerate(ATTRIBUTE[i]):
    #             if k in ATTRIBUTE_ID[i]:
    #                 continue
    #             else:
    #                 ATTRIBUTE_ID[i][k] = a_id
    #                 a_id += 1

    # # delete training data answer (answer is in ATTRIBUTE[-1])
    # for f_id in range(len(FEATURE)):
    #     del FEATURE[f_id][-1]

    # create input data
    # for feature_id, feature_ele in enumerate(FEATURE[1:]):
    #     input_t = []
    #     for ele_id, ele in enumerate(feature_ele):
    #         try:
    #             FEATURE[feature_id + 1][ele_id] = int(ele)
    #             input_t.append(int(ele))
    #         except Exception:
    #             FEATURE[feature_id + 1][ele_id] = ATTRIBUTE_ID[ele_id][ele]
    #             for i in range(len(ATTRIBUTE_ID[ele_id])):
    #                 if i == ATTRIBUTE_ID[ele_id][ele]:
    #                     input_t.append(1)
    #                 else:
    #                     input_t.append(0)
    #     INPUT_DATA.append(input_t)


# divide input data into CV parts
def cross_validation():
    global CV, FEATURE
    split_num = int(math.ceil(len(FEATURE)/CV))
    split_list = []
    for num in range(0, len(FEATURE), split_num):
        sp = []
        # sp = [train_data.values[num + i].tolist() for i in range(split_num) if num+i < len(train_data)]
        for i in range(split_num):
            try:
                sp.append(FEATURE[num + i])
            except:
                break
        split_list.append(sp)
    return split_list


# tensorflow add layer
def add_layer(inputs, input_tensors, output_tensors, activation_function=None):
    w = tf.Variable(tf.random_normal([input_tensors, output_tensors]))
    b = tf.Variable(tf.truncated_normal([output_tensors]))
    formula = tf.add(tf.matmul(inputs, w), b)  # matmul = dot
    if activation_function is None:
        outputs = formula
    else:
        outputs = activation_function(formula)
    return outputs


# testing accuracy
def ans_predict(ans, my_prediction):
    my_prediction = list(my_prediction)
    prediction = my_prediction.index(max(my_prediction))
    if prediction == (ans - 3):
        return 1
    else:
        return 0


catch_feature()
print('catch feature down')

# layer units define
INPUT = len(FEATURE[0]) - 1
HIDDEN = INPUT
OUTPUT = 6

y_feed = tf.placeholder(tf.int32, [1])  # no:0 yes:1
y_hat = tf.one_hot((y_feed-3), OUTPUT)  # 3~8 => 0~5
input_feed = tf.placeholder(tf.float32, [None, INPUT])

# layer define you can add more hidden in there
hidden_layer = add_layer(input_feed, input_tensors=INPUT, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
hidden_layer2 = add_layer(hidden_layer, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
hidden_layer3 = add_layer(hidden_layer2, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
hidden_layer4 = add_layer(hidden_layer3, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
hidden_layer5 = add_layer(hidden_layer4, input_tensors=HIDDEN, output_tensors=HIDDEN, activation_function=tf.nn.sigmoid)
output_layer = add_layer(hidden_layer5, input_tensors=HIDDEN, output_tensors=OUTPUT, activation_function=tf.nn.softmax)

loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_hat, logits=output_layer))   # loss function use cross entropy
# loss = tf.losses.mean_squared_error(y_hat, output_layer)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)  # use adadeleta(adagrand的加強版) change learning rate
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options))
sess.run(init)


print('training')

cv_list = cross_validation()
for cv in range(CV):
    print('cv' + str(cv))

    if cv != 0:
        cv_list[0], cv_list[cv] = cv_list[cv], cv_list[0]

    for iteration in range(10000001):

        total_loss = 0
        total_num = 0
        for i in range(1, len(cv_list)):
            for fid, feature in enumerate(cv_list[i]):
                _, cost = sess.run([train, loss], feed_dict={input_feed: [feature[0:-1]], y_feed: [feature[-1]]})
                total_loss += cost
                total_num += 1

        if iteration % 100 == 0:
            print('iteration:', iteration)
            print('loss:')
            print(total_loss / total_num)
            print('testing')
            acc = 0
            for _, test_data in enumerate(cv_list[0]):
                output = sess.run(output_layer, feed_dict={input_feed: [test_data[0:-1]], y_feed: [test_data[-1]]})
                acc += ans_predict(test_data[-1], output[0])
            print(acc / len(cv_list[0]))

    sess.run(init)

print('down')
#
