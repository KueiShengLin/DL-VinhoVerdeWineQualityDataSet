import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import os
import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# load dataset & normalize
def preprocess():
    dataset = pd.read_csv("./dataset/winequality-red-dot.csv")
    #Normalize
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(dataset)
    a = pd.DataFrame(x_scaled, columns=dataset.columns)
    a.drop('quality', axis=1, inplace=True)

    # split train and test set
    quality = list(dataset['quality'])
    train, test = train_test_split(a, test_size=0.1, random_state=42)
    train_q, test_q = train_test_split(quality, test_size=0.1, random_state=42)
    return train, test, train_q, test_q


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


train_data, test_data, train_y, test_y = preprocess()
np.savetxt("./prediction/train_label.txt", train_y, delimiter=',')
np.savetxt("./prediction/test_label.txt", test_y, delimiter=',')
print('preprocess down')

# layer units define
INPUT = len(train_data.values[0]) #11
HIDDEN = 5
OUTPUT = 6


y_feed = tf.placeholder(tf.int32, [1])  # no:0 yes:1
y_hat = tf.one_hot((y_feed-3), OUTPUT)  # 3~8 => 0~5
x_feed = tf.placeholder(tf.float32, [None, INPUT])


# pca encode (acc is bad so i not use it)
PCA = 2
encode_weight = {"wen1": np.loadtxt("./encode/w_pca1.txt", delimiter=','),
                 "wen2": np.loadtxt("./encode/w_pca2.txt", delimiter=','),
                 "wen3": np.loadtxt("./encode/w_pca3.txt", delimiter=',')}
encode_bias = {"ben1": np.loadtxt("./encode/b_pca1.txt", delimiter=','),
               "ben2": np.loadtxt("./encode/b_pca2.txt", delimiter=','),
               "ben3": np.loadtxt("./encode/b_pca3.txt", delimiter=',')}

wen1 = tf.constant(encode_weight['wen1'], name='wen1', dtype=tf.float32)
ben1 = tf.constant(encode_bias['ben1'], name='ben1', dtype=tf.float32)
encode1 = tf.add(tf.matmul(x_feed, wen1), ben1)  # matmul = dot

wen2 = tf.constant(encode_weight['wen2'], name='wen2', dtype=tf.float32)
ben2 = tf.constant(encode_bias['ben2'], name='ben2', dtype=tf.float32)
encode2 = tf.add(tf.matmul(encode1, wen2), ben2)  # matmul = dot

wen3 = tf.constant(encode_weight['wen3'], name='wen3', dtype=tf.float32)
ben3 = tf.constant(encode_bias['ben3'], name='ben3', dtype=tf.float32)
x_pca = tf.add(tf.matmul(encode2, wen3), ben3)  # matmul = dot
#

# hidden_layer 1
w1 = tf.Variable(tf.random_uniform([INPUT, 4], minval=-1, maxval=1))
b1 = tf.Variable(tf.random_uniform([4], minval=-1, maxval=1))
formula1 = tf.add(tf.matmul(x_feed, w1), b1)  # matmul = dot
hidden_layer1 = tf.nn.sigmoid(formula1)
# hidden_layer 2
w2 = tf.Variable(tf.random_uniform([4, HIDDEN], minval=-1, maxval=1))
b2 = tf.Variable(tf.random_uniform([HIDDEN], minval=-1, maxval=1))
formula2 = tf.add(tf.matmul(hidden_layer1, w2), b2)  # matmul = dot
hidden_layer2 = tf.nn.sigmoid(formula2)
# output
w3 = tf.Variable(tf.random_uniform([HIDDEN, OUTPUT], minval=-1, maxval=1))
b3 = tf.Variable(tf.random_uniform([OUTPUT], minval=-1, maxval=1))
output_formula = tf.add(tf.matmul(hidden_layer2, w3), b3)  # matmul = dot
output_layer = tf.nn.softmax(output_formula)

loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_hat, logits=output_layer))   # loss function use cross entropy
# loss = tf.losses.mean_squared_error(y_hat, output_layer)
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
train = optimizer.minimize(loss)

# Accuracy if argmax(output_layer) == label: return 1
correct_prediction = tf.equal(tf.arg_max(output_layer, 1), tf.arg_max(y_hat, 1))
acc = tf.cast(correct_prediction, tf.float32)


init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))
sess.run(init)


print('training')


data_num = len(train_data.values)
test_num = len(test_data.values)

for iteration in range(1000001):
    cost_total = 0  # loss score
    acc_total = 0  # train set accuracy
    quality_acc = 0  # train set kaggle accuracy
    train_prediction_list = []
    for pid, point in enumerate(train_data.values):
        _, cost, output_acc, train_prediction = sess.run([train, loss, acc, output_layer], feed_dict={x_feed: [point], y_feed: [train_y[pid]]})

        # accuracy
        cost_total += cost
        acc_total += output_acc

        # kaggle accuracy
        train_prediction = np.argmax(train_prediction)
        train_prediction_list.append(train_prediction+3)
        if train_prediction > 2 and train_y[pid]-3 > 2:
            quality_acc += 1
        elif train_prediction <= 2 and train_y[pid]-3 <= 2:
            quality_acc += 1

    if iteration % 10 == 0:
        test_acc = 0
        test_acc_quality = 0
        test_pca = []
        test_prediction_list = []
        for tid, test in enumerate(test_data.values):
            accuracy, test_prediction, test_g = sess.run([acc, output_layer, x_pca], feed_dict={x_feed:[test], y_feed: [test_y[tid]]})
            test_pca.append(test_g[0])

            #accuracy
            test_acc += accuracy
            #kaggle accuracy
            test_prediction = np.argmax(test_prediction)
            test_prediction_list.append(test_prediction + 3)
            if test_prediction > 2 and test_y[tid] - 3 > 2:
                test_acc_quality += 1
            elif test_prediction <= 2 and test_y[tid] - 3 <= 2:
                test_acc_quality += 1

        print("iteration:", iteration)
        print("train acc:", cost_total, acc_total / data_num, quality_acc / data_num)  # loss, acc, kaggle acc
        print("test acc:", test_acc / test_num, test_acc_quality / test_num)  # acc, kaggle acc
        np.savetxt("./prediction/train_prediction.txt", train_prediction_list, delimiter=',')
        np.savetxt("./prediction/test_prediction.txt", test_prediction_list, delimiter=',')
        # np.savetxt("./encode/test_list.txt", test_pca, delimiter=',')
        print('save down')

print('down')
#
