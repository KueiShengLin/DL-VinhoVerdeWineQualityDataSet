import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import numpy as np
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def preprocess():
    dataset = pd.read_csv("./dataset/winequality-red-dot.csv")

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(dataset)
    a = pd.DataFrame(x_scaled, columns=dataset.columns)
    a['quality'] = dataset['quality']
    train, _ = train_test_split(a, test_size=0.1, random_state=42)
    # train, _ = train_test_split(dataset, test_size=0.1, random_state=42)
    print(train.columns.values)
    train.drop('quality', axis=1, inplace=True)
    name = list(train.columns.values)
    x_bar = []
    for n in name:
        avg = sum(train[n]) / len(train[n])
        x_bar.append(avg)

    return train, x_bar


train_data, x_bar = preprocess()

print('preprocess down')

# layer units define
INPUT = len(train_data.values[0])
PCA = 2
OUTPUT = INPUT

x_feed = tf.placeholder(tf.float32, [None, INPUT])


wen1 = tf.Variable(tf.random_uniform([INPUT, int(INPUT/2)], minval=0, maxval=1), name="wen1")
ben1 = tf.Variable(tf.random_uniform([int(INPUT/2)], minval=0, maxval=1), name="ben1")
formula1 = tf.add(tf.matmul(x_feed, wen1), ben1)
encode1 = tf.nn.sigmoid(formula1)

wen2 = tf.Variable(tf.random_uniform([int(INPUT/2), int(INPUT/4)], minval=0, maxval=1), name="wen2")
ben2 = tf.Variable(tf.random_uniform([int(INPUT/4)], minval=0, maxval=1), name="ben2")
formula2 = tf.add(tf.matmul(encode1, wen2), ben2)
encode2 = tf.nn.sigmoid(formula2)

wen3 = tf.Variable(tf.random_uniform([int(INPUT/4), PCA], minval=0, maxval=1), name="wen3")
ben3 = tf.Variable(tf.random_uniform([PCA], minval=0, maxval=1), name="ben3")
formula3 = tf.add(tf.matmul(encode2, wen3), ben3)
encode3 = tf.nn.sigmoid(formula3)

wde1 = tf.Variable(tf.random_uniform([PCA, int(INPUT/4)], minval=0, maxval=1), name="wde1")
bde1 = tf.Variable(tf.random_uniform([int(INPUT/4)], minval=0, maxval=1), name="bde1")
formula4 = tf.add(tf.matmul(encode3, wde1), bde1)
decode1 = tf.nn.sigmoid(formula4)

wde2 = tf.Variable(tf.random_uniform([int(INPUT/4), int(INPUT/2)], minval=0, maxval=1), name="wde2")
bde2 = tf.Variable(tf.random_uniform([int(INPUT/2)], minval=0, maxval=1), name="bde2")
formula5 = tf.add(tf.matmul(decode1, wde2), bde2)
decode2 = tf.nn.sigmoid(formula5)

wde3 = tf.Variable(tf.random_uniform([int(INPUT/2), INPUT], minval=0, maxval=1), name="wde3")
bde3 = tf.Variable(tf.random_uniform([INPUT], minval=0, maxval=1), name="bde3")
formula6 = tf.add(tf.matmul(decode2, wde3), bde3)
x_hat = tf.nn.sigmoid(formula6)

loss = tf.losses.mean_squared_error(x_hat, x_feed)   # loss function use cross entropy
# loss = tf.losses.mean_squared_error(y_hat, output_layer)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)  # use adadeleta(adagrand的加強版) change learning rate
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.Session(config=tf.ConfigProto(
  allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options))
sess.run(init)

saver = tf.train.Saver({"wen1": wen1, "ben1": ben1, "wen2": wen2, "ben2": ben2, "wen3": wen3, "ben3": ben3})


for iteraion in range(1000001):
    cost_total = 0
    for pid, point in enumerate(train_data.values):
        x = point #- x_bar

        _, cost, prediction = sess.run([train, loss, x_hat], feed_dict={x_feed: [x]})
        cost_total += cost

    if iteraion % 100 == 0:
        print(iteraion)
        print(cost_total / len(train_data.values))
        print(x)
        print(prediction)
        np.savetxt("./decode/w_pca1.txt", sess.run(wen1), delimiter=',')
        np.savetxt("./decode/b_pca1.txt", sess.run(ben1), delimiter=',')
        np.savetxt("./decode/w_pca2.txt", sess.run(wen2), delimiter=',')
        np.savetxt("./decode/b_pca2.txt", sess.run(ben2), delimiter=',')
        np.savetxt("./decode/w_pca3.txt", sess.run(wen3), delimiter=',')
        np.savetxt("./decode/b_pca3.txt", sess.run(ben3), delimiter=',')
        save_path = saver.save(sess, "I:/Jupyter/DNN/DL-VinhoVerdeWineQualityDataSet/tfsave/save.ckpt")
        # print("Model saved in file: %s" % save_path)
        print("save down")
#
