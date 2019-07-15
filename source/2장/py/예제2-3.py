# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
import random

infert = pd.read_csv("infert.csv", delimiter = ",")                #  데이터 불러오기

X_data = infert[["parity", "induced", "spontaneous"]] 		 #  X, y 정의
y_data = infert["case"]

train_idx = random.sample(list(range(len(infert))), int(len(infert) * 0.7 ))
train_X = X_data.iloc[train_idx]  			 #  train, test 데이터로 분할
train_y = y_data[train_idx]

test_X = X_data.drop(train_idx)
test_y = y_data.drop(train_idx)

X = tf.placeholder(shape = (None, 3), dtype = tf.float32)
y = tf.placeholder(shape = (None), dtype = tf.float32)

W1 = tf.Variable(initial_value= tf.random_normal([3, 1]), dtype = tf.float32)
b1 = tf.Variable(initial_value= tf.random_normal([1]), dtype = tf.float32)
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(initial_value= tf.random_normal([1, 1]), dtype = tf.float32)
b2 = tf.Variable(initial_value= tf.random_normal([1]), dtype = tf.float32)
hypo = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, 		      logits = hypo))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.0001)
train = optimizer.minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(100):   			# 신경망 훈련
  _, cost_val = sess.run([train, cost], feed_dict = {X: train_X, y: train_y})
  if i % 20 ==0:
    print("step: ", i, "\t", "cost: ", cost_val)
print("train finished!!")

predict = tf.cast(hypo > 0.5, dtype = tf.float32)   	# 예측
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype = tf.float32 ))
print("accuracy: ", sess.run(accuracy, feed_dict = {X: test_X, y: test_y}))