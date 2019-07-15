# -*- coding: utf-8 -*-

import pandas as pd	
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder, normalize

credit_card = pd.read_csv("CreditCard.csv", delimiter = ",")    # 데이터 불러오기

credit_cat = credit_card[["card", "owner", "selfemp"]]        
label_enc = LabelEncoder()
card = label_enc.fit_transform(credit_cat["card"])		# 범주형 변수 더미화
card.shape = (len(card), 1)
owner = label_enc.fit_transform(credit_cat["owner"])
owner.shape = (len(owner), 1)
selfemp = label_enc.fit_transform(credit_cat["selfemp"])
selfemp.shape = (len(selfemp), 1)


### 수치형 변수 정규화 ###
credit_num  = credit_card.drop(["card", "owner", "selfemp", "share"], axis = 1) 
credit_num_norm = normalize(credit_num)                    #   수치형 변수 정규화 

credit_X = np.concatenate([card, owner, selfemp, credit_num_norm], axis = 1)
credit_y = np.array(credit_card['share'])
credit_y.shape = (len(credit_y), 1)

train_idx = random.sample(list(range(len(credit_card))), int(len(credit_card) * 0.7))
train_X = credit_X[train_idx, :]   			#  train, test 데이터로 분할
train_y = credit_y[train_idx]

test_X = np.delete(credit_X, train_idx, axis = 0)
test_y = np.delete(credit_y, train_idx)
test_y.shape = (len(test_y), 1)

X = tf.placeholder(dtype = tf.float32, shape = (None, 11)) 
y = tf.placeholder(dtype = tf.float32, shape = None)

W1 = tf.Variable(initial_value = tf.random_normal([11,4]), dtype = tf.float32)
b1 = tf.Variable(initial_value = tf.random_normal([4]), dtype = tf.float32)
L1 = tf.add(tf.matmul(X, W1), b1)

W2 = tf.Variable(initial_value = tf.random_normal([4, 1]), dtype = tf.float32)
b2 = tf.Variable(initial_value = tf.random_normal([1]), dtype = tf.float32)
hypo = tf.add(tf.matmul(L1, W2), b2)

cost = tf.reduce_mean(tf.square(hypo- y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):  				     #   신경망 학습
  _, cost_val = sess.run([train, cost], feed_dict = {X: train_X, y : train_y})
  if i % 100 ==0:
    print("cost: ", cost_val)
print("train_finished!")				     

pred_val, pred_cost = sess.run([hypo, cost], feed_dict = ({X: test_X, y : test_y}))
print("predict value: ", pred_val, "\n", "predict cost: ", pred_cost)   # 예측 결과 확인