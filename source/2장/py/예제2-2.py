# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import pandas as pd	
import numpy as np
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


def build_model():		        #  예제 2-1과 동일한 데이터를 활용해 모델 생성
  model = keras.Sequential([
    layers.Dense(3, activation=tf.nn.relu, input_shape=[train_X.shape[1]]),
    layers.Dense(1),
])
  optimizer = tf.keras.optimizers.RMSprop(0.001)
  model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
  return model

model = build_model()
model.summary()

# 모델 훈련
history = model.fit(train_X, train_y, epochs=1000, validation_split = 0.2, verbose=0)

test_predictions = model.predict(test_X)   		 #  예측
loss, mae, mse = model.evaluate(test_X, test_y, verbose=0)
print("predict value: ", test_predictions, "\n", "Mean Abs Error: {:5.5f}".format(mae))