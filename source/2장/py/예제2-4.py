# -*- coding: utf-8 -*-

from tensorflow import keras
from tensorflow.keras import layers 
import pandas as pd
import tensorflow as tf
import random

infert = pd.read_csv("infert.csv", delimiter = ",")           #  데이터 불러오기

X_data = infert[["parity", "induced", "spontaneous"]] 	 #  X, y 정의
y_data = infert["case"]

train_idx = random.sample(list(range(len(infert))), int(len(infert) * 0.7 ))
train_X = X_data.iloc[train_idx]  			 #  train, test 데이터로 분할
train_y = y_data[train_idx]

test_X = X_data.drop(train_idx)
test_y = y_data.drop(train_idx)

model = keras.Sequential([	         # 예제 2-3과 동일한 데이터를 활용해 모델 생성
    layers.Dense(3, activation=tf.nn.relu, input_shape = [train_X.shape[1]]),
    layers.Dense(1, activation=tf.nn.relu),
    layers.Dense(1, activation=tf.nn.sigmoid)
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


model.summary()  	
model.fit(train_X, train_y, epochs = 100)	

predictions = model.predict(test_X) 		 # 예측
test_loss, test_acc = model.evaluate(test_X, test_y)
print('Test accuracy:', test_acc)