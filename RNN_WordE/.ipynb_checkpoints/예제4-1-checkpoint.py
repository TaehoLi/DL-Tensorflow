# 필요한 라이브러리 불러오기
import tensorflow as tf
import numpy as np
tf.set_random_seed(777) # reproducibility
from keras.utils import np_utils
import matplotlib.pyplot as plt
# 그래프 리셋
tf.reset_default_graph()
# 재현성을 위해 시드 지정
tf.set_random_seed(1)

#--------------------------------------------------
# 데이터 불러오기
#--------------------------------------------------
raw_data = 'First Citizen'
n_samples = len(raw_data) # 전체 문자 수
unique_chars = list(set(raw_data)); # 고유한 문자
# 문자를 정수로 변환하는 딕셔너리
char_to_int = { ch:i for i,ch in enumerate(unique_chars) }
# 정수를 문자로 변환하는 딕셔너리
int_to_char = { i:ch for i,ch in enumerate(unique_chars) }
# 고유한 문자 수
n_unique_chars = len(unique_chars)
# 입력층의 노드 수(입력 크기) = 원-핫 벡터 크기
input_dim = n_unique_chars
# 출력층의 노드 수 = 고유한 문자 수
num_classes = n_unique_chars
# 서열의 길이
seq_len = n_samples-1
#-------------------------------------------
# 매개변수 설정
#-------------------------------------------
# 미니배치 크기: 1
# 은닉층 크기: 16
# 학습률: 0.1
# 반복 수: 20
batch_size = 1
hidden_size = 16
learning_rate = 0.1
nepochs = 20
# 입력(input) 데이터
x = raw_data[:-1]
# 목표(target) 데이터
y = raw_data[1:]
# 입력 데이터를 정수로

x_int = [char_to_int[n] for n in x]
# 목표 데이터를 정수로
y_int = [char_to_int[n] for n in y]
# 1차원에서 2차원으로
y_int = np.array(y_int).reshape(1,12)
# 입력 데이터를 원-핫 벡터로 변환
x_one_hot = np_utils.to_categorical(x_int, 
n_unique_chars).reshape(batch_size,seq_len, input_dim)
# 입력 데이터 플레이스 홀더(배치 크기*서열 길이*입력 크기)
X = tf.placeholder(tf.float32, [None, seq_len, input_dim])
# 목표 데이터 플레이스 홀더(배치 크기*서열 길이)
Y = tf.placeholder(tf.int32, [None, seq_len]) # Y label
# RNN으로 셀 정의
cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_size)
# 초깃값 설정
initial_state = cell.zero_state(batch_size, tf.float32)
# 셀 연결
outputs,_states = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state,
dtype = tf. float32)
# 은닉층의 결과를 완전 연결층을 통하여 분류
outputs = tf.contrib.layers.fully_connected(inputs = outputs, num_outputs = num_classes,
activation_fn = None)
# 최종 결과 재표현
outputs = tf.reshape(outputs, [batch_size, seq_len, num_classes])
# 서열 각 위치의 loss를 합하여 전체 loss 계산
# 각각의 loss에 대한 가중치를 같게
weight = tf.ones([batch_size, seq_len])
# 서열 각 위치의 cross entropy
sequence_loss = tf.contrib.seq2seq.sequence_loss(logits = outputs,
targets = Y, weights = weight)
# 전체 loss
loss = tf.reduce_mean(sequence_loss)

# optimizer 정의
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# 각 문자의 softmax 결과에서 라벨 확인
prediction = tf.argmax(outputs, axis=2)
#------------------------------------------------
# 텐서플로 그래프 생성 및 학습
#------------------------------------------------
losses = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(nepochs):
        l, _ = sess.run([loss, optimizer], feed_dict = {X: x_one_hot, Y: y_int})
        result = sess.run(prediction, feed_dict = {X: x_one_hot})
        print('epoch = {}, loss = {}' .format(epoch, l))
        losses.append(l)
# 결과를 확인
result_str = [int_to_char[k] for k in result[0]]
print("predicted : {}" .format(''.join(result_str)))

# 훈련과정의 loss 그림
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses)
ax.set_xlabel('epochs')
ax.set_ylabel('Losses')
