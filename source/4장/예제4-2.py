# 필요한 라이브러리 불러오기
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import os
# 그래프 리셋
tf.reset_default_graph()
# 재현성을 위해 시드 지정
tf.set_random_seed(1)
#-------------------------------------------------
# 데이터 불러오기
#-------------------------------------------------
os.chdir("C:\\Users\\admin\\Dropbox\\저서\텐서플로_딥러닝\\RNN_tensorflow")
raw_data = open('input.txt', 'r').read() # input.txt 예제 데이터
# 계산 시간 때문에 일부만 사용
raw_data = raw_data[:200000]
# 전체 문자 수
n_samples = len(raw_data)
# 고유한 문자
unique_chars = list(set(raw_data));
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
#-------------------------------------------
# 매개변수 설정
#-------------------------------------------
# 은닉층 크기: 32
# 미니배치 크기: 100
# 반복 수: 200
# 학습률: 0.1
# Gradient Clipping에 사용할 임곗값: 2
batch_size = 100
seq_len = 50 # 한 번에 입력되는 서열 길이
hidden_size = 32 # 은닉층 노드 수
learning_rate = 0.05
grad_clip = 5 # Gradient Clipping에 사용할 임곗값
nepochs = 50 # 반복 수
# 배치 개수
num_batches = int(len(raw_data)/(batch_size * seq_len))
# 데이터 - (배치 크기*배치 개수)
data = raw_data[:num_batches*batch_size*seq_len]
# 데이터를 정수로 변환
data = np.array([char_to_int[n] for n in data]);
# 입력과 목표 데이터 설정
xdata = data; ydata = np.copy(data)
ydata[:-1] = xdata[1:]; ydata[-1] = xdata[0]
# 배치 개수*배치 크기*서열 길이 = (?, 100, 32)
x_batches = np.split(xdata.reshape(batch_size, -1),num_batches, 1)
x_batches = np.asarray(x_batches)
#x_batches.shape

y_batches = np.split(ydata.reshape(batch_size, -1), num_batches, 1)
y_batches = np.asarray(y_batches)
# 입력 데이터 플레이스 홀더 설정([batch_size, seq_len])
X = tf.placeholder(tf.int32, shape = [None, None])
# 목표 데이터 플레이스 홀더 설정([batch_size*seq_len, n_unique_chars])
Y = tf.placeholder(tf.int32, shape = [None, None])
state_batch_size = tf.placeholder(tf.int32, shape=[]) # Training: 100, Sampling: 1
# LSTM으로 cell을 정의한다.
cell_1 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
# 드롭아웃 사용
cell_1 = tf.nn.rnn_cell.DropoutWrapper(cell_1, output_keep_prob=0.8)
# 은닉층 추가
cell_2 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
# 은닉층 추가
cell_3 = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
# 은닉층이 3개인 LSTM
cell = tf.nn.rnn_cell.MultiRNNCell([cell_1, cell_2, cell_3])
# 각 문자를 원-핫 코딩이 아닌 hidden_size 크기의 숫자 벡터로 변환
# 즉 n_unique_chars -> hidden_size
embedding = tf.Variable(tf.random_normal(shape=[n_unique_chars, hidden_size]),
dtype = tf.float32)
X_one_hot = tf.nn.embedding_lookup(embedding, X)
# X_one_hot = tf.one_hot(X, n_unique_chars)
# 초기 state 값을 0으로 초기화
initial_state = cell.zero_state(state_batch_size, tf.float32)
# 학습을 위한 tf.nn.dynamic_rnn을 선언
# 서열의 길이가 일정하기 때문에 static를 사용해도 되지만 dynamic이 메모리 관점에서
# 장점이 있음
# outputs의 형태는 [batch_size, seq_len, hidden_size]
outputs, final_state = tf.nn.dynamic_rnn(cell, X_one_hot, initial_state = initial_state,
dtype = tf.float32)

# ouputs을 [batch_size * seq_len, hidden_size]] 형태로 바꾸기
output = tf.reshape(outputs, [-1, hidden_size])
# 은닉층의 값으로 출력을 만듦
# 은닉층의 결과를 완전 연결층을 통하여 분류
model = tf.contrib.layers.fully_connected(inputs = output, num_outputs = num_classes,
activation_fn = None)
prediction = tf.nn.softmax(model)
# 손실함수
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model, labels = Y))
# 변수 선언
tvars = tf.trainable_variables()
# 기울기 클리핑
grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
# 옵티마이저
optim = tf.train.AdamOptimizer(learning_rate)
# 기울기 적용 (optim.AdamOptimizer().minimize() 대신 사용됨)
train_step = optim.apply_gradients(zip(grads, tvars))
#------------------------------------------------
# 텐서플로 그래프 생성 및 학습 
#------------------------------------------------
sess=tf.Session();  
sess.run(tf.global_variables_initializer())

losses=[]
for epoch in range(nepochs):   
    bat_loss=[]
    state = sess.run(initial_state, feed_dict={state_batch_size : batch_size})
    for j in range(x_batches.shape[0]):
        state = sess.run(initial_state, feed_dict={state_batch_size : batch_size})
        xx = x_batches[j,:,:]
        y0 = y_batches[j,:,:]
        y0 = np_utils.to_categorical(y0, n_unique_chars)  # [batch_size, seq_len, n_unique_chars]
        
        yy = np.reshape(y0, [-1, n_unique_chars])       # [batch_size * seq_len, n_unique_chars]
        yy=np.array(yy, dtype=np.int32)
        # feed-dict에 사용할 값들과 LSTM 초기 cell state(feed_dict[c])값과
        # hidden layer 출력값(feed_dict[h])을 지정
        feed_dict = {X:xx, Y:yy, state_batch_size:batch_size}
        for i, (c, h) in enumerate(initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        # 한스텝 학습을 진행.
        _, loss, state = sess.run([train_step, cost, final_state], feed_dict=feed_dict)  
        bat_loss.append(loss)
    mean_bat_loss=np.array(bat_loss).mean()        
    
    if ((epoch%1==0)):
            print('epoch={}, loss={:.4f}' .format(epoch, mean_bat_loss))
    losses.append(mean_bat_loss)  
        #print(bat_loss)

# 훈련과정의 loss 그림  
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses)
ax.set_xlabel('epochs')
ax.set_ylabel('Losses')


# 생성할 문장의 길이
generated_text_len = 200  

# 시작 문자를 't'로 지정.
generated_text = 't' 
# RNN의 최초 state값을 0으로 초기화.
state = sess.run(cell.zero_state(1, tf.float32))

for n in range(generated_text_len):
    
    if len(generated_text)> seq_len:
        gen_text_input=generated_text[1:]
    else:   
        gen_text_input=generated_text
        
    gen_text_input=np.array([char_to_int[x] for x in generated_text]).reshape(-1,len(generated_text))
    
    [probs_result, state] = sess.run([prediction,final_state], feed_dict={X: gen_text_input, state_batch_size:1,initial_state:state})   
    p = np.squeeze(probs_result)[-1] 
    # 다음 문자를 예측할때 확률을 이용한 random sampling을 사용
    sample=int(np.searchsorted(np.cumsum(p), np.random.rand(1)*np.sum(p)))  
    pred = int_to_char[sample]
    # 생성된 문자열에 현재 스텝에서 예측한 문자추가. 
    generated_text += pred
   
print(generated_text)