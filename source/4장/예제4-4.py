# 필요한 라이브러리 부름
import tensorflow as tf
import numpy as np
from keras.utils import np_utils 
import matplotlib.pyplot as plt

# 그래프 리셋
tf.reset_default_graph() 
# 재현성을 위해 시드 지정
tf.set_random_seed(1) 

# 매개변수 지정
# 학습률: 0.1
# 반복 수: 10000
# 임베딩 차원: 2
# 윈도우 크기: 1
learning_rate=0.1
nepochs=10000
embedding_dim=2
window_size = 1

# 워드 임베딩에 적용할 문장
#text = 'Arthur is a famous king He is a brave man The king is royal Elizabeth is the royal  queen She is a beautiful woman'
text ="King is a brave man Queen is a beautiful woman"

# 소문자로 변환
text = text.lower()

# 간단한 불용어와 문자 그리고 숫자 제거
word_seq = []
for word in text.split():
    if ((word != '.') & (word not in '0123456789')& (word not in ['a','is', 'the'] )):
        word_seq.append(word)
        
# 고유한 단어들로 만든 집합
unique_words = set(word_seq) 
n_unique_words = len(unique_words) 

# 단어와 정수 매핑
word_to_int = {w: i for i, w in enumerate(unique_words)}
int_to_word = {i: w for i, w in enumerate(unique_words)}

# 훈련에 사용될 데이터 [input, target] 만듬              
data = []
for i in range(1, len(word_seq) - 1):
    # [input, target] = [neighbors, target]
    target = word_seq[i]
    neighbor=[]
    for j in range(window_size):
        neighbor.append(word_seq[i - j-1])
        neighbor.append(word_seq[i +  j+ 1])

    for w in neighbor:
        data.append([w, target])
        
# 원-핫 벡터로 변환
x_train = [] 
y_train = [] 

for w in data:
    x_train.append(np_utils.to_categorical(word_to_int[w[0]] , n_unique_words))
    y_train.append(np_utils.to_categorical(word_to_int[w[1]] , n_unique_words))
    
# 넘파이 arrary로 변환
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

# 훈련에 사용될 placeholder
X = tf.placeholder(tf.float32, shape=(None, n_unique_words))
Y = tf.placeholder(tf.float32, shape=(None, n_unique_words))

# 입력층과 은닉층의 가중치
W1 = tf.Variable(tf.random_normal([n_unique_words, embedding_dim]))
b1 = tf.Variable(tf.random_normal([embedding_dim]))
# 은닉층 값
hidden_representation = tf.add(tf.matmul(X,W1), b1)
# 은닉층과 출력층의 가중치
W2 = tf.Variable(tf.random_normal([embedding_dim, n_unique_words]))
b2 = tf.Variable(tf.random_normal([n_unique_words]))

# 출력값
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))


# 손실함수 
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(prediction), reduction_indices=[1]))

# optimizer 정의
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

#------------------------------------------------
# 텐서플로 그래프 생성 및 학습 
#------------------------------------------------
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) 
losses=[]
for epoch in range(nepochs):
    sess.run(train_step, feed_dict={X: x_train, Y: y_train})
    loss=sess.run(cross_entropy_loss, feed_dict={X: x_train, Y: y_train})
    if epoch%100==0:
        print('epoch={}, loss = {}' .format(epoch, loss))
    losses.append(loss)

# 훈련과정의 loss 그림      
fig, ax = plt.subplots(figsize=(7,7))
losses = np.array(losses)
plt.plot(losses)
ax.set_xlabel('epochs')
ax.set_ylabel('Losses')


# 임베딩 결과
vectors = sess.run(W1+b1) 
print(vectors)

print(unique_words)

fig, ax = plt.subplots()

#임베딩 그림
for  i, label in enumerate(word_to_int):
    x, y = vectors[i]
    plt.xlim(-10,10)
    plt.ylim(-10,10)
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2),
                 textcoords='offset points', ha='right', va='bottom')
    
plt.show()










