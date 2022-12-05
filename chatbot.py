import tensorflow as tf

import numpy as np
import os
import time
import random

from konlpy.tag import Komoran
from konlpy.tag import Mecab

text = open("data/clean/ts.txt", 'rb').read().decode(encoding='utf-8')
# print(text[:200])

vocab = sorted(set(text))
# print(vocab[:10], len(vocab))
char2idx = {u:i for i, u in enumerate(vocab)}
# index -> character로 변환하는 사전 
idx2char = np.array(vocab)

# 문자로 된 어휘 사전의 크기
vocab_size = len(vocab)

# 임베딩 차원
embedding_dim = 256

# RNN 유닛 개수
rnn_units = 1024

checkpoint_path = './models/my_checkpt.ckpt'

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[1, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
])

model.load_weights(checkpoint_path)
model.build(tf.TensorShape([1,None]))

def generate_text(model,start_string, temperature = 1.0,num_generate = 1):
    # 평가 단계(학습된 모델을 사용하여 텍스트 생성)

    # 생성 할 문자의 수
    # num_generate = 1000
    
    # 시작 문자열을 숫자로 변환(벡터화)
    # if len(start_string) == 0:
    #     start_string = "침묵"
    input_eval = []
    for s in start_string:
        if s in char2idx.keys():
            input_eval.append(char2idx[s])
        else :
            input_eval.append(char2idx["안"])
    # [char2idx[s] for s in start_string]

    # input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # 결과를 저장 할 빈 문자열
    text_generated = []

    # 여기서 배치 크기 == 1
    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        # 배치 차원 제거
        predictions = tf.squeeze(predictions, 0)

        # 범주형 분포를 사용하여 모델에서 리턴한 단어 예측
        # 온도가 낮으면 더 예측 가능한 텍스트가 된다.
        # 온도가 높으면 더 의외의 텍스트가 된다.
        # 최적의 세팅을 찾기 위한 실험
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # 예측된 단어를 다음 입력으로 모델에 전달
        # 이전 은닉 상태와 함께
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
        ending_words = ["\n","?",".","!"]
        if idx2char[predicted_id] in ending_words:
            return (start_string + ''.join(text_generated))

    return (start_string + ''.join(text_generated))

mecab = Mecab()

def start_chat():
    print("---채팅 시작---")
    while True:
        ip = input()
        if(ip == "바이") : break

        if(len(ip) == 0) : ip = "..."
        
        tokens = mecab.morphs(ip)
        token = random.choice(tokens)
        answer = generate_text(model, start_string=token,temperature=1,num_generate=50)
        
        if len(answer) == 0 : print("답없음")
        
        print("나 : ", ip)
        print("또다른 나 : ", answer)

start_chat()
