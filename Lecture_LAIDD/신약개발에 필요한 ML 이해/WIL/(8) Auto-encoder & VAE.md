# Auto-encoder & VAE

## Auto-Encoder(AE)의 이해

**Auto-encoder**

- 이미지, 사람, 상품, 분자, 튜플형 등 **어떤 객체(object)를 표현하는 input 신호를 적은 크기의 정보량으로 압축한 후 다시 재구성**하는 모델
    
    입력 자신을 재구성하므로 ‘auto-encoder’
    
- 입력 신호를 작은 크기의 압축된 벡터로 표현하면 이를 이용하여 discrete하게 표현되던 객체를 연속공간의 벡터 값들로 표현할 수 있게 된다
    
    ⇒ **잠재 벡터(latent vector) 또는 임베딩 벡터**
    
- 오토 인코더는 잠재 벡터를 얻는 편리한 방법 중 하나
    - 잠재 벡터를 얻는 방법은 매우 다양하며 대표적인 것이 단어를 임베딩 벡터로 표현한 **word2vec**
    - word2vec을 만들때는 입력 문장에서 가린 단어를 예측하는 훈련을 시킨다 (2차원 이상의 고차원 벡터 표현 가능)

**어떤 객체를 어떤 공간에 임베딩 "벡터"로 표현하면 얻는 장점**

- 벡터로 표현되면 샘플간의 **거리를 쉽게 계산**할 수 있고 유사한 샘플, 거리가 먼 샘플을 쉽게 찾아낼 수 있다. 또한 **클러스터링도** 쉽게 구현할 수 있다.
- **벡터 공간상에서는 방향을 계산**할 수 있다
- **경사하강법 기반의(즉 gradient 미분에 기반한) 연속형 탐색**이 가능해진다. 바로 옆의 벡터값을 갖는 샘플은 성격이 비슷할 것으로 추정할 수 있다

신약개발에 활용 : 분자를 잠재 공간에서 표현하면 **분자 간 interpolation(선형 보간법)이 가능하고, 원하는 특성을 갖는 분자를 찾거나 생성**할 수 있다

### 예제

- MNIST 이미지 재구성하는 데 오토인코더 사용
    
    인코더와 디코더로 **MLP(Dense)** 사용
    
- [참고블로그](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fidiotdeveloper.com%2Fintroduction-to-autoencoders%2F)

### 데이터 준비 및 전처리

- 입력 이미지를 재구성하므로 입력 이미지를 레이블로 사용할 수 있다. 별도의 레이블이 필요없다.
- 스케일링과 flat화
- MLP 모델은 입력으로 차원이 1인, 벡터만 받을 수 있다

```python
import keras
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, add
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras import regularizers
from keras.regularizers import l2
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

(X_train, _), (X_test, _) = mnist.load_data()   # y 대신 _

# flat
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.
X_train_flat = X_train.reshape(-1, 28*28)
X_test_flat = X_test.reshape(-1, 28*28)
```

### Auto-encoder model

1개의 히든 계층만 있는 단순한 모델 사용

**# embedding size, hidden layer 수/차원을 줄여보면 어떻게 될까?**

```python
input_size = 784
**hidden_size** = 64
output_size = 784

x = Input(shape=(input_size,))
h = Dense(hidden_size, activation='relu')(x)
r = Dense(output_size, activation='sigmoid')(h)

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_3 (InputLayer)        [(None, 784)]             0         
                                                                 
 dense_4 (Dense)             (None, 64)                50240     
                                                                 
 dense_5 (Dense)             (None, 784)               50960     
                                                                 
=================================================================
Total params: 101,200
Trainable params: 101,200
Non-trainable params: 0
_________________________________________________________________
```

### Auto-encoder 학습

epochs, batch_size 정한 후 학습

핵심 features를 MLP로 학습

```python
epochs = 5
batch_size = 32

history = autoencoder.**fit**(X_train_flat, X_train_flat, batch_size=batch_size, 
        epochs=epochs, verbose=1, validation_data=(X_test_flat, X_test_flat))

Epoch 1/5
1875/1875 [==============================] - 5s 2ms/step - loss: 0.0241 - val_loss: 0.0093
Epoch 2/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0069 - val_loss: 0.0053
Epoch 3/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0050 - val_loss: 0.0046
Epoch 4/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0046 - val_loss: 0.0043
Epoch 5/5
1875/1875 [==============================] - 4s 2ms/step - loss: 0.0044 - val_loss: 0.0041
```

### 벡터값 시각화 / 원본-재구성 이미지 비교

```python
img_encoder = Model(x, h)
encoded_imgs = img_encoder.predict(X_test_flat)

**# 잠재 벡터값 보기**
print(encoded_imgs[:2])

# 그림을 그릴 개수 
n = 10

plt.figure(figsize=(12, 6))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(16, 4))  **# embedding size, hidden layer 수/차원을 줄여보면 어떻게 될까?**
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()   # **64bit만 있어도 재구성 가능하다!**

**# 이미지 비교**
decoded_imgs = autoencoder.predict(X_test_flat)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 원본 이미지
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    
    # **재구성된** 이미지
    ax = plt.subplot(2, n, i+n+1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()
```

## VAE 기반 분자 생성모델

- **Variational** Autoencoder은 오코인코더를 **개선하여 평균과 표준편차 두개의 성분을 나타내는 2개의 임베딩 벡터를 생성**하게 한다.
    
    → 더 다양한, 변화된 출력을 생성할 수 있다 (By **Probabilistic Encoder** - Mean, Std.dev)
    
- 새로운 분자의 구조를 생성하는 모델에 적용 가능
- 분자 표현으로 **SMILES**를 사용하며 새로운 SMILES를 얻는다.
    - **sampled latent vector** - 압축된 저차원 표현형
    - MolrculeNet이 제공하는 SMILES 데이터셋 MUV 사용 (약 90000개 제공)
    - Maximum Unbiased Validation(MUV) - 17개의 태스크 포함
- [VAE 소개](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftowardsdatascience.com%2Fan-introduction-to-variational-auto-encoders-vaes-803ddfb623df)

![https://github.com/StillWork/image/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-23%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.06.41.png?raw=1](https://github.com/StillWork/image/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-23%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%204.06.41.png?raw=1)

---

### 데이터 준비

SMILES 문자열(text) : token 단위

```python
!pip install DeepChem

import deepchem as dc
import tensorflow as tf
import tensorflow.keras.layers as layers

import pandas as pd
import numpy as np
%config InlineBackend.figure_format = 'retina'

**# 학습 데이터 (SMILES 문자열 -> token 단위)**
tasks, datasets, transformers = dc.molnet.load_muv()
train_dataset, valid_dataset, test_dataset = datasets
train_smiles = train_dataset.ids   # 약 74000개

**# SMILES 문자열의 규칙을 파악**: 문자(토큰)의 목록, 문자열의 최대길이 등

tokens = set()
for s in train_smiles:
  tokens = tokens.union(set(s))
tokens = sorted(list(tokens))
max_length = max(len(s) for s in train_smiles)
```

### VAE 모델 & 학습

- AspuruGuzikAutoEncoder 사용: 인코더는 합성곱신경망을, 디코더는 순환신경망 사용
- 학습속도를 조절하기 위해서 ExponentialDecay 사용
    - 0.001에서 시작하고 epoch마다 0.95배씩 감소시킴.

```python
from deepchem.models.**seqtoseq** import AspuruGuzikAutoEncoder
from deepchem.models.optimizers import ExponentialDecay
batch_size = 100
batches_per_epoch = len(train_smiles)/batch_size
learning_rate = **ExponentialDecay**(0.001, 0.95, batches_per_epoch)
model = **AspuruGuzikAutoEncoder**(tokens, max_length, model_dir='vae', 
                batch_size=batch_size, learning_rate=learning_rate)

**# 시퀀스 생성 함수 정의**
def generate_sequences(epochs):
  for i in range(epochs):
    for s in train_smiles:   # train_smiles의 array에서 같은 입출력 s 각각 출력
      yield (s, s)

**# 학습**
# **AspuruGuzikAutoEncoder**이 제공하는 자체 학습 함수 (epoch수 지정)
model.**fit_sequences**(generate_sequences(50)) # 50 epoch 수
```

### 분자 생성

- 학습된 모델을 이용하여 새로운 분자 생성 : **random**.normal
- 모델에 들어가는 **벡터의 크기**를 지정 (예: 196)
- 벡터를 2000개 생성
- 생성된 분자들중 유효한 SMILES를 걸러내기 위해서 RDKit의 MolFromSmiles 사용

```python
from rdkit import Chem

predictions = model.predict_from_embeddings(np.**random.normal**(size=(2000,196)))   # input을 random.normal로 생성
molecules = []
for p in predictions:
  smiles = ''.join(p)
  if Chem.**MolFromSmiles**(smiles) is not None:
    molecules.append(smiles)
print()

print('Generated molecules:')
for m in molecules:
  print(m)
```

### 유효한 분자 필터링

- 생성된 SMILES 들에 대해서 유효하지 않거나 약물로서 가치가 없는 분자를 걸러내야 한다!
- 제거하고 싶은 분자가 있는지 찾는다
- **MolFromSmiles**()을 사용해 SMILES 문자열들을 분자 객체로 변환
- **분자의 크기**를 확인한다 (10보다 작으면 상호작용에 필요한 에너지가 불충분하고, 50 이상이면 분자의 용해도가 너무 낮아 문제가 된다)
    
    수소를 제외한 분자의 크기를 **GetNumAtoms**()로 얻는다
    
- **약물과 얼마나 유사한지를 판단**하기 위해서 **QED**(Quantitave Estimate of Drugness)를 많이 사용한다
    - QED: 계산된 속성 집합과 판매된 약물의 동일한 특성 분포를 정량화 한 것 (Richard Bickerton 이 제안)
    - 1에 가까울수록 기존의 약물과 유사하다고 본다
    - QED > 0.5 인 분자만 고른 후 결과를 시각화

→ Mean, Std.dv를 먼저 주고(원하는 값 고정), 그 주위의 샘플 생성할 수도 있음!

```python
from rdkit import Chem
molecules_new = [Chem.MolFromSmiles(x) for x in molecules]
print(sorted(x.GetNumAtoms() for x in molecules_new))

# 분자의 크기
good_mol_list = [x for x in molecules_new if x.GetNumAtoms() > 10 and x.GetNumAtoms() < 50]
# print(len(good_mol_list))

# QED
from rdkit.Chem import QED
qed_list = [QED.qed(x) for x in good_mol_list]

final_mol_list = [(a,b) for a,b in zip(good_mol_list, qed_list) if b > 0.5] # 
final_mol_list

img=Draw.MolsToGridImage([x[0] for x  in final_mol_list],
                         molsPerRow=4,subImgSize=(200,200),
                         legends=[f"{x[1]:.2f}" for x in final_mol_list])
# img : 0.75, 0.

**# Mean, Std.dv를 먼저 주고(원하는 값 고정), 그 주위의 샘플 생성할 수 있도록**
predictions_2 = model_2.predict_from_embeddings(np.random.normal(size=(10,196)))
molecules_2 = []
for p in predictions_2:
  smiles = ''.join(p)
  # if Chem.MolFromSmiles(smiles) is not None:
  molecules_2.append(smiles)
```