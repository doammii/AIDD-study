# DL 이해 및 구현

## 딥러닝 이해

### MLP(Multi Layer Perceptron)

- 퍼셉트론 : 인지. (input - weight - net - activation function)
- FC network - dense network / 여러 개 **hidden layer**을 추가함으로써 “**비선형”** 데이터셋 구분 가능
- 대부분 활성화 함수로 ReLU 사용

### CNN

- 패턴을 찾기 위해 **filter(Conv Layer)** 이용
- Feature Extractor(Conv-Pooling set 여러 개) → Classifier(FC layer로 전체 특징 한 줄로 나열, 최종 판단)
    
    feature map 크기를 pooling 작업을 통해 축소 - 중요 feature만 남김. overfitting 해소에도 기여.
    
- object segmentation, image enhancement 등에 사용됨. → Style transfer(cycle GAN)

▼ **DTA(Drug Target Affinity)** : SMILES & Sequence data를 **CNN 통해 representation** → **concatenation** → 둘의 interaction 확인

![Untitled](https://github.com/doammii/AIDD-study/assets/100724454/1805737e-165d-4fef-8606-0441ed60c9a5)


### RNN

- **NN** : feed-forward neural network (단방향)
- 과거의 입력에 대한 상태(h) 정보를 **순환적으로** 재사용 → 순서
- Encoder(source sequence 압축) & Decoder(target sequence 생성)

### NLP

- word **vector**(단어 임베딩) - 단어들을 2차원상의 공간, 특정 위치에 벡터(방향성)로 표현 가능
- **Transformer** : NLP에서 RNN 기반의 Seq2seq 모델 개선
    - 문장의 **길이**가 길어지면 멀리 떨어진 단어에 대한 상호 정보가 줄어들고, 순차적으로 연산하면 연산의 병렬화가 불가능해져 연산 속도 저하됨.
    - **입력 token의 self-attention**을 사용한 모델
    
    ![Untitled 1](https://github.com/doammii/AIDD-study/assets/100724454/e1b46506-3655-457e-8919-586fd9f29952)

    
- **Self-attention** : sequence 요소들 중 task 수행에 **중요한** 요소에 집중하고 그렇지 않은 요소는 무시해 task 수행 성능을 올리는 개념
    
    입력 자신 전체에 대해 수행하는 attention → CNN, RNN 단점 개선
    
    중요 요소 판단 by multi-head attention
    

---

## 딥러닝 구현

**신경망 특징**

- 60년대부터 가능성을 연구하였으나 오랫동안 성과를 내지 못했다
    - 학습이 어렵고, 계산량이 많고, 과대적합이 많고, 블랙박스 동작
    - 지금은 이러한 문제들이, 데이터의 증가, 알고리즘의 발달(공개 SW), 하드웨어 계산능력의 증대로 해결되었다
- small 데이터에 대해서는 전통적인 **머신러닝 모델 (RF, SVM 등)**을 사용해도 과대적합도 적게 발생하고 잘 동작했다
- 2012년 Imagenet 경진대회에서 **CNN** 기반의 신경망이 매우 우수한 성적을 내면서 딥러닝 모델이 확산되었다
- 자연어 처리분야에서는 2014년 word2vec(단어 임베딩)의 도입으로 성능이 크게 향상되었다
    - 비슷한 의미를 갖는 단어는 비슷한 표현형을 갖게 함
    - 자연어 처리에서는 초기에는 RNN을 사용했으나 현재는 **Transformer**를 사용한다

**Feature learning**

- Feature Engineering은 **사람이** 전문성을 가지고 **중요한 특성**을 만드는 것
- Feature Learning은 **신경망이 학습을 통해서 스스로** 머신러닝에 필요한 특성을 만들어내는 것

신약개발에 신경망 도입

- 2012년 Merck Molecular Activity Challenge(bioactivity 예측)에서 신경망 모델 성능 > RF 모델 성능

![Untitled 2](https://github.com/doammii/AIDD-study/assets/100724454/4a00726d-f8af-4ff0-98b5-e05bf11a0839)

```python
!pip install deepchem

import deepchem as dc
import numpy as np
import matplotlib.pyplot as plt
import keras
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras import layers
from keras import models
```

### MLP - Fingerprint를 사용한 독성 예측

- MLP 모델을 사용 (1개의 히든 계층 사용)
    
    MLP를 구현하기 위해서 **deepchem이 제공하는 `MultitaskClassifier()`**를 사용한다
    
- 12개(output)의 이진 분류 태스크를 수행 ⇒ y의 shape을 통해 12개의 multi-task가 있음을 알 수 있다.
- **w (가중치)의 의미**
    - 이 값이 **0이면 결측치**를 나타낸다. 해당 샘플에 대해서는 손실함수나 성능평가 시에 무시하도록 한다.
    - 이 값은 대부분의 경우 1 또는 1 근처의 값을 갖는다.
    - 이 값은 각 태스크별로 레이블 분포의 불균형을 보완하기 위해서 사용된다.
    - 12개 태스크가 균등하게 성능에 기여하도록 조정하는데 사용된다.
- **roc_auc(분류)와 accuracy**를 측정

```python
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='ECFP')
train_dataset, valid_dataset, test_dataset = datasets

# tasks : ['NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD','NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53']

# train_dataset.X.shape : (6264, 1024)
# train_dataset.y.shape : (6264, 12)
# train_dataset.w

**# 모델 구현 및 성능 평가**
model = dc.models.**MultitaskClassifier**(n_tasks=12, n_features=1024, layer_sizes=[1000]) 
model.fit(train_dataset, nb_epoch=20)

metric1 = dc.metrics.Metric(dc.metrics.roc_auc_score)
metric2 = dc.metrics.Metric(dc.metrics.accuracy_score)

print('training set score:', model.evaluate(train_dataset, [metric1, metric2], transformers))
print('test set score:', model.evaluate(test_dataset, [metric1, metric2], transformers))

```

### CNN

- 데이터 전처리

```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show

train_images.shape
```

- **input data reshape ★**
    - CNN의 입력은 (sample, shape(28x28), feature)과 같은 모양으로 만들어주어야 **2차원 데이터인지** 알 수 있다
        - 맨 앞은 샘플을 구분하고, 맨 뒤는 feature를 구분하는데 사용하며 중간이 모양을 보고 2차원인지 알 수 있다
        - **(샘플, 행, 열, 특성)**

```python
X_train_cnn = train_images.reshape(-1,28,28,1)/255.   # 가로, 세로 인자 줘야 함. 255로 나눈 이유(0~1 사이 아날로그값으로 변환)
X_test_cnn = test_images.reshape(-1,28,28,1)/255.

y_train_cat = to_categorical(train_labels)
y_test_cat = to_categorical(test_labels)
```

- **구현**

```python
from keras import layers
from keras import models

model_cnn = models.**Sequential**()
model_cnn.**add**(layers.Conv2D(32, (3,3), activation='relu', 
                    padding="same", input_shape = (28, 28,1)))
model_cnn.add(layers.MaxPooling2D((2,2)))
model_cnn.add(layers.Conv2D(32, (3,3), 
                    padding="same", activation='relu'))
model_cnn.add(layers.MaxPooling2D((2,2)))
model_cnn.add(layers.Conv2D(64, (3,3), 
                    padding="same", activation='relu'))
model_cnn.add(layers.MaxPooling2D((2,2)))
model_cnn.add(layers.Conv2D(64, (3,3), 
                    padding="same", activation='relu'))

model_cnn.add(layers.**Flatten**())
model_cnn.add(layers.Dense(128, activation='relu'))
model_cnn.add(layers.Dense(10, activation='softmax'))

model_cnn.summary()
```

- **성능 평가**

```python
callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)]

model_cnn.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])   # 환경 설정

# **최적화 알고리즘으로 'adam'과 'rmsprop'**이 널리 사용된다.
h = model_cnn.fit(X_train_cnn,
                      y_train_cat,
                      batch_size=20,
                      epochs=3,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)
# batch_size로 30이나 32정도 사용되기도.

test_loss, test_acc = model_cnn.evaluate(X_test_cnn, y_test_cat)
print('test_acc = ',test_acc)
# test_acc =  0.9858999848365784
```