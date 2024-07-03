# GAN

## GAN 이해

- Generative Adversarial Network (적대적 생성 신경망)
- Random Noise + **Generator**(생성자)와 **Discriminator**(Real/Fake 분별)가 **경쟁적으로 학습**(많은 데이터로 학습)하면서 서로 성능을 개선하는 것
    
    **GAN의 최종 목적 : 새로운 데이터를 생성**
    

![https://github.com/StillWork/image/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-20%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2011.42.33.png?raw=1](https://github.com/StillWork/image/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-20%20%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE%2011.42.33.png?raw=1)

**응용**

- 예술품 창작(CAN)
- 일부 지워진 이미지 복원(Image completion)
- 스타일 트랜스퍼(cycleGAN)
- 고품질의 이미지 생성(BigGAN)
- 흑백 사진에 컬러채색
- **잠재공간**(latent space) 연산으로 새로운 이미지 생성

### 예제

- GAN을 이용하여 MNIST 이미지 생성 예시
- **Wasserstein GAN** 사용
- 참고 [GAN 개념 소개](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftowardsdatascience.com%2Ffundamentals-of-generative-adversarial-networks-b7ca8c34f0bc)

### 데이터 준비

`(60000, 28, 28)`

```python
!pip install deepchem
import numpy as np
import deepchem as dc
import tensorflow as tf
from deepchem.models.optimizers import ExponentialDecay
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Reshape
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# MNIST 이미지 데이터
mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
mnist[0][0].shape
```

### 데이터 변환

- 스케일링
- **4차원 어레이로 재구성**
- ndarray에서 dataset 구성

```python
images = mnist[0][0].reshape((-1, 28, 28, 1))/255   # 4차원 reshape
dataset = dc.data.NumpyDataset(images)

# <NumpyDataset X.shape: (60000, 28, 28, 1), y.shape: (60000, 1), w.shape: (60000, 1), task_names: [0]>
```

- 샘플 이미지 보기

```python
def plot_digits(im):
  plt.figure(figsize=(5, 5))
  grid = gridspec.GridSpec(4, 4, wspace=0.05, hspace=0.05)
  for i, g in enumerate(grid):
    ax = plt.subplot(g)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(im[i,:,:,0], cmap='gray')

plot_digits(images[:16])
```

## GAN 구현

- GAN은 크게 생성자와 판별자로 구성된다.
- **생성자** : 랜덤 신호(Random noise - 가장 정보량 많아서 모든 작업 가능)를 입력으로 받고, 훈련 데이터에 있는 MNIST를 닮은 출력을 생성
    
    랜덤 신호를 통해 다양한 task 생성 가능
    
- 판별자 : 생성자가 만든 (가짜) 출력 이미지와 (실제) 훈련 이미지를 입력으로 받아 가짜 이미지와 실제 이미지를 구분하는 작업을 수행한다
- Wasserstein GAN (WGAN)을 상속받아 사용
- 생성자는 Dense망을 사용하여 입력 노이지를 7x7 크기의 이미지로 바꾸며 8개의 채널(8 sets 조합 위해)을 사용한다
    - 두 번의 컨볼류션 계층을 수행하며 각각 업샘플링을 하여 14x14 그리고 28x28 크기의 이미지로 변형한다
- 판별자는 위의 작업의 역순의 작업을 수행한다
    - 두 번의 컨볼류션 계층을 수행하며 각각 1/2 다운 샘플링을 하여 14x14 그리고 7x7 크기의 **피처맵을** 만든다
    - 이후 전결합망(dense)을 통과하여 단일값을 갖는 출력을 생성한다
- **WGAN**에서는 softmax 대신 softplus 활성함수를 사용하여 (0~1) 사이의 값이 아닌, **임의의 크기를 갖는 값(얼마나 틀렸는지)**을 출력한다
    - (분류) 확률을 얻는 것이 아니라 **거리**(distance)를 예측한다
    - WGAN에서는 실제 이미지일 확률을 예측하는 것이 아니라, 훈련 이미지의 분포와 생성 이미지의 분포 사이의 "거리"를 나타낸다
    - 이 거리 값을 생성자를 훈련시키는데 손실함수로 사용하여 학습 속도를 개선했다

### GAN 클래스 정의

```python
class DigitGAN(dc.models.WGAN):
  def get_noise_input_shape(self):
    return (10,)

  def get_data_input_shapes(self):
    return [(28, 28, 1)]

  def create_generator(self):
    return tf.keras.Sequential([
        Dense(7*7*8, activation=tf.nn.relu),
        Reshape((7, 7, 8)),   # depth : 8
        Conv2DTranspose(filters=16, kernel_size=5, strides=2, activation=tf.nn.relu, padding='same'),
        Conv2DTranspose(filters=1, kernel_size=5, strides=2, activation=tf.sigmoid, padding='same')
    ])

  # softplus 사용
  def create_discriminator(self):
    return tf.keras.Sequential([
        Conv2D(filters=32, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
        Conv2D(filters=64, kernel_size=5, strides=2, activation=tf.nn.leaky_relu, padding='same'),
        Dense(1, activation=tf.math.**softplus**)
    ])

gan = DigitGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), model_dir = 'gan')
```

### 모델 학습

- 이미지 생성자는 dataset에서 데이터를 가지고 오며 epoch는 100으로 설정
- 일반적인 GAN을 구현할 때는 **생성자와 판별자를 훈련시키는데 서로 너무 큰 실력 차이가 나지 않도록** 세심하게 주의를 해야 한다
    
    생성자보다 판별자 역할이 쉬움.
    
    → 실력(성능)의 차이가 너무 크면 균형있게 두개의 모델을 학습시키기가 어렵기 때문이다.
    
- **WGAN**에서는 이러한 문제를 피할 수 있는데, 판별자의 성능이 좋아지면 이에 **비례한 손실함수(거리)**를 알려주고 이를 통해 생성**자가 학습을 더 잘** 할 수 있게 한다.
- `generator_steps=0.2`로 설정하였다. 이는 판별자가 5회 학습하면 생성자가 1회 학습하는 비율인데, 이렇게 하여 학습 속도를 높이고 더 나은 성능을 얻도록 한다.

```python
# softmax와 softplus
fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111)

x = np.arange(-6, 6, 0.01)

def sigmoid(x): # 시그모이드(Sigmoid, S-커브), Logistic Regression
    return 1 / (1 + np.exp(-x)) 
ax.plot(x, sigmoid(x), color='r', linestyle='-', label="Sigmoid")

def softplus_func(x): # SoftPlus 함수
    return np.log( 1 + np.exp(x) )
ax.plot(x, softplus_func(x), color='b', linestyle='-', label="SoftPlus")

ax.grid()
ax.legend()
plt.ylim(-0.1, 3)
```

### 데이터 읽기 / 결과 확인

- 배치단위로 읽는다
- iterbatches() 함수 사용

```python
def iterbatches(epochs):
  for i in range(epochs):
    for batch in dataset.iterbatches(batch_size=gan.batch_size):
      yield {gan.data_inputs[0]: batch[0]}

gan.fit_gan(iterbatches(100), generator_steps=0.2, checkpoint_interval=5000)

# 이미지 생성 결과 보기
plot_digits(gan.predict_gan_generator(batch_size=100))
```

---

## GAN을 이용한 분자 생성

- GAN을 **그래프로 표현**된 분자 데이터에 적용하고 강화학습을 통해서 특정 속성을 가진 분자를 생성
    - Cao and Kipf가 제안한 **MolGAN**을 사용하겠다
    - MolGAN을 **tox21 dataset**로 학습시킨다
    - 12,060개의 훈련 샘플과 647개의 테스트 샘플 제공
- **모델 구성 : generator, dicriminator, reward network**
- **생성기는** 정규 분포(random.normal)를 갖는 **sample (z)**로부터 **MLP**를 사용하여 **그래프를** 생성
    - dense adjacency tensor A (bond types)와 annotation matrix X (atom types) 생성
- **판별기와 보상 네트워크**는 같은 구조를 가지며 **그래프를 입력으로** 받는다
    - Relational GCN과 MLP를 사용하여 출력 얻음.

![https://github.com/StillWork/image/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-22%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2012.10.58.png?raw=1](https://github.com/StillWork/image/blob/main/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA%202022-11-22%20%E1%84%8B%E1%85%A9%E1%84%8C%E1%85%A5%E1%86%AB%2012.10.58.png?raw=1)

### 데이터 준비

**Tox21** 데이터를 다운로드하고 SMILES을 추출

```python
!pip install deepchem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from collections import OrderedDict

import deepchem as dc
import deepchem.models
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
import tensorflow as tf
from tensorflow import one_hot
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw

from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix
import warnings
warnings.filterwarnings("ignore")
%config InlineBackend.figure_format = 'retina'

tasks, datasets, transformers = dc.molnet.load_tox21()
data = pd.DataFrame(data={'smiles': datasets[0].ids})

# datasets 확인
<DiskDataset X.shape: (6264, 1024), y.shape: (6264, 12), w.shape: (6264, 12), task_names: ['NR-AR' 'NR-AR-LBD' 'NR-AhR' ... 'SR-HSE' 'SR-MMP' 'SR-p53']>
...
```

### 데이터 전처리

- MolGAN 특성을 선택하면서, 최대 원자수, 원자번호 등을 제한할 수 있다 → by `MolGanFeaturizer`
- 계산량을 줄이기 위해서 원자의 최대수를 12로 제한

```python
# create featurizer
num_atoms = 12

feat = dc.feat.**MolGanFeaturizer**(max_atom_count=num_atoms, 
  atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14]) #15, 16, 17, 19, 20, 24, 29, 35, 53, 80])

# 원자수가 12 이상인 분자 제거
smiles = data['smiles'].values   # len : 6264
filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]
filtered_smiles[:10]

# 분자 특성화
**features** = feat.featurize(filtered_smiles)

**# features 개수 : 2081**
```

### 모델 구성

- deepchem이 제공하는 **BasicMolGANModel** 사용 (Keras 기반)
- MolGAN 모델을 만들고, learning rate와 최대 원자 갯수 지정
- MolGAN의 입력에 맞도록 dataset를 준비 - **feature과 adjacency_matrix 필요**!

+) 배치 단위로 읽는 함수 정의

```python
# 모델 생성
gan = **MolGAN**(learning_rate=ExponentialDecay(0.001, 0.9, 5000), 
             vertices=num_atoms, model_dir = 'molgan')
dataset = dc.data.NumpyDataset([x.**adjacency_matrix** for x in features],[x.**node_features** for x in features])

def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}
```

### 모델 학습

**fit_gan** 함수로 모델을 학습시키고, **predict_gan_generator** 함수로 분자를 생성한다

```python
gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)

# gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), 
#             vertices=num_atoms, model_dir = '/content/drive/MyDrive/molgan')
```

### 결과 보기

생성된 그래프를 RDKit molecule 형태로 반환 ⇒ 변환이 제대로 되지 않는 분자 제거 ⇒ 유효 분자수 출력

```python
# 생성된 그래프를 RDKit molecule 형태로 반환
generated_data = gan.predict_gan_generator(1000)
nmols = feat.defeaturize(generated_data)
nmols[:10]
(nmols == None).sum()   # 260

# 변환이 제대로 되지 않는(nmols==None) 분자 260개 제거
nmols = list(filter(lambda x: x is not None, nmols))

# 유효 분자수 출력
len(nmols)   # 740
```

### 중복된 분자 제거

똑같은 그래프 그림 그려진 분자 제거

```python
nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]   # nmols_smiles[:20] 확인

OrderedDict.fromkeys(nmols_smiles)

nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_viz = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]

len(nmols_viz)   # 13
nmols_viz[:3]
```

### 생성된 분자 출력

```python
img = Draw.**MolsToGridImage**(nmols_viz, molsPerRow=5, subImgSize=(250, 250), maxMols=20, legends=None, returnPNG=False)
```