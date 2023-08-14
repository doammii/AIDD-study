# Graph Model & Graph convolution

## GNN 이해

그래프에 포함되는 정보 (*embedding : 아날로그값. 벡터)

- Vertex(node) embedding
- Edge(link) attirbutes & embedding
- Global(master node) embedding

Matrix

- Adjacency matrix : node간의 **관계(edge)**를 표현하며 방향성이 없는 경우 대칭적이다.
- Feature matrix : node에 담긴 **정보를** 나타내며 각 노드마다 서로 구분되는 특성.
    
    node마다 다른 feature 값(n-bit) 가짐.
    

Graph prediction task

- graph-level : 그래프 전체의 특성 예측
    - 분자 냄새, ring 포함, 수용체 결합도 예측 등
    - image classification이나 문서의 감성예측과 유사
- node-level : node의 identity나 역할 예측
    - 각 노드가 어느 특정 노드와 가까운지 분류 예측
    - image segmentation(각 픽셀 역할 예측), 문장에서 POS 예측과 유사
- edge-level : node간의 관계 예측 (이미지 객체 간의 관계 기술)

### GCN(Graph Convolution Model)

CNN에서 커널 계수를 학습으로 찾아내듯이 **분자구조를 기술하는 계수**를 학습으로 찾는다.

분자그래프의 노드와 엣지를 벡터로 변환한다.

변형 : 그래프 컨볼류션(**GraphConvModel**), 위브 모델(**Weave** model), 메시지 전달 신경망(MPNNModel), 딥텐서 신경망(DTNNModel)

단점

- 분자그래프만 사용하므로 분자구조에 대한 정보가 사라진다. **연결정보만** 존재.
- 거대 분자에는 잘 동작하지 않는다.

edge로 연결된 **node끼리 정보 교환**을 표현하는 방법이 필요 → **convolution** 사용하는 방법 채택!

> 전통적인 CNN의 2D convolution과 차이 : **연결된 node들만 선택적으로 업데이트. 주변 node의 정보를 spatial하게 얻어 정보 업데이트.**
> 

### Graph pooling

복잡한 구조의 graph structure을 **단순화**(low-dimension화)

edge나 node의 정보를 서로 전달하고자 할 때에도 사용 가능!

- edge별로 연결된 node features 모으기
- features를 합쳐서 edge로 보내기(**pooling**)
- parameter를 이용해 prediction 수행

3D 분자의 **구조적 정보**를 반영 → Graph feature representation

### Message passing

⇒ 가중치를 어떻게 구할 것인가 + 주변의 구성에 따라 다른 embedding value 가짐.

node 혹은 edge간의 **연관성을** 고려하면서 **feature을 업데이트**하는 방법

Message passing을 여러 번 반복해서 receptive field를 넓힐 수 있고 더 좋은 representation을 얻는다!

예시) 

node를 주변 node 정보를 이용해서 업데이트하고 싶을 때의 message passing 과정

1. Edge로 연결되어있는 node의 정보(features, messages)를 모은다.
2. 모든 정보를 aggregate function (sum, average 등)을 이용하여 합친다.
3. Update function(parameter)을 이용해서 새로운 정보로 업데이트한다.

## GNN 구현

**GNN**

- 분자의 표현형으로 그래프를 사용
- 분자의 동작을 그래프 컨볼류션 네트워크로 모델링하는 방법

**분자의 그래프 표현형**

SMILES(문자열), Descriptors(테이블), Fingerprint(비트 벡터), Graph

- 각 원자의 속성을 나타내는 **feature 벡터**가 있고 이들로부터 구성된 **feature matrix**를 정의한다.
- 각 원자들의 연결 정보를 나타내는 **adjacency matrix**를 정의한다.

![Untitled](Graph%20Model%20&%20Graph%20convolution%2092cca071d2974ec5a17db78a59f72cd5/Untitled.png)

**Graph Convolution Network**

- 일반 CNN
    - 이미지나 시계열의 패턴 분석에 널리 사용된다
    - 신호가 주변의 샘플들과 같이 **필터를 통과하면서 어떤 추상적인 패턴**을 추출한다
    - **컨볼류션 계층’layer’**을 여러번 통과하면서 점차 추상적인 패턴을 찾는다
    - **풀링**(max pooling)을 수행하여 패턴신호(특성)의 공간적인 이동과 정보 축약을 수행한다
    - [CNN 개요](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fmedium.com%2F%40ricardo_fideles%2Fdog-breed-classification-with-cnn-827963a67bdf), [이미지 필터링](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fsetosa.io%2Fev%2Fimage-kernels%2F), [CNN 동작 설명](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftranscranial.github.io%2Fkeras-js%2F%23%2Fmnist-cnn)
- 그래프 컨볼류션
    - 일반 CNN과 달리 그래프로 표현된 입력을 대상으로 동작한다
    - 주변 샘플 전체가 아니라 **그래프로 연결된 샘플들만 사용**하여 컨볼류션과 풀링을 수행한다
    - [Distill GCN](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fdistill.pub%2F2021%2Fgnn-intro%2F)

### Import

```python
!pip install deepchem
import deepchem as dc

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit import DataStructs
import tensorflow as tf

from deepchem.feat.mol_graphs import **ConvMol**
from deepchem.models.layers import GraphConv, GraphPool, GraphGather
from deepchem.models.graph_models import GraphConvModel
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from deepchem.metrics import to_one_hot
from deepchem.utils.data_utils import load_from_disk
import tensorflow.keras.layers as layers
```

### 회귀 모델

1. **데이터 다운로드 (logP 예측 : 알려진 실제 화합물의 분배계수로부터,유사한 화합물의 분배계수를 예측하는 데 사용)**
    
    featurizer 선택 / splitter 정한 후 GraphConvModel 함수로 모델 만들기
    

```python
# 데이터 다운로드
logP_data = pd.read_csv('https://raw.githubusercontent.com/StillWork/data/master/logP_dataset.csv', names=['smiles', 'logP'])
print(logP_data.shape)

# csv 파일로 저장
logP_data[:4000].to_csv('logP.csv')
# logP_data.to_csv('logP.csv')

# **표현형을** 그래프 표현형인 **ConvMol로 선택**
**featurizer** = dc.feat.**ConvMolFeaturizer**()

# loader로 CSVLoader를 사용하면서 X, y, 표현형 등을 정의
loader = dc.data.CSVLoader(tasks=["logP"], feature_field="smiles",featurizer=featurizer)
dataset = loader.create_dataset('logP.csv')
# (14610, 2)

dataset.X.shape   #(4000, ) -> 목적 변수 y는 logP값.

# Splitter 선택
splitter = dc.splits.RandomSplitter()
# splitter = dc.splits.ScaffoldSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)

from deepchem.models.graph_models import GraphConvModel
**model = GraphConvModel(n_tasks=1, mode='regression', batch_size=50,** 
                       model_dir="./logP", random_seed=0)
```

1. **모델 학습**

```python
model.fit(train_dataset, nb_epoch=50)
```

1. **예측 결과 보기(샘플, 실제값)**

```python
def show_reg_result(y_test, y_pred, N=50):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    max_err = np.abs(y_test - y_pred).max()
    
    print('R2:', round(R2,4))
    print('MAE:', round(mae, 4))
    print('RMSE:', round(rmse,4))
    print('Max error:', round(max_err, 4))

    # 일부 실제값과 예측값 샘플을 plot으로 비교하여 그려본다 (N 개)
    
    if N > 0:
      plt.figure(figsize=(10, 6))
      plt.plot(y_pred[:N], ".b-", label="prediction", linewidth=1.0)
      plt.plot(y_test[:N], '.r-', label="actual", linewidth=1.0)
      plt.legend()
      plt.ylabel('logP')
      plt.show()

y_pred = model.predict(test_dataset)
show_reg_result(test_dataset.y, y_pred)
'''
강의에서는 더 좋은 성능
R2: 0.9134
MAE: 0.2711
RMSE: 0.3875
Max error: 1.8576
'''

# 20개 샘플만 예측해보기 

solubilities = model.predict_on_batch(test_dataset.X[:20])
for molecule, solubility, test_solubility in zip(test_dataset.ids, solubilities, test_dataset.y):
    print(solubility, test_solubility, molecule)
'''
[-0.3775419] [-0.5] CC(OO)OO
[3.0686867] [2.6] [2H]C([2H])([2H])C([2H])([2H])CI
'''
```

1. **모델 적용**

```python
# 임의의 샘플에 적용해보기
smiles = ['COC(C)(C)CCCC(C)CC=CC(C)=CC(=O)OC(C)C',
          'CCOC(=O)CC',
          'CSc1nc(NC(C)C)nc(NC(C)C)n1',
          'CC(C#C)N(C)C(=O)Nc1ccc(Cl)cc1',
          'Cc1cc2ccccc2cc1C']

mols = [Chem.MolFromSmiles(s) for s in smiles]

featurizer = dc.feat.**ConvMolFeaturizer**()   # 중요

x = featurizer.featurize(mols)
predicted_solubility = model.predict_on_batch(x)
for m,s in zip(smiles, predicted_solubility):
    print()
    print('Molecule:', m)
    print('Predicted solubility:', s)
```

---

### ConvMol 구조

분자의 그래프 표현형 종류 중 하나

```python
x[1]   # <deepchem.feat.mol_graphs.ConvMol at 0x7fedec1b19d0>

# 두번째 분자 보기
mol = Chem.MolFromSmiles("CCOC(=O)CC")

# ConvMol 특성의 구조 (원자가 7이고, 특성 벡터수가 75임)
x[1].atom_features.shape   => **feature matrix**
x[1].get_atom_features()

# 7개 원자의 인접 노드 번호들 보기
x[1].get_adjacency_list()
```

### 분류 모델

- tox21 데이터 사용
- featurizer='GraphConv' 적용

**모델 정의, 학습, 평가**

- GraphConvModel 모델 사용
- 기본 모델 구조를 사용

```python
tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = datasets

n_tasks = len(tasks)
model_c = dc.models.GraphConvModel(n_tasks, mode='classification')
model_c.fit(train_dataset, nb_epoch=50)
metric1 = dc.metrics.Metric(dc.metrics.roc_auc_score)
metric2 = dc.metrics.Metric(dc.metrics.accuracy_score)
print('Training set score:', model_c.evaluate(train_dataset, [metric1, metric2], transformers))
print('Test set score:', model_c.evaluate(test_dataset, [metric1, metric2], transformers))
```

### GCN 직접 구현

- `GraphConv` layer: 그래프 컨볼류션을 수행
- `GraphPool` layer: 주변 노드의 특성 벡터로부터 max-pooling을 수행
- `GraphGather`: 노드(원자) 단위의 특성을 수집하여 그래프 단위(분자)의 특성을 계산: a graph level feature vector
- 이외에 [Dense](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkeras.io%2Fapi%2Flayers%2Fcore_layers%2Fdense%2F), [BatchNormalization](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkeras.io%2Fapi%2Flayers%2Fnormalization_layers%2Fbatch_normalization%2F), [Softmax](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fkeras.io%2Fapi%2Flayers%2Factivation_layers%2Fsoftmax%2F) 를 사용한다

```python
batch_size = 100

class MyGraphConvModel(tf.keras.Model):

  def __init__(self):
    super(MyGraphConvModel, self).__init__()
    self.gc1 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm1 = layers.BatchNormalization()
    self.gp1 = GraphPool()

    self.gc2 = GraphConv(128, activation_fn=tf.nn.tanh)
    self.batch_norm2 = layers.BatchNormalization()
    self.gp2 = GraphPool()

    self.dense1 = layers.Dense(256, activation=tf.nn.tanh)
    self.batch_norm3 = layers.BatchNormalization()
    self.readout = GraphGather(batch_size=batch_size, activation_fn=tf.nn.tanh)

    self.dense2 = layers.Dense(n_tasks*2)
    self.logits = layers.Reshape((n_tasks, 2))
    self.softmax = layers.Softmax()

  def call(self, inputs):
    gc1_output = self.gc1(inputs)
    batch_norm1_output = self.batch_norm1(gc1_output)
    gp1_output = self.gp1([batch_norm1_output] + inputs[1:])

    gc2_output = self.gc2([gp1_output] + inputs[1:])
    batch_norm2_output = self.batch_norm1(gc2_output)
    gp2_output = self.gp2([batch_norm2_output] + inputs[1:])

    dense1_output = self.dense1(gp2_output)
    batch_norm3_output = self.batch_norm3(dense1_output)
    readout_output = self.readout([batch_norm3_output] + inputs[1:])

    logits_output = self.logits(self.dense2(readout_output))
    return self.softmax(logits_output)

# 케라스 모델 사용
gcn_model = dc.models.KerasModel(MyGraphConvModel(), loss=dc.models.losses.CategoricalCrossEntropy())

# 입력은 ConvMol 타입임
test_dataset.X[0]
```

**입력 데이터 생성자**

[https://github.com/deepchem/deepchem/blob/master/deepchem/feat/mol_graphs.py](https://github.com/deepchem/deepchem/blob/master/deepchem/feat/mol_graphs.py)

- 모델은 ndarray 타입의 어레이를 사용하므로 `ConvMol` 객체로부터 X, y, w 를 생성해 주는 함수가 필요하다
- 배치단위로 데이터를 생성해야 한다
- 주요 변수:
    - `atom_features`: 각 원자에 대한 특성 표현 벡터이며 크기는 75이다.
    - `degree_slice`: 주어진 degree에 대해서 원자를 구분하는 인덱싱
    - `membership`: 분자 내에서 원자의 멤버쉽을 정의 (atom `i` belongs to molecule `membership[i]`). `deg_adjs`: 특정 degree에 대한, 인접 원자 리스트
- Data Generator
    - X, y, w를 계속 자동으로 생성해주는 함수 정의
- 모델을 훈련시키기 위해서 fit_generator(generator)를 사용한다
    - generator는 위에서 정의한 data_generator 함수가 생성해준다.

```python
def data_generator(dataset, epochs=1):
  for ind, (X_b, y_b, w_b, ids_b) in enumerate(dataset.iterbatches(batch_size, 
              epochs, deterministic=False, pad_batches=True)):
    multiConvMol = ConvMol.agglomerate_mols(X_b)
    inputs = [multiConvMol.get_atom_features(), multiConvMol.deg_slice, 
              np.array(multiConvMol.membership)]
              
    for i in range(1, len(multiConvMol.get_deg_adjacency_lists())):
      inputs.append(multiConvMol.get_deg_adjacency_lists()[i])
    labels = [to_one_hot(y_b.flatten(), 2).reshape(-1, n_tasks, 2)]
    weights = [w_b]
    yield (inputs, labels, weights)

gcn_model.fit_generator(data_generator(train_dataset, epochs=50))
```

**성능 평가**

- 위에서 정의한 generator를 사용한다.

```python
print('Training set score:', gcn_model.evaluate_generator(data_generator(train_dataset), [metric1, metric2], transformers))
print('Test set score:', gcn_model.evaluate_generator(data_generator(test_dataset), [metric1, metric2], transformers))
```