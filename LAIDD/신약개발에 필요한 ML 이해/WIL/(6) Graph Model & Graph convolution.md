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