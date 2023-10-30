# Self-Attention

### Attention layer

기본

![Untitled](https://github.com/doammii/AIDD-study/assets/100724454/0a6f3702-a011-4dad-9d92-ed8693448591)

**변화된 점**

- Use (scaled) dot product for similarity computation
    
    scaled → sqrt 함수 이용 → gradient 잘 나오도록(similarity 커지도록)
    
- Multiple query vectors → parallel 계산 위해
- Separate key & value

Features로부터 query, key, value 출력

![Untitled 1](https://github.com/doammii/AIDD-study/assets/100724454/d5bd5582-46fd-4a71-9add-a6de2c98f339)

![Untitled 2](https://github.com/doammii/AIDD-study/assets/100724454/d1a2ec83-6e3a-4ce6-8f21-e5a700fee05d)

---

![Untitled 3](https://github.com/doammii/AIDD-study/assets/100724454/e1b8a4b4-86a4-4efa-9e11-5b416c5411f6)

### Self-Attention layer

One query per (input vector)

![Untitled 4](https://github.com/doammii/AIDD-study/assets/100724454/807827e3-75d0-411d-aa0b-a9be5bb35fd7)

**SA layer 특징**

- **Input vector의 순서를 바꾸면**(**permute**) Similarity, V(values), output은 같지만 순서는 바뀜.
    
    → Permutation Equivariant : f(s(x)) = s(f(x))
    
- SA는 벡터 연산 진행순서를 알지 못한다!
- SA layer은 벡터 집합에 대해 동작

**Masked SA layer**

- Don’t let vectors “look ahead” in the sequence. (미리 보기 방지. E값 일부를 다른 값으로 대체)
- language modeling에 사용됨.(predict next word)

**Multihead SA layer**

- H를 독립적인 “attention Heads”에 병렬적 사용
- Hyperparameter : Query 차원 & Heads 수

![Untitled 5](https://github.com/doammii/AIDD-study/assets/100724454/1360028f-1d09-472d-8027-e51dadc98d3c)

### Three ways of processing sequences

|  | RNN | 1D convolution | SA |
| --- | --- | --- | --- |
| 특징 | ordered sequences | multidimensional grids | sets of vectors |
| 장점 | long sequences에 좋음
(RNN layer 이후 hT는 모든 sequence 반영) | highly parallel
(각 출력이 병렬적으로 계산) | - highly parallel
- long sequences에 좋음.
(SA layer 이후 각 출력은 모든 입력 반영) |
| 단점 | parallelizable하지 않아 hidden states를 순차적으로 계산할 필요가 있음. | long sequences에 나쁨.
(많은 conv layers를 쌓아야 전체 sequence를 반영할 수 있음. y1은 x4 못보고 y4는 x1 못 보는 문제) | memory intensive |

> Attention is all you need!
> 

### Transformer

A sequence of transformer blocks

1. X → **SA** (All vectors interact with each other. vector들간의 유일한 interaction)
2. Residual connection
3. Layer Normalization(vector마다 독립적으로 작용)
4. **MLP** independently on each vector(vector마다 독립적으로 작용)
5. Residual connection
6. Layer Normalization(vector마다 독립적으로 작용)
7. Y

**⇒ 매우 scalable & parallelizable**

**Transfer learning (pretraining - finetuning)**

![Untitled 6](https://github.com/doammii/AIDD-study/assets/100724454/e5b4fa19-6447-452c-92e5-11f609fa8c20)

## Summary

### Attention Mechanism and Transformers

p. 213~230

- **sequence to sequence models**
    - RNN → encoder & decoder
        
        (input sequence) → encoder → context(capture한 information을 vector로 compile) → decoder(produce output sequence)
        
        - Encoder : input word sequences, hidden state of encoder in previous time step의 word embeddings ⇒ ‘context’를 representation으로 생성
        - Decoder : predict output sequences & update hidden state. attention 가능하지만 어디에 집중할지가 문제.
    - word embedding : learned representation for text (의미 기준)
    Language tasks에 대해 더 나은 일반화 성능을 가질수록 dense, lower dimensional representation.
        
        seq2seq : tokenizer prompt → word embedding → model input
        
    - Attention이 필요한 이유? : ‘context’는 decoder에게만 사용가능한 정보로서 long sequences를 처리하기 어려울 수 있음. 비효율적 연산.
        
        → Encoder에서 Decoder로의 모든 hidden state를 제공하고 decoder에게 어느 부분에 집중할지를 알려줌.
        
    - Attention Mechanism 작동 원리
        - Attention(alignment) score이 function(feed-forward NN)에 의해 계산됨. (이전 decoder hidden state, encoder hidden state도 같이)
        - Attention scores의 softmax가 대응되는 encoder hidden state에 곱해지면서 현재 time step의 context vector를 계산
- **Transformers (216~219)**
    - Deep feed-forward ANN with SA mechanism
        
        Each encoding & decoding part → SA 또는 encoder-decoder attention layers와 함께 encoder & decoder blocks이 쌓여서 구성. 
        
        task 종류에 따라 encoders only / decoders only / encoders-decoders
        
        (RNN은 hidden states를 이전 words의 representation을 통합하기 위해 사용했다면, Transformers는 input sequence에서 모든 단어로의 직접적, 병렬적 접근 허용)
        
    - **Encoder** : input sequence의 embedded words가 SA layer, feed-forward layer을 따라 병렬적으로 진행. 각 encoder의 출력이 다음 encoder block으로 보내짐. **input으로부터 features 추출**
        - **Positional encoding** : 각 단어의 위치 또는 단어들 간의 거리를 제시하기 위해 사용. 각 단어 위치는 위치 및 차원의 sinusoid function의 벡터에 매핑.
    - **Decoder** : top encoder의 결과는 encoder-decoder attention layer에 사용되는 Key, Value의 집합으로 변형됨. 다음 time step의 맨 아래 decoder로 이어짐. **features로 결과를 생산하는데 사용**
    - **Self-Attention** : 더 나은 encoding을 위해 input 각 단어들의 processing이 다른 낱말들에도 접근할 수 있도록 허용하는 mechanism
        
        자기 자신에 대한 attention 또는 모든 input에 대한 embedding 및 token동시에 ⇒ **데이터 기반으로 순서나 위치 별로 중요 X**
        
        **Computation** of SA : **Vector Form**(input word의 embedding으로부터 얻은 Query, Key, Value가 가중치 행렬에 따라 계산됨. Q와 K dot product) / **Matrix Form**(faster, efficient processing을 위해 softmax, sqrt 사용)
        
        - **Masked** SA : decoder blocks의 SA layer은 output sequence의 더 이른 위치들에만 집중할 수 있도록 허용. 
        → masking future positions with negative infinite value(-inf)
    - 주로 Multi-headed Attention : multiple representation subspaces & diversified focus on input sentence를 위해 multiple attention mechanisms 사용
        - Q, K, V 행렬을 분리하여 각 SA head를 train
        - 다음 feed-forward layer에 multiple representation을 요약 및 전달하기 위해 attention heads의 output representation은 학습가능한 가중치 행렬에 의해 연결되어 곱해짐.
        - Q → K → 토큰에 대한 가중치 * value : 동시에 효율적으로 병렬 연산.
    - **Output layer**
        - final linear FC layer : decoder 결과의 차원을 vocabulary 크기로 확장
        - softmax layer : linear layer의 logit value를 output words의 probability로 변환
            
            The highest value of probability에 해당하는 단어가 각 time step의 결과로 선택됨.
            
        - model 결과와 target label 사이의 오차를 최소화하는 방향으로 훈련됨.
- BERT & GPT
    
    Transformer : 많은 데이터 기반, 그 자체로 bias 없이 학습.
    
    BERT : transformer encoder (auto encoding)
    
    - input의 각 위치에 대해 같은 위치의 output은 같은 token([MASK] token for masked tokens)
    - BERT와 같이 encoder stack만 있는 모델은 결과들을 한 번에 출력
    
    GPT : auto-regressive transformer decoder
    
    - 각 token은 이전 token에 의해 예측되고 조건 결정됨.
    - language generation에는 좋지만 classification에는 좋지 않다.
    - unlabeled large text corpora에 의해 훈련
- **Challenges & Future Directions**
    - Domain Gap in pre-training : 현재 learning-based medical imaging 접근은 transferring learning(ImageNet pretraining)에 의존. pre-training의 cross-modality, cross-task performance 관련 영향도 연구해볼 필요 있음.
    - Intensive Computational requirement : memory&computation은 resource constraint edge device의 확산을 제한함.
    - Unvalidated Robstness(견고성) in Domain shift : 몇몇(비정상) 클래스에 관한 불확실성의 보정된 추정치를 제공하면서 훈련 중에 다룬 클래스들이 정확한지 여부에 대한 ViT-based medical imaging systems에 대한 연구
