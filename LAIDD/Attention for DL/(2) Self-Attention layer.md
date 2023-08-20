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