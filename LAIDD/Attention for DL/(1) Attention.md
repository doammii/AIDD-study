# Attention

## RNN

Neural Network : “FC networks” 또는 “MLP”. 주로 Many to many network(for machine translation)에 대해 다룰 것.

**RNN** : sequence가 진행중일 때 “**internal state**” 계속 update

 모든 time step마다 function과 parameters set은 똑같다.

(아래에서 Ht가 Encoder로서의 역할을 한다.)

![Untitled](https://github.com/doammii/AIDD-study/assets/100724454/1263c857-c61a-447b-b12f-e9149ec05ac7)

## Sequence-to-Sequence with RNNs and Attention

1. **Encoder**
2. From final hidden state predict ⇒ **Initial decoder state S0**(문장) & **Context vector**(**c** = Ht. 의미)
3. **Decoder** (Yt-1, **Ht-1**, c를 인자로)
    
    Encoding, Decoding과정에서 서로 다른 RNN 사용했다!
    
    ![Untitled 1](https://github.com/doammii/AIDD-study/assets/100724454/e49e2657-d707-4d80-884b-f3cc0331e2c3)
    
    ![Untitled 2](https://github.com/doammii/AIDD-study/assets/100724454/024639ad-006e-4c29-b21d-5b8d744105df)
    
    ---
    

**위 방식의 문제점** : Input sequence 길이가 너무 길다면 c가 **fixed size vector**이기 때문에 bottlenecked될 수 있음.

> 각 decoder 단계마다 새로운 context vector를 사용하자! ⇒ Attention
> 

### 새로운 network

1. Compute (scalar) alignment scores
    
    Et,i = Fatt(St-1, Hi)
    
2. Normalize alighment scores with softmax → Attention weights 얻기 위해
3. Compute context vector as linear combination of hidden states
    
    a, h의 곱의 합을 c로 계산
    
4. Use context vector in decoder

![Untitled 3](https://github.com/doammii/AIDD-study/assets/100724454/2dc9320f-23dd-4828-aaad-af3b10cfbdb6)

**Intuition**

- **Don’t supervise attention weights → backprop으로**
- **Context vector** attends to the **relevant** part of the **input sequence**
    
    y와 연관된 x에 대한 a값이 상대적으로 더 높다.
    
![Untitled 4](https://github.com/doammii/AIDD-study/assets/100724454/c6c8e0d3-c91c-4f57-b751-9e2e8d1c4e9e)


### 정리

- Use a different context vector in each time step of decoder
- Input sequence ~~not bottlenecked~~ through single vector
- At each time step of decoder, context vector “looks at” different parts of the input sequence
    
    encoder, decoder에서 각각 다른 부분에 집중
    
- Decoder은 h가 정렬된 순서를 형성한다고 생각하지 않는다! (H1, H2, H3…)
    
    단지 H(hidden layer)를 unordered set으로 취급!
    
- 예시(English → French) + image captioning에도 사용될 수 있음.
    
    diagonal attention은 순서에 따라 대응되는 단어 의미
    
    attention은 다른 단어 순서들을 찾아냄.
    

![Untitled 5](https://github.com/doammii/AIDD-study/assets/100724454/d9d107a1-6d16-4405-a802-4be9dcb30a3d)