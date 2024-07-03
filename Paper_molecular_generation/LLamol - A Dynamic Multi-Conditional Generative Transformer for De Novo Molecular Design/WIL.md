# LlaMol : A Dynamic Multi-Conditional Generative Transformer for De Novo Molecular Design

## Abstract

- GPT와 같은 generative 모델은 molecular design에도 적용 가능하다.
- 목적 : 잠재적인 electro-active 화합물에서 유기 chemical space를 탐색하고자 한다.
- Llamol은 LLama2 구조에 기반한 generative transformer 모델로, 다양한 public 소스들로부터 얻은 13M 크기의 유기 화합물 데이터로부터 훈련되었다.
    - 유기 화합물의 single-conditional & multi-conditional(4 conditions까지) molecular generation에서 충분한 능력을 보여준다.
    - SMILES 기반의 valid한 분자 구조를 생성하고, 생성 과정에서 3개의 숫자 및/또는 1개의 token sequence를 유연하게 통합가능하다.
    - Llamol은 개별적/ 수치 특성의 조합 또는 conditioning → 새 특성들로 확장할 수 있는 de novo molecule design의 유용한 도구로서 쓰일 수 있다.
- 불완전한 데이터의 시각에서 사용성과 견고성을 높이고자 새로운 훈련 방식인 **Stochastic Context Learning**을 소개한다.

## Introduction

- 새로운 material을 찾고 발전시키는 것은 화합물의 unavailability, 높은 production 비용 등의 challenge가 있다.
→ **Generative model은 *most likely*한 후보들에 집중**함으로써 이러한 challenge들을 해결하는데 도움을 줄 수 있다.
- Transformer를 활용한 GPT 모델은 generative natural language application에 많이 활용되고, 그 중 conditional molecular generation 분야도 포함되어 있다. Generative model은 conditional generation으로 알려진, pre-defined features를 활용한 새로운 molecular generation에 사용되는데, chemical space의 범위를 좁혀 새 후보 분자들을 찾는 것을 가속화할 수 있다.
- **MolGPT** : multiple tasks를 낮은 비용으로 동시에 다룰 수 있는 single 모델을 개발하는 것이 목적인 모델로, 특정 task에 맞는 target 특성들을 쉽게 제공하고 바로 검증할 수 있도록 하였다.
- 다양한 conditions가 있을 때 SMILES로 표현된 분자들을 생성하는 single 모델을 훈련시키기 위해 conditional generation의 방법으로서 ‘**Stochastic Context Learning**’을 제시한다.
    - condition : desired molecular property. 모델은 반드시 이를 충족시키는 분자를 생성해야 한다.
    - token sequence는 반드시 valid하지 않아도 되지만, 생성 과정에 포함될 때 valid한 분자의 일부가 되어야 한다.
- LLama2 기반의 GPT-style transformer 모델을 이용하여 **1개 이상의 conditions/target property**에 기반한 새로운 화합물들을 생성하였다. 이를 위해, **각 특성값에 학습 가능한 embedding을 할당**하여 모델이 수치값뿐만 아니라 associated label도 인식할 수 있도록 한다.
    - numerical properties : SAScore(production cost에 영향), logP, molecular weight(energy density에 영향)
    - optional property : user-defined core structure

## Architecture

**Modified version of LLama2 → 15M parameters + 8 decoder blocks**

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled.png)

**Decoder block**

**(1) masked multi-head self-attention layers**

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%201.png)

$X \in \mathbb{R}^{L \times d_{e m b}}$ (L : input sequence, $d_{emb}$ : embedding vector의 dimension)

attention matrix의 $\operatorname{head}_i \in \mathbb{R}^{L \times d_v}$

**Dot product Self-attention을 통해 Q, K**($L \times d_k$)와 **V**($L \times d_v$) 행렬을 구할 수 있다.

→ 각 $head_i$에 $W_Q,W_K (d_{emb}\times d_k)$ & $W_V (d_{emb}\times d_v)$ 적용

특정한 경우에 $d_k$와 $d_v$를 $d_{emb}/n_{heads}$ = 384/8 = 48로 설정하였다.

Mask matrix $M$을 활용하여 upper right triangle를 mask-out. ($W_O \in R^{h\cdot d_{emb}\times d_v}$)

**(2) Feed Forward Network layer + SwiGLU**

각 FFN layer 뒤에 dropout-layer가 존재한다. 

**Standard decoder architecture과 차이점**

- **Rotary Positional Embedding(RoPE)** 사용 : absolute positional information과 relative positional information을 attention matrix에 바로 encode 가능하다.
- Layer normalization 대신 **RMSNorm**을 사용하여 정규화 → 학습 효율의 향상
- 새로운 context ingestion process를 제시 → SCL 방법과의 조합

**Llama2 architecture과 차이점** : Grouped-Query Attention (GQA) 대신 full multi-head attention mechanism 사용

Context ingestion process에 사용되는 **concat function** 정의

$Concat(A,B) = \begin{bmatrix}    x  \newline y
\end{bmatrix}$

$X = Concat(C,S)$ 

$C = Concat((t_1,t_2,...,t_n)^T, t_{ts})$  (*t : embedded vectors. *n: # of numerical conditions)

$S$는 SMILES 그 자체를 의미한다.

Context에서 Multiple tokens → Multiple embeddings. Control of property generation.

- conditions - numerical conditions & token sequence embeddings
- SMILES itself

모든 context(condition) 요소에 embedded ‘**type identifier**’과 **RoPE** 적용

- 각 property에 고정된 type의 number를 할당하고 learnable vector로 mapping ($\sim$type encoding과 결합)
- 사실 numerical values에는 RoPe와 같은 positional information을 적용할 필요는 없지만 적용하더라도 성능이 저하되지 않음. (통일성)

Temperature parameter

- model output의 creativity 정도를 의미
- positive real number (divide output log probabilities)

## Training

### Dataset

***OrganiX13***라고 명명한 dataset 사용 : 13.1M (# of SMILES strings of mostly organic and/or druglike molecules)

- RDKit에 의해 parsing 안되는 entry들 제거
- **256** token 제한을 넘어가는 분자들($\because$ Max_SMILES 영향)이나 ionic structure를 가진 분자들 제거
- **LogP, Molecular weight, SAScore** 반영 - LogP(12 units) & Molecular weight, SAScore(1-6)

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%202.png)

### **Procedure**

(1) SMILES → a sequence of tokens : **DeepChem의 BERTtokenizer** 사용 (591 token으로 vocab size 고정). character level에서 single token으로 취급되는 token을 제외하고 split.

(2) token → $d_{emb}$-dimensional embedding space로 mapping하는 **lookup table**로 token 전달. 

(3)-1. 각 numerical property → embedding space → embedded type identifier를 사용하여 모두 결합 (properties는 미리 계산)

(3)-2. token sequence를 context로 사용할 경우, 현재 토큰화된 SMILES의 이후 토큰 연속적으로 예측. 다양한 size와 content.

(4) token embedding을 decoder로 전달

*context token의 최대 sequence 길이 ≤ 50

input embedding table과 embedding layer를 공유하여 학습된 label embedding이 결합된 sequence embedding에 더해진다.

### - Stochastic Context Learning(SCL)

**다양한 condition들을 조합하여 multi-conditional generation을 하기 위해서** 사용된다.

input sequence에 추가되는 **context에 적용** : **numerical conditions** + **token sequence** 에 대한 embedding

- numerical conditions : SAScore, molecular weight, logP
    
    desired molecular property. 모델은 반드시 이를 충족시키는 분자를 생성해야 한다.
    
- a token sequence : user-defined core structure, associated label (**optional**)
    
    token sequence는 반드시 valid하지 않아도 되지만, 생성 과정에 포함될 때 valid한 분자의 일부가 되어야 한다.
    

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%203.png)

Condition이 삭제 대상으로 선택되면 batch의 모든 entries로부터 대응되는 row를 제거한다.

batch의 모든 분자가 n개의 numerical properties를 가진다고 가정한다. 만약 분자가 properties의 일부만 가진다면 단순히 결측값들을 pad한다.

만약 모든 n개의 numerical conditions 와 token sequence가 제거된다면? ⇒ **unconditional**!

### - Loss

actual next token과 predicted probability 사이의 **cross-entropy loss**로 training. (autoregressitve loss)

Adam optimizer 사용. 256 batch size + gradient accumulation steps of 4 batches.

각 sequence는 CLS token으로 시작해 SEP token으로 종료되며 PAD token도 존재한다. SEP token을 예측하거나 token limit에 도달하면 다음 token을 예측하는 반복 작업을 멈춘다.

Computing resource : Nvidia A100 GPU 2일 + 35 GB VRAM 사용

## Results and Discussion

**Temperature = 0.8**

**Metrics**

- Novelty : reference dataset에 없는 새 분자 생성 비율. training data에 얼마나 의존하지 않는지를 의미한다.
- Uniqueness
- Validity : 생성된 SMILES의 수 대비 적절하게 parse된 SMILES(by RDKit)
- Mean average deviation : context 영향 측정

$y_i$ (the target value of the respective property) $x_i$ (real property value) 의 절대 편차로 측정하며, 낮을수록 좋다.

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%204.png)

### Unconditional Generation

20,000개 SMILES 생성

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%205.png)

분포의 빈도는 training molecules의 representative sample에서 얻은 것과 거의 비슷하다. (training data의 inherent 분포 학습)

MolGPT와 비교했을 때 novelty가 상당히 더 높다. (더 큰 데이터셋을 사용했기 때문이라고 추측)

### Single Condition

10,000개 분자의 sample 생성

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%206.png)

$[a, b; c]$ : a부터 b까지 c 간격으로 구성된 이산 분포값

$\{{a_1,a_2, ...}\}$ : 특정 집합으로부터 sampling한 값 묶음

- in-distribution & out-of-distribution을 다루는 broad interval of values
- 선택된 몇몇 in-distribution target values의 성능에만 집중

한 가지 특성에만 훈련되어 낮은 probability를 가짐에도 불구하고 task를 잘 수행한다. 

MolGPT와 비교하였을 때 single condition case에서 약간 낮은 MAD값을 가지면서 동시에 같은 uniqueness, validity를 가진다.

### Multiple Conditions

각 target property의 쌍별로 10,00개 SMILES 생성

생성된 분자의 실제 특성들은 원하는 값 가까이 둘러싸여있다. 

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%207.png)

Llamol에서 차례대로 unconditional, (logP+SAScore), (logP+molecular weight), (SAScore+molecular weight), all properties 를 의미한다. (property를 condition으로) 

만약 condition을 사용하지 않은 경우에는 빈칸으로 나타내었다.

MolGP보다 (logP + SAScore)에서 LogP MAD값이 낮게 나왔다.

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%208.png)

Fig4) 낮은 logP에 대해서는 잘 작동했지만 높은 logP에 대해서는 SAScore 편차가 상승하였다. → logP가 SAScore과 비교해서 생성 과정에 대해 우선순위가 높다다. (해당 영역에서 training data가 부족한 영향일 것이라고 추측)

Fig5) 더 작은 편차를 가지므로 molecular weight가 생성에서 logP보다 우선순위가 높다는 것을 알 수 있다. 엄격한 크기 제한이 있음에도 logP의 정확도가 유지된다. (logP 값이 요구하는 보다 광범위한 고려 사항과 달리 각 원자의 기여도를 세어 분자량을 쉽게 결정할 수 있기 때문이다.)

Fig6) molecular weight가 생성 과정에서 더 높은 우선순위를 가진다. 모든 경우에 SAScore을 고정할 수 없다. 사용 가능한 요소의 크기와 범위가 제한되어 있기 때문에 충분한 수의 challenging motif를 작은 분자에 통합하기에는 어렵다. 반대로 큰 분자의 경우에는 더 낮은 편차를 가진다.

Fig7) 생성된 분자들을 시각화한 결과로, 교집합이 없는 point cloud의 형태인 것을 확인해볼 수 있다. 

### Token Sequence Incorporation

**Substructure Matches(SM)** : target 일부를 명시적으로 포함하는 생성 분자들의 비율 측정. → 새로운 metric

- 먼저 target 구조를 SMARTS 패턴으로 변환한다. SMARTS는 분자 구조 안에서 특정 원자들 또는 하위 구조들을 match하기 위한 정규 표현이다.
- 결합 순서와 관련된 정보는 제거하고 연결성만 유지된다. 따라서 전체 topology는 유지하면서 전자 구조의 세부적인 사항의 변경은 허용한다.
- target 구조가 생성 과정동안 얼마나 자주 유지되는지 측정한다.

![Untitled](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%209.png)

위의 table은 다른 condition를 적용하지 않은 organic target 구조(SMILES 형태의 context token sequence)를 보여준다. column은 1k/SM uniqueness를 가진다.

Thiophene를 제외하고 새로 생성된 대부분의 SMILES에서 적어도 한 번 이상은 target 구조를 복구할 수 있기 때문에 잘 작동되는 모델임을 알 수 있다. 하지만 Morphine과 같은 더 큰 target 구조가 주어지면 생성된 구조가 매우 반복적으로 변하는 것을 관찰할 수 있다.

Building block Thiophene를 포함하는 구조를 생성하는 것에 대한 성공률은 SMILES 문자열의 공식에 의존한다. Thiophene은 training 데이터셋에서 일반적으로 hetero-aromatic 하부구조를 가지는데, training data의 대부분은 aromatic 표기법을 사용한다.(고리 원자가 시작하는 위치에 의존) 전체적으로, 적어도 하나의 Thiophene 하부구조를 포함하는 약 144K 개의 SMILES 문자열을 발견했다.

Target 부분의 회복률은 aromatic synonym의 경우 10%부터 70%까지 다양하지만 training 데이터에서 관찰된 각 synonym의 상대 빈도와 낮은 상관 관계를 가진다. 반대로, SMILES는 kekulized(같은 분자의 Lewis 구조를 번갈아나오게) 하기 떄문에 특정 token sequence에 대해 90%의 회복률을 가진다.  

이러한 성능의 차이는 Thiophene의 주요 특징으로서 aromaticity를 정의하는 복잡한 규칙과 tell-tale sulfur 원자의 상대적 위치때문에 target context token sequence 내부의 측면으로 볼 수 있다. 반대로 이중 결합은 training 데이터셋이 거의 불완전함에도 전체 구조보다 reliable하게 (재)구성할 수 있다.

### Token Sequence with a Single Numerical Condition

![column: condition으로 사용한 property](LlaMol%20A%20Dynamic%20Multi-Conditional%20Generative%20Tran%20286c5593c86a45dd9a3946064008fc7e/Untitled%2010.png)

column: condition으로 사용한 property

single numerical conditions와 함께 token sequence conditions의 조합을 연구해보았다. (각 조합은 1000개의 생성 분자들로 테스트, 수치값은 table header에 나와있는 것과 같은 범위로 sampling)

대부분의 경우, 이전에 numerical conditions 없이 실행했을 때와 비교하여 테스트 분자들과 일치하는 하위 구조의 수가 감소하였다. logP와 SAScore의 MAD값도 token sequence 없이 생성하는 것과 비교하여 현저히 높았지만 허용 가능한 한계 내에서 유지되었다. → 모델이 2 개의 경쟁 가능성이 있는 조건을 동시에 처리해야 하기 때문일 수도 있다. 만약 2개의 조건들이 충돌한다면, 특정 중요한 이상치가 발생할 수도 있다. 반대로 조건들이 잘 할당된다면 error가 이전 결과와 같이 유지된다. 

Morphine의 경우 SAScore가 다른 예제들보다 상당히 높은데, 이 경우에 모델은 SAScore보다 token sequence에 우선순위를 두며,  더 높은 MAD로 이어진다.

**<정리>**

Token sequence condition은 높은 MAD값에 근거해 대부분 logP와 SAScore 기준보다 우선시된다. 하지만 Morphine과 같이 더 큰 분자의 경우, 매우 낮은 MAD값에 근거해 분자량을 token sequence보다 우선시한다.

### Token Sequence with Multiple Numerical Conditions

모델은 낮은 MAD값에서도 일관된 성능을 유지하지만, condition들이 token sequence와의 조합에서 매우 제한적으로 작용한다면 높은 MAD값이 나올 수 있다. 

예를 들어, logP와 molecular weight condition이 적용된 Paracetamol의 경우 molecular weight이 더 검증하기에 쉽고 logP에 비해 더 확실한 제약이 있기 때문에 molecular weight condition을 우선시한다.

그러나 모델은 위의 표에서 볼 수 있듯이 하위 구조 일치 비율이 더 높고 특성에 대해 낮은 MAD값을 가지고 있어 대부분의 경우 3가지 condition을 효과적으로 만족한다. 특히 3가지 특성을 갖는 분자를 생성할 때 일부 MAD 값은 2개의 특성 생성 과정에서 관찰된 것보다 더욱 낮다. (모델이 더 많은 수의 3가지 특성 배치에 대해 훈련되어 성능이 향상되었기 때문)

## Conclusion

**목적** : 유기적이고 잠재적인 electro-active 화합물의 subspace와 연관된 chemical space를 탐색하는 도구 제공

- LLama2 architecture에 기반한 GPT-style transformer 개발 : single-conditional & multi-conditional generation에서 좋은 성능을 보인다.
- 다양한 소스 기반의 13M 유기 화합물들로 구성된 훈련 데이터셋을 활용하여 다양한 분자 구조들을 생성하는 능력을 향상시켰다.
- Stochastic Context Learning(SCL)이라는 새로운 훈련 방식을 제시하여 single 모델을 활용한 다양한 condition 조합의 multi-conditional generation을 가능하게 하였다.

다른 application에도 적용할 수 있는 generic & adaptable한 모델 → condition으로서 특성들의 수와 종류를 선택하는 것은 탐색하고자 하는 chemical space의 범위를 좁히는데 도움이 된다.

이론적으로 single 모델은 훈련 과정 중 SCL 접근 방법을 통해 넓은 범위의 conditions 조합을 학습할 수 있다. SAScore, molecular size, logP를 선택하였다. 추가적으로, single 모델의 훈련 비용도 감소한 효과를 얻을 수 있었다. 

⇒ 모든 샘플에 적용 가능한 전체적 특성을 필요로 하지 않기 때문에 더 유연하고 확장 가능한 훈련 프로세스를 가능하게 한다.

**Future works**

- 특성들에 대해 concentrated 분포를 가지지 않은 데이터를 추가로 수집하여 거대한 데이터셋에서도 잘 작동할 수 있는 모델로 발전시키고자 한다.
- HOMO-LUMO gap과 같이 실제 application에 유용한 조건이 더 많기 때문에 모델에 부여되는 특성의 수를 확장하고자 한다. (extensive property 예시 : enthalpy of reaction)

## - 요약 -