# Predicting Chemical Properties using Self-Attention Multi-task Learning based on SMILES Representation


## Abstract

computational prediction : Molecular descriptors & fingerprints → low-dimensional vectors로 encode됨.

모델의 성능이 **descriptors**에 매우 의존적이라는 challenge → SMILES를 input으로 사용하는 NLP models와 transformer-variant models 연구

Transformer-variant 모델의 구조적 차이를 알아보고 새로운 self-attention based model 제안

Imbalanced chemical datasets를 사용하여 **self-attention module**의 representation learning 성능을 **multi-task learning environment**에서 평가

## Introduction

Computational methods의 challenge : 많은 양의 chemicals의 properties 평가 어떻게? (전통적 in vivo & in vitro 방법은 비용 측면에서 제한적)

- 예시 : read-across, dose and time response, toxicokinetics and toxicodynamics, QSAR models
- QSAR model : compound’s biological property와 구조적 features간의 대략적인 관계를 파악하기 위한 statistical-mathematical model
    
    **ML QSAR model의 중요한 challenge : Molecular-structural features 선택 → 선택된 features를 고정 길이의 string/vector로 encoding**
    
    **Molecular descriptors & fingerprints는 cheminformatics library(RDKit) + SMILES로 compute 가능**
    
    - 전통적 molecular ML 방법은 molecular descriptors, fingerprints와 같은 제한된 데이터 양으로부터 electronic/topoligical features 얻고자 함.

**Fingerprints** 기반 DL 모델들의 문제점

- SMILES input이 fingerprint로 변형됨 : train 결과를 해석할 때 추가 변환 과정 필요
- fingerprints로 변환되는 특정 substructures의 search space가 제한되거나 무시됨 (다른 원자 환경이 같은 bit로 매핑)

**Embedding vectors를 학습시키는데 SMILES를 input으로** 사용(NLP 영향)

- character → embedding vector → train DL models
    
    chemical compound는 자연어 문장과 비슷한 구조를 가짐.(양적 분석을 통한 frequency 측면에서)
    
    → 각 character notation in SMILES 대응
    
- SMILES를 input으로 사용하려는 시도 : embedding vectors는 vector spaces 가까이에 위치하므로 semantically 유사한 item으로 표현 가능
    
    > **Fingerprints를 사용하는 ML 모델과 비교** : embedding vectors는 데이터셋에 의해 fine-tuned 가능하지만, fingerprints는 생성된 이후에 고정됨.
    > 

Transformer model & variants는 다양한 dataset에 대해 최첨단의 성능 보여줌.

**Transformer model은 multi-head self-attention 개념 구현**

- self-attention method : classification datasets 문장으로부터 다른 정보 추출 가능
- Transformer model의 장점
    - input data에서 long-range dependency 학습 가능
    - recurrent connection을 포함하지 않으므로 paralled computation 실현 가능!

> **Self-attention based** DL QSAR model : descriptors와 fingerprints **compute하지 않고 SMILES를 input으로!**
> 

**multi-task learning(train multiple tasks)**도 포함

- **shared hidden** layers & **discrete output** layers
- 장점
    - training samples의 generalization & implicit augmentation
    - overfitting의 위험 감소 : 다양한 tasks는 다른 양의 noise 가짐.
    - shared hidden layers가 중요하고 고유한(내재된) **features(representation learning)**에 집중하도록 할 수 있음.
- **Representation learning** : DL model이 가진 features
    
    자동으로 유용한 정보 추출 : shared **hidden** layers가 각 특정 task와 관련돼있을만한 정보의 underlying subset capture
    
- **multiple task resemblance** : 다른 task들로부터 시너지를 내려면 input data와 tasks 모두 다른 tasks와 resemble해야됨!
    
    multiple task resemblance를 정의하는건 쉽지 않기 때문에 일반적으로 tasks사이에 여러 **features 공유**
    

> Self-attention Multi-Task learning(**SA-MTL**)을 QSAR model로 제안
> 
- 여러 dataset에 대해 최첨단 성능 달성
- 각 transformer-variant model에 대해 구조적 차이 기술 + 학습에 미친 변화의 영향 보여줌.
- trained model을 분석하는 SMILES embedding의 구현 정보

## Related work

~~모든 new article에 대해 새 benchmark 형성~~하는건 standard benchmark의 부재때문에 chemical compound property prediction의 발전 막음.

- [**http://moleculenet.ai/](http://moleculenet.ai/)** 참고 : **HIV / SIDER / BBBP / clintox** dataset
- **post-processing 또는 split method**를 변화!
- 전통 ML에서 DL로 변화하면서 각 dataset에서 최고 성능을 보여준 모델과 비교
1. **SCFP**(SMILES Convolution Fingerprint) model과 **FP2VEC**(FingrePrint To VECtors) model
- 각 SMILES sequence : 21-bit atomic 특성 + 21-bit chemical 특성 ⇒ **SMILES feature matrix**  compute
    
    FP2VEC : 선택된 ECFP는 lookup table의 매치된 randomized vector로 embedding
    
- 2개 DL models 공통점 : molecular fingerprint 대체 가능 / CNN layers 일부는 충분한 learning capability 제공 가능
    - **SCFP** : custom word embedding method를 사용하여 **embedding layer**을 SMILES에 추가하는 새 방법 제안
        
        **Tox21 data에만 집중 + pre-processing data**(rebundant하거나 problematic instances 제거)
        
    - **FP2VEC** model의 중요 feature은 **multi-task learning** → practical approach
1. **Smiles_Transformer model** : utilize encoder&decoder structures
- pre-training approach(unsupervised) : similar to orignial transformer model (**transformer model을 pretraining에만 사용**)
    
    Encoder model ← SMILES representation
    
    Decoder의 target ← 같은 compound의 다른 SMILES로 설정
    
    - SMILES-Enumerator을 사용하여 target SMILES 생성
- pre-training 이후
    - encoder에 의해 생성된 hidden layer 사용 : molecular fingerprint로서
    - fully connected layer에 적용하여 predict
1. **Transformer-CNN model**
- Smiles_Transformer과 유사점 : similar pre-training method
    
    input - 화합물로부터 생성된 SMILES
    
    output - canonical SMILES
    
- Smiles_Transformer과 차이점 : encoder&decoder 모두 사용하는 ST와 다르게 **encoder(SA module)만** 사용
    
    encoder model의 hidden layer output은 molecular fingerprint 구성
    
- pretraining 이후에 CNN의 여러 계층들 구현
- **SA-MTL model과 차이점** : CNN layers의 위치
    - **SA module** : long-range dependency를 학습할 수 있음. 
    (CNN layer이 학습할 수 있는 최대 dependency 범위는 상대적으로 작은 filter size에 의해 정의됨.)
    - 우리 model의 CNN layer은 **self-attention module 전에** 위치
        
        ⇒ CNN layer은 unit learning을 주로 맡고, SA module은 long-range relationship 관리
        
1. **BiLSTM self-attention(BiLSTM-SA) model**
- SA-MTL model과 같은 basic concept 공유 - ‘**single task**’
- 1) BiLSTM : RNN variant
    
    2) modified SA module
    
    3) dense layer
    
- pre-training보다 prediction 단계에서 **attention module**을 사용하는 것이 더 나음.
(attention module은 trained model을 분석하는 능력)
- **final predictor** : 선행 연구들은 dense layer 채택 → 우리는 **discrete output layer** 채택
- **~~two-character atoms~~** 언급 X

## Method

![Untitled](https://github.com/doammii/AIDD-study/assets/100724454/d28d1b0a-5603-4d60-9ea2-f66bf15a8408)

### SA-MTL model

compound property classification - CNN layer, SA module, discrete output layer로 구성

- **Processing SMILES : character embedding → embedding vector 저장!**
    
    각 symbol(atom/bond)을 하나의 character로 → character-level embedding 수행
    
    특정 원자가 **2 characters로 구성(represent)되었을 때의 문제점**
    
    - 대응되는 원자 부분에서 analyzing error 발생
    - train후에 모델 결과를 **분석할** 때 **모델의 weights는 반드시 input과 같은 차원으로** 변환돼야 한다!
    - Cl에서 C와 l이 분리되면 제대로 변환될 수 없음.
    
    **Character embedding** : 대응되는 embedding vector에 index number 할당
    
    2개의 characters로 구성된 원자에 **atom-embedding vector index** 할당
    
    **수소 원자** : 주로 생략되지만, 아닌 몇몇 경우 존재
    
    - **분석** 단계에서 특정 문자가 atom 또는 bond signal인지는 중요
    - 수소 원자들을 “@”와 같은 bond signal로 생각
    
    > **Embedding vectors**를 분석 과정을 위해 **저장하는** 것 중요!
    > 
- **CNN layer : shared hidden layer**
    - 2차원 convolution + **filter** size 7 (same padding)
    - output shape : [batch size, sequence size, hidden size]
    - CNN layer을 추가하면 multi-task learning 환경에서 **sharing multiple tasks를 통해 features를 학습**하는 것 도움.
    - 고려한 차이
        - training dataset의 size가 작고 class-imbalanced
        - classification task는 target sentence를 가진 tasks과 비교하여 피드백 내용이 적음.
- **SA module**
    
    transformer model의 **encoder만** 포함 : target sentence를 생성하는 decoder은 classification에 필요하지 않음.
    
    output shape : [batch size, sequence size, hidden size]
    
    **position embedding을 없애도 performance 저하 거의 없음.**
    
    - sinusoidal position embedding : 원래 transformer model에서 중요
        
        source 문장의 각 단어와 target 문장의 대응되는 단어 간의 관계 고려
        
    - position embedding의 영향은 우리 모델의 target이 binary-class이기 때문에 제한됨.
    
    SA module의 **multi-head structure**에도 적용
    
    - multi-head code는 SMILES의 sequence를 heads 수에 따라 나눔.
    - 원래 transformer의 각 head는 다른 representation, 다른 position으로부터의 정보 결과에 수반
- **Discrete output layer** : 2개의 fully connected layers로 구성
    
    FC layer은 ~~single/multi 여부~~에 따라 변하지 않음.
    
    각 output layer은 오직 그 task에 따라 fine-tuned
    
    - output size : FC layers 모두 1로 (target classes는 [batch size]의 모양을 가지기 때문)
    
    SA-MTL 구조가 다른 차원 축소 방법들보다 효율적
    
    - max-pooling 그 자체는 computational load를 줄이는 효과가 있지만 변수들의 computation을 아무것도 필요로 하지 않음.
    - pooling : representation을 input의 작고, 불변하는 translations로 만드는 것 도와줌.
    
    **Balancing bias를 적용한 FC layer을 2번** 구현하는 것이 더 효율적일 수도. (ablation study)
    
    - balancing bias : 데이터의 class-imbalance 바로잡음.
        
        훈련 데이터에서 negative to positive instances 비율
        
    - Weighted cross-entropy function : cost function 계산하는데 활용
    
    > Balancing bias를 **discrete output layer의 가중치**로 적용 → class imbalance 문제 완화
    > 

### Smiles_Transformer model과의 차이

**pre-train approach** : transformer model을 사용하여 **large unlabeled** data로부터 **atom-level embedding** 학습

> **주의할 점 : pre-training(SMILES data)를 구현하고 싶다면 protein targets도 고려돼야 한다!**
> 

Tox21 dataset에서 특정 화합물은 NR-AhR target task에서 독성이지만 NR-ER-LBD target task에서는 비독성

특정 화합물이 ligand로서 역할 → 화학적 독성은 protein target에 따라 달라짐.

**simple predictor model** 사용 : pre-train의 중간 결과를 기반으로 dataset **fine-tune**에 사용.

### Transformer-CNN model과의 차이

pre-training approach

two-character atoms : SMILES_Transformer model과 비교하여 더 복잡한 predictor을 사용하여 더 높은 scores 달성

~~**multi-task learning scheme**~~ 구현 X

### BiLSTM-SA model과의 차이

**[차이점 1]** : **BiLSTM**을 첫번째 구성 요소로 사용 ↔ SA-MTL은 CNN layer

**CNN을 RNN으로 대체**하여 실험 (ablation study) : 우리 모델은 중요한 성능 차이 없었음.

recurrent 구조는 CNN 구조보다 **computational speed**가 안 좋음.

**[차이점 2]** : BiLSTM-SA model의 SA module은 **반복되지 않는다**.

원래 SA는 multi-layered 구조를 가짐. 

우리는 hyperparameter optimization 이후에 **7-layered SA** 사용

**성능 차이 발생하는 메인 이유 : BiLSTM-SA model은 ~~multi-task learning~~ 구현 X**

- transformer-variant model은 모두 final prediction layer로서 dense 또는 fully connected layer 사용
    
    → 우리의 **discrete output layer**
    
- **two-character atoms** : 훈련된 모델의 가중치를 분석하는데 중요! (BiLSTM-SA 모델은 없음.)
    
    중요성 무시하기 쉬운 이유 : Transformer-CNN의 저자가 주장한 바와 같이 two-character atoms는 성능에 중요한 영향 X
    

## Experiments

SMILES로 chemical compounds 표현

- Tox21, HIV, CLINTOX에서 positive-negative ratio가 매우 **imbalanced**
- 검증 데이터에 대해 **AUC를 최대화**할 수 있는 model hyperparameters를 선택 → **최적화**
    
    random search → optimal hyperparameter 선택
    
    각 **검증** 데이터의 model **weights**는 저장 + **test** 데이터에서 모델을 사용하여 예측 성능 측정(**5번동안 반복된 평가의 평균**)
    
    **Performance metric : ROC-AUC**
    
- **비교 model** : CNN-based(SCFP, FP2VEC) / Transformer-variant(BiLSTM-SA, Transformer_CNN, Smiles_Transformer) / DeepTox 방법
- Tensorflow 2.1 ver.
- Nvidia Titan RTX GPU 환경에서 수행

### Tox21 (*Table III*)

TOX 21 Challenge(2014) : chemical structure data로부터 추론될 수 있는 화합물로 biochemical pathways 간섭 해명

대략 8000개의 SMILES compounds 포함 → 12 target class들로 classify

compound instance는 multiple class들로 태그될 수 있음 → **multi-task learning**

- **1st evaluation : train & test data with random split**
    
    SA-MTL이 가장 높은 AUC 기록 (CNN > BiLSTM-SA > GC > Transformer…)
    
    - Transformer_CNN : CNN을 encoder 뒤에 사용하는 것이 부적절
    - Smiles Transformer : transformer을 pretraining에만 사용
- **2nd evaluation : score data - ensemble**
    
    타 data에 비해서 약간 다른 분배
    
    **DeepTox**(TOX 21 Challenge 우승 모델) : **여러 다른 모델들의 ensemble** 기법 사용. 하나의 SA-MLT 모델의 여러 결과 사용.
    
    **우리 ensemble method : 같은 모델의 5 output probabilities 합침. (초기 가중치는 다르게 결정될 수 있음)**
    
    → DeepTox & SCFP와 비교하면 “score” data에서 더 나은 성능
    

### SIDER (*Table IV*)

drugs, associated adverse reactions(**ADR**) - clinical trials동안 기록됨. public documents & package

SIDER dataset의 organizers도 NLP 사용하여 ADR 추출

1427 instances ↔ 27 targets

TOX21과 유사하게 compound instance는 multiple class들로 태그될 수 있음. → **multi-task learning**

적은 data instances & 많은 class임에도 불구하고 Tox21 dataset과 비교하여 좋은 성능 보여줌.

### HIV (*Table V*)

Test anti-HIV activity of over 40,000 compounds (원래 data : 3 categories + binary target class)

**ligand의 SMILES만 input으로** 제공되기 때문에 성능 제한 존재.

**scaffold** splitting - 구조적으로 다른 분자들을 다른 train/test subset으로 분리

결과 분석

- Transformer_CNN과 비교하여 AUC 감소
- scaffold splitting으로 얻은 결과는 분리
- KernelSVM과 비교해서더 나은 AUC - DL model보다 전통적 ML model이 더 나은 결과

### BBBP & clintox (*Table VI*)

**BBBP**

- human blood-brain barrier → nervous system 보호
- publications 수로부터 데이터 compile
    
    Filtering out 이후에 2031 compounds 남음. (초기에는 2052개였지만 특정 원자의 valence가 비정상적이어서 제거)
    
- scaffold split method 적용하여 SA-MTL 평가
    
    0.966 AUC score과 같이 높은 score 원인 : **positive to negative ratio**
    

**CLINTOX :** FDA-승인 drugs list + 독성 가진 drugs list를 사용하여 dataset compile

⇒ **두 dataset 모두 그렇게 중요하지 않은 noise 포함 + 목적이 ML 예측에 적합**

### Ablation Study (*Table VII*)

SA-MTL의 여러 **features**의 효율성 평가. (best performing model부터 시작해 개별적으로 features **제거해나감. → 성능의 변화 추적**)

Tox21 dataset의 5번 모델 테스트의 평균

- **two-character embedding / SA / multi-task learning / CNN layer**
    - character embedding assign (two-character atoms)
        
        character embedding model이 검증 데이터에 대해서는 0.9이지만 **테스트** 데이터에 대해서는 시간에 따라 0.87까지 **내려감**.
        
        → two-character atoms의 random 분배 때문
        
    - SA module + multi-task learning scheme
        - multi-task learning없이 평가했더니 AUC score 나빠짐.
        - SA module없이 평가했더니 0.798 (SA module 없는 SA-MTL은 FP2VEC model과 유사)
    - discrete output layer을 max-pooling layer로 대체하면 성능 향상
        - simple model + imbalanced data에서는 더 나은 성능
        
        > 차원 축소 단계동안 ~~정보 손실~~의 가능성 있는 **layer 선택** : SA module같은 복잡한 요소에 중요한 영향!
        > 
    - CNN layer 없이 평가했더니 안 좋은 결과
        
        multi-task learning 환경에서 SA module은 multiple tasks의 shared factors를 학습하는데 불충분 (+ imbalanced data)
        
        > **Additional** **data + 목적 함수로부터 더 명시적 피드백** → 성능 향상!
        > 
- **CNN layer / discrete output layer**
    - CNN을 RNN(gated-recurrent unit)으로 대체
        
        하지만 recurrent structure이 **느려서 CNN** 구조 선택
        
    - **final layer에서** discrete output layer 대신 max-pooling layer 사용하여 평가
        
        max-pooling은 simple model에서 좋은 성능, discrete output layer이 더 좋은 성능
        
- **multi-head feature / position encoding**
    
    5 multi-heads & position encoding
    
    - multi-head feature은 화합물 예측에 중요 영향 끼치지 않는다. **classification에 효율적이지 않다**.
        
        multiple different position 기반 정보로 수정
        
        → **over-parametrization issue** (SA heads는 사소한 정보를 포함하거나 가끔 같은 위치에 - redundancy)
        
    - 각 원자에 position embedding value를 추가해도 모델 성능은 같게 유지
        
        원자의 position은 문법적 의미를 뜻하지 않기 때문에 자연어 문장과 SMILES string는 다름
        
        target 문장과 같은 feedback 없이 classification dataset에 적용 → position embedding은 성능에 제한된 영향
        

## Conclusion

SMILES가 direct input ⇒ “**descriptor-free**”

self-attention + multi-task learning method 구현

Transformer의 encoder은 classification predictor로 제시될 수 있다! ⇒ 여러 dataset에 대해 최첨단 성능
