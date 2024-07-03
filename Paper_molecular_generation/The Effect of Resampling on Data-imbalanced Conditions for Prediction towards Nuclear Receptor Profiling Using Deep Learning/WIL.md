### Abstract

NR signalling pathway에 기반한 toxicity evaluation → *in silico* prediction tool이 사용됨.

- *in silico* prediction tool이 사용되는 분야
    - early stages of long-term toxicities의 detection
    - 새로 합성된 화학 물질의 prioritization
    - **selectivity & sensitivity의 acquisition**

Computational prediction model : chemical-protein interaction의 **toxicity screening**에 사용됨. (DL이 예측 accuracy를 높인 영향으로)

**문제점** : toxic chemical compound 데이터셋 양이 nontoxic 데이터셋 양보다 매우 적은 **data-imbalanced condition**

→ **toxicity hazard**에 대한 정보를 제공할 수 있는 **toxic dataset의 낮은 예측 accuracy**

1. data imbalance의 effect 조사 - toxicity assessment data ( AR(LBD), ER(LBD), AhR, PPAR을 NR로 사용)
2. toxic-nontoxic 데이터셋 간의 심각한 imbalance 확인

**Selectivity & Sensitivity의 균형** 

→ toxicity hazard의 평가에 필요 

→ (simple) **Data resampling** 방법이 **NR의 toxicity hazard profiling을 위한 이진 분류 task의 bias 문제**를 해결하는데 사용

### 사전 지식

- **nuclear receptor(NR)** : 세포 내 단백질. 유전체 DNA에 직접 결합해 유전자 발현 조절, 전사 인자에 포함됨. ligand-activated transcription factors
    - a family of ligand-regulated transcription factors ← activated by steroid hormones
    - steroid hormones 예시 : estrogen(ER) / androgen(AR) / signalling pathways(PPAR, AHR, thyroid hormone 포함)
- signalling pathway(신호 전달 경로) : 세포 외 신호를 전달하여 세포 내 전사 인자 매개 유전자 조절에 영향을 미치는 핵심 생물학적 메커니즘
    - ligands(chemicals)는 plasma membrane을 거쳐가고 NR와 직접 상호작용 가능 (cell membrane receptors 통한 상호작용보다)
    - ligands가 receptors에 결합하면 즉시 유전자의 transcription 규제

---

## Introduction

- Data-driven models의 발전
    - in silico prediction model → toxicity evaluation
    - computational model(**QSAR**) : activity와 safety profiles 예측 위해 대량의 chemical assay 데이터셋 사용
        
        (+ OECD가 QSAR을 chemical regulatory application으로)
        

Toxicity prediction models 

- 전통적 ML model은 accuracy는 향상시켰지만 **descriptor selection** 필요
    - toxicity와 관련된 중요 structural & physicochemical features 결정
    - domain 지식, 대량의 수동 데이터 처리 필요 +  selected descriptors의 quality에 매우 의존
- DL을 사용하며 toxicity prediction의 accuracy 크게 향상 (DeepTox : toxicity in silico model에 DL application 적용)
- Advanced DL 기술(**CNN, RNN, GAN**) → 예측 accuracy & 다양한 약학 연구 영역의 applicability 향상

Early-stage DL toxicity prediction models 문제점 : **DL의 자동화된 feature 추출 능력**을 완전히 사용하지 못함.

- Screening같은 전처리(by toxicophore) 작업동안 대량 features 추출됨. (모델이 **DL의 representation learning 능력**이 있음에도)
- 예시 : CNN에서 분류를 위한 representative DL feature은 convolutional layers에서 추출됨.

CNN-based toxicity prediction model : **SCFP & FP2VEC**(molecular featurizer)

- chemical structure & fingerprints가 encode됨 → CNN의 **representation learning**
    - SCFP : 각 SMILES sequence이 42-bit으로 represent됨 → SMILES feature matrix 계산
    - FP2VEC : selected ECFP가 lookup table의 매치된 randomized vector로 embedding됨.
- **CNN-based classifiers** : SMILES feature matrix와 Fingerprint Embedding의 features 추출 → active/inactive로 분류
- **Oversampling model** : Augmentation+Shifting 같은 randomized zero-padding position 사용
- SCFP & FP2VEC를 선택한 이유
    1. CNN은 powerful classification model로 알려짐.
    2. 두 모델 모두 toxicity hazard와 매우 관련되어있는 motif를 추출 가능(by convolutional layers의 feature maps 분석)
    3. 두 모델 모두 CNN의 고정된 크기 input을 만들기 위해 zero-padding 사용
    4. DeepTox, GraphConv보다 **ROC-AUC**에서 높은 accuracy 보여줌.

여전히 해결해야 할 문제 : **data imbalanced condition**에서 **minority** class dataset(**toxicity assessment dataset-hazard**) 예측

- classification - minority class(underrepresented class) + majority class(overrepresented class)
- 심각한 imbalanced 데이터셋에서 minority class를 잘 인식하지 못함 → **classificaion의 boundary(결과)가 unclear/biased되는 문제** 발생
    
    특히 rare case들이 연구에서 집중하고 있는 부분일 때 더욱 문제가 됨.
    
- toxicity assessment data의 hazard를 이진 분류 : **inactive** 데이터셋 양이 active 데이터셋 양보다 월등히 많아 **심각하게 imbalanced**
    
    **active**(**minority**, positive instances, **sensitivity** of receptor binding) ↔ **inactive**(**majority**, negative instances, **specificity** of receptor binding)
    
- Toxicity hazard 이진 분류 모델 : 꽤 높은 ROC-AUC임에도 **높은 specificity & very low sensitivity**

**Resampling effect** of toxicity assessment imbalanced (NR 관련) dataset의 data-imbalanced conditions 분석

Specificity & Sensitivity 사이의 예측 accuracy 균형을 맞추는 것은 예전에도 이루어졌었지만, DL toxicity prediction model의 데이터 불균형 영향은 처음 조사

- **Simple resampling** methods(random undersampling, augmentation with shifting) → 6 NR datasets in Tox21 challenge
    - FP-FN 제어하는 trade-off의 한계 존재 → ROC-AUC loss 없이 sensitivity&specificity 사이의 차이를 최소화하고자 함.
    - **hybrid resampling** methods 제안
- SCFP & FP2VEC 분석

## Data Imbalance and Deep Learning Based Toxicity Prediction

### Dataset : NR Dataset in Tox21

toxicity prediction modeling에 사용되는 benchmark 데이터셋

- standard experimental conditions + low noise level
- multiple NRs에 대해 똑같은 화합물 가지고 있어서 multi-tasking learning 적용 가능

training, validation, test sets가 하나로 합쳐진 후 active/inactive로 나뉨.(train : test = 8 : 2)

training set에 대해서는 cross validation 수행, internal test set으로 final validation 수행

**Imbalance ratio range** : 1 : 7.35(ER) ~ 1 : 32.79(PPAR)

12 endpoints는 이진값으로 라벨링. **NR-related assessment data**(AR, AR-LBD, ER, ER-LBD, AhR, PPAR)

### Deep Learning Toxicity Prediction

DL QSAR models : SCFP & FP2VEC

- SMILES : standard representation of compounds - 고정된 특성의 string 형태
- 공통점 : DeepTox, MOL2VEC보다 좋은 성능 / CNN model / Zero-padding 사용
- **zero-padding** 사용 효과 : CNN model의 input을 고정 크기로 만들 수 있다. encoding과정 중 화학적 특성을 유지하면서 **Minority data를 shifting과 함께 augment**할 수 있음.
    
    zero-padding을 위한 encoding 방법 : SCFP는 SMILES symbols의 feature matrix 이후, FP2VEC은 Morgan fingerprint ‘1’로 look-up table encoding 이후
    
    화학 구조가 feature matrix, fingerprint embedding으로 encode될 때 zero-padding은 CNN model input의 고정된 크기를 유지할 때 필요
    
    zero-padding의 랜덤 사이즈가 molecular encoding 앞 또는 뒤에 붙음.
    
- RDkit : 화학적 특성 연산에 사용
- SCFP : improved prediction accuracy as well as the motif detection (by conv layers에서 만들어진 feature maps 분석)
- SCFP modification : 더 나은 feature 추출을 위해 (1) reshape feature matrix (2) reduce filter size
    - feature matrix : 400x42 → 800x21 (odd:atom 특성, even:나머지 특성)
    - filter size of 1st conv layer :  1x42 → 1x3에 stride 2로 바뀜. (SMILES symbols에서 feature 추출하는데 너무 큼.)
    - maxpooling 통한 abstracting : 2x1
    
    장점 : feature extraction을 atom과 그 나머지 특성으로 분리하고 maxpooling으로 abstracting
    
    - filter size of 2nd conv layer : 10x2에 stride 1로 바뀜. average pooling은 1x9
    - FC layer : neuron 수, ReLu, Tanh 함수
    - output 크기 : 2 / softmax 함수가 active 여부 판정에 사용.
- FP2VEC : multi-task learning을 사용하는 CNN-based classifier

### Unbalanced Sensitivity and Specificity

training model → **stratified**(계층화된) **5-fold cross-validation**. 각 fold마다 같은 active/inactive compounds 수

performance analysis(train) & external validation(test) → **4 metrics(Accuracy, ROC-AUC, Sensitivity, Specificity)**

GraphConv 추가 : graph convolution featurizer과 CNN-based classifier 사용. inactive chemical class weight는 제거함.

**[Data-imbalanced condition + no sampling]** *(6개 NR dataset 결과의 평균)*

- SCFP & FP2VEC : accuracy와 AUC는 높지만 sensitivity는 0.5보다도 낮음.
    
    **sensitivity**는 toxicity assessment data 분석에서 중요한 metric. **QSAR의 acceptable accuracy은 약 0.7**
    
- SCFP0 & SCFP : modified 버전의 sensitivity가 약간 더 상승
    
    more feature maps + more small-sized filters 덕분.
    
- GraphConv : AUC는 SCFP와 FP2VEC의 중간이지만 더 imbalanced

## Resampling: Balancing Sensitivity and Specificity

Sampling 기법은 imbalanced training dataset의 문제점을 해결하기 위해 사용 → under-sampling & oversampling이 imbalance ratio를 다루기 위해 사용됨.

**Resampling**은 모든 classifier로의 적용가능성을 다룰 수 있는 방법 중 하나. **training dataset에만 적용 가능.**

원래 균형을 맞추며 AUC를 향상시키는 방법으로 ROC의 optimal threshold를 설정하는 방법이 있지만 resampling 했을 때보다 성능 안 좋음.

- **random under-sampling(undersampling of majority data)**
    - majority class의 data points가 랜덤하게 제거됨. (x*(minority data 수)*(majority 수) 반영)
    - under-sampled dataset + sampling probabilities(U1, U3, U5, U7) → probability가 높으면 inactive 데이터셋 양 증가
    - (no sampling보다) **high sensitivity, low specificity, low accuracy**(전체 accuracy는 majority 데이터셋의 감소한 sensitivity에 영향받음.)
    - **AUC** - SCFP는 U1를 제외하고 높아짐. FP2VEC은 낮아짐.
    - 요약 : **minority data의 accuracy 증가(specificity)** & majority data의 accuracy 감소(sensitivity)
- **oversampling + shift augmentation(oversampling of minority data. Ox)**
    - minority class의 data points가 **random shifting과 함께 augmented**
        - **random shifting**은 zero-padding의 위치를 바꾸는 molecular encoding으로 구현됨.
        - **Augmentation** → zero-padding 위치를 랜덤화는데 사용
        - 화학 구조가 feature matrix, fingerprint embedding으로 encode될 때 zero-padding은 CNN model input의 고정된 크기를 유지할 때 필요
    - oversampled dataset + sampling probabilities(O1, O3, O5) → **sensitivity** 증가
    - **specificity** : SCFP는 낮아졌지만 FP2VEC은 그대로.
    - 증가한 sensitivity는 오히려 **AUC 감소**시킴.
- **hybrid resampling(UxOy)** : undersampling과 oversampling 동시 실행
    - baseline(under-sampling U3, U5) + oversampling(U3O3, U5O5)
    - **Less(limited) oversampling over under-sampling**(U3O2, U5O2, U5O3)이 더 균형을 잘 이룸. AUC loss 없이.
    - AUC : SCFP는 낮아졌지만 FP2VEC은 그대로.
- **two-phase learning(Ux-UyOz)** : 2 iterative training runs + 다른 데이터셋
    - **기본 아이디어** : second phase learning에서 균형을 좀 더 맞추고 under-sampled condition에서 train할 때 inactive chemicals의 losss 최소화.
        - pre-training : imbalance conditions(no sampling, under-sampling)에서 실행
        - fine-tuning : 더 balanced condition에서 실행
    - condition 1(U3-U5O2) : first phase(U3) → second phase(U5O2. over the first-phase trained model)
    condition(U0-U3O2) 2 : first phase(no sampling) → second phase(U3O2). inactive dataset은 phase간 exclusive
    - **SCFP** two-phase learning ↔ **hybrid**(U5O2)와 비교
    - 결과 : U5O2와 비교해 **specificity & sensitivity 증가**. U3-U5O2는 AUC와 accuracy도 향상.
- **mini batch retrieval** : minority data의 learning rate 증가
    - mini-batch는 majority-minority 사이의 비율을 따름. (hybrid resampling + FP2VEC과 비교)
    - active:inactive = 1:2 ← original fingerprint embedding + shifting augmentation이 mini batch에 1:2 비율로 계속 insert돼야 함.
    - sensitivity(0.7) specificity(0.8) - U5O2와 비슷하거나 약간 낮은 성능

### Discussion

DL-based toxicity prediction models(SCFP, FP2VEC) / NR-related data in Tox21

no sampling → imbalance 때문에 낮은 sensitivity (minority active dataset의 accuracy가 더 중요)

Resampling methods가 sensitivity 향상, AUC 향상

- hybrid method of oversampling minority dataset over under-sampling majority dataset
- two-phase learning
- mini batch retrieval

CNN 학습에서 얻어지는 kernel → motif(functional substructure) 추출 위해 분석. feature map에서 motif discover.

- SCFP에서 motif는 resampling에 상관없이 올바르게 추출
- **FP2VEC** : 부분적으로 motif 발견 → FP2VEC에서 feature map의 해석이 더 필요함.
- **resampling**으로 훈련된 모델은 두 모델에서 모두 motif를 올바르게 발견

More complex resampling 기법

- **RF** → clustering based under-sampling + synthetic oversampling for minority dataset(SMOTE)
    
    NR-AhR & NR-ER-LBD (Tox21) 데이터셋 사용 → well-balanced & smaller loss of accuracy & AUC
    
- class weight method(algorithmic learning approach) + **GraphConv**
- **GAN** for synthetic molecular generation : SMOTE에서 좋은 성능

## Conclusion

*in silico* toxicity prediction에 사용된 DL-based toxicity prediction models

SCFP and FP2VEC : benchmarking Tox21 NR dataset → effect of data imbalanced conditions in binary toxicity classification

DL-based models(SCFP, FP2VEC)의 **문제점**

- accuracy, ROC-AUC는 좋으나 sensitivity가 안 좋음.
- specificity와 sensitivity의 심각한 불균형 ← majority(inactive) ↔ minority(active) 간의 차이

5 resampling 방법을 이용해본 결과 : **oversampling minority data가 specificity 향상 + ROC-AUC loss없이 sensitivity와 균형을 이룸.**

(under-sampled majority data, two-phase learning, mini batch retrieval 보다)

→ sensitivity(0.714) / specificity(0.787) / overall accuracy(0.829) / ROC-AUC(0.822)

→ *in silico* combined approach가 화학 독성 스크리닝에 유용한 toxicity prediction model로 통합될 수 있음을 보여줌.

**future studies** (더 발전된 resampling 방법)

- clustering based under-sampling : RF
- synthetic minority dataset oversampling(SMOTE) : RF
- GAN + SMOTE
- (class weight method → GraphConv)

## Reference

P. Banerjee, F. O. Dehnbostel, R. Preissner, Frontiers Chem. (2018, 6, 362)