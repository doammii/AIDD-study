# SCFP

*Convolutional neural network based on SMILES representation of compounds for detecting chemical motif* (BMC Bioinformatics)

**SCFP** : Smiles Convolutional FingerPrint

## Abstract

화합물의 SMILES representation에 기반한 CNN → **detecting chemical motif**

**SMILES**(Simplified Molecular Input Line Entry System)

- Background
    
    신약을 위해 lead compounds를 screening하는 효율적 접근으로 딥러닝 제시
    
    various kinds of (fingerprints & graph convolution architecture) 사용해왔음.
    
    하지만 장점 또는 단점 → (chirality. 휘발성)을 포함한 structural differences 구별 / 자동으로 effective features 찾기
    
- Results
    
    **Compound classification**을 위한 딥러닝 모델
    (SMILES notation 기반 compounds의 **분산된 representation**) + (SMILES-based representation을 **CNN**에 적용)
    
    - **SMILES의 사용 장점**
        - 광범위한 structure information을 통합하는 동안 모든 compounds type 처리 가능
        - CNN에 의한 representation learning : 자동으로 input features의 **저차원 representation** 획득 가능
    
    TOX 21 데이터셋을 이용한 benchmark 실험 → 전통적 fingerprint 방법 능가 + TOX 21 challenge의 winning model 능가
    
    Multivariate analysis : SMILES-based representation learning에 의해 학습된 features로 구성된 chemical space는 richer feature space 적절하게 표현
    
    **Learned filters로 Motif detection** → known structures(motifs)뿐만 아니라 **unknown** functional groups
    
- Conclusions : Chainer 딥러닝 프레임워크를 이용한 software / 성능 평가를 위해 사용된 dataset

## Background

**chemical properties 예측** → chemical analysis를 위해 중요!

in silico analysis : 컴퓨터가 chemical compounds를 읽을 수 있게 **<digital file formats>** 정의됨.(가상 환경 - 컴퓨터 시뮬레이션)

- **MOL** : graph connection table의 형태로 화합물 represent
- **SDF** : multiple compounds를 하나의 파일에 쓰기 위한 MOL의 확장 버전
- **Fingerprints : chemical compound의 property**를 표현하기 위한 **vector(bit string. 이진값 벡터)**
    
    분자들 structure 간의 유사도 측정 위해 features 추출
    
    **2D** fingerprint : compound의 partial structure 종류 나타냄.
    
    - **ECFP**(Extended-Connectivity FingerPrint)
        
        circular fingerprint, Morgan fingerprint… - algorithm의 종류
        
        각 원자 주변의 **partial structures** 탐색 → 정수 identifier 할당 → hash function을 이용하여 **이진 vector**로 write
        
        chemical space에 무한대의 structures 있을 수도 ⇒ **ECFP**는 large number of bits를 가진 vectors 필요!
        
    
    **3D** fingerprint : shape, electrostatics를 포함한 3D information encoding
    
    - ROCS(Rapid Overlay of Chemical Structures) : 단순 force field로 정의된 “color” features 사용
    - USR(Ultrafast Shape Recognition) : 화학 구조 할당 없이 3D similarity 계산
- **SMILES -** modern chemical information processing을 위한 standard representation of compounds로 사용
    - linear notation → 고정된 알파벳을 넘어 chemical compounds를 string 형태로 unique하게 표현
    - 특정 grammar & characters 사용
    - strictly express **structural** differences
    - **SMILES string** : SMILES representation의 선형 구조
        
        → **CNN**의 (virtual screening of chemical compounds) & (functional substructures의 identification) 간단한 적용 가능하게 함.
        
        **chemical motifs** : functional substructures
        
    

Chemical analysis + machine learning(DNN) → contests(TOX 21 Challenge 2014, Merck Molecular Activity Challenge 2013…)

하지만 딥러닝의 full use of capability를 만족시킬 수 없음.

**딥러닝**(CNN) : 수동으로 features를 devise하는 것 대신 최대한 “data로부터 자동으로 feature 획득할 수 있다(*capability*)”는 장점

**⇒ representation learning**

**machine-learning-based** finterprinting 기술 발전의 발판

compounds의 **graph structure**를 수동-설계 fingerprints의 대체용

- backpropagation convolutional network를 이용하여 fingerprints 일반화(Duvenaud)
- graph convolution을 사용해 fingerprints 향상

(**한계**) 

- **고정된 구조**의 compounds 집합만 input으로 가능 / stereoisomers(입체 이성질체) 중 몇몇 구별 불가
- **~~graph structure~~** : 일반적으로 **CNN**을 효과적으로 사용할 수 있는 2D와 같은 **grid-like topology**의 데이터 구조가 아니다.

---

> SMILES linear representation of chemical compounds  ⇒ CNN 적용
: chemical compounds의 classification + detection of chemical-motifs
> 
- **String** : simplest **grid**-like(1D-grid) structure
    
    예시) Molecular sequences(DNA, protein)
    
- **CNN**
    
    DNA sequences의 classification & sequence motif의 추출
    

4 DNA nucleotides의 one-hot coding representation 사용

→ **Filter(kernel)** + (sequnece에 적용된 1D convolution operation)

- sequence - 1D” convolutional operation [해석] : input sequence를 같은 width(dimension)의 filter와 함께 sequence를 따라 한 방향으로만 scanning (입력의 분산된 representation)
- **접근** : **1D CNN** → (**SMILES** strings representing chemical compounds)에 적용. 
chemical compounds의 **classification + chemical motifs(structures)의 추출**을 위해

→ motif를 represent하기 위한 **position weight matrix**

- **Filters**
    
    CNN training에 의해 학습 - positive&negative samples of sequences
    
    예시) chromatin immunoprecipitation with HTS **(ChIP-seq)**
    
    크로마틴 면역 침전 : 세포 내 단백질과 DNA간의 상호작용을 조사하는데 사용되는 실험기법
    

TOX 21 dataset으로 실험 + ROC-AUC score로 결과 분석

결과 : **One-dimensional CNN**(using the **SMILES** representation)가 ECFP fingerprint methods, graph convolution method보다 뛰어남.

(+) protein-binding sites 같이 중요한 known structures(motifs)는 1D CNN에서 learned filters에 의해 발견됨.

또 다른 CNN의 advanced **feature** : “**representation learning**”

- effective features가 자동으로 머신러닝 과정에서 발견될 수 있음.
- 예측 모델에 fit한, 화합물을 위한 새로운 fingerprints 또는 descriptors를 확장적으로 얻을 수 있음.
(특정 단백질에 결합 예측)
- 중요한 functional substructure인 “chemical motif”를 추출할 수 있음.

⇒ SMILES representation 기반 representation learning에 의해 발견된 **new fingerprints : richer chemical space 제공**
(화합물의 정확한 구별 가능하게 함. **ECFP를 사용하는 방법들은 properties를 표현불가.**)

**chemical space** : “**multi**-dimensional descriptor space”

---

![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled.png)
![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%201.png)

## Methods

SMILES string을 SMILES feature matrix라는 분산된 representation으로 표현했고, CNN에 적용했다!

CNN은 SMILES feature matrix → SCFP(저차원 feature vector)로 transform

SCFP가 이후의 fully-connected layers를 위한 input! 화합물을 위한 classification model 만듦. 

CNN에서 얻은 feature representation을 “chemical motif”의 형태로 추출

![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%202.png)

### SMILES notation for representing chemical compounds

symbols의 2 sets → (atomic symbols) + (SMILES original symbols)

- atoms ← atomic symbols
- double.triple bonds… ← original SMILES symbols
- **rings** : 각 고리의 bonds 중 하나를 끊어서 represent. 
ring의 존재는 broken bond의 2개 atom 각각에 정수를 추가하여 나타냄.
- **branch point**의 존재 : 왼쪽/오른쪽 소괄호에 의해 나타냄. → branch의 모든 원자들이 방문.

ex) 아스피린

만약 compound가 SMILES를 사용하여 1가지 방법 이상으로 표현되면 모호성 발생

→ **Normalization algorithm**을 사용하여 하나의 SMILES representation이 하나의 화합물 표현하도록 보장 [**unique/canonical SMILES**]

![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%203.png)

### SMILES feature matrix

- input compound → SMILES string으로 표현
- SMILES **string의 각 symbol** 또는 **‘feature vector’**(distributed representation of the symbol)이 계산됨.
    
    각 feature vector은 **42**개의 features(elements)로 구성. (21 features : atom symbols + 나머지 : original SMILES symbols)
    
    - 각 원자의 **21-dimensional vector**의 각 dimension : atom type, degree, charge, chirality
    - original SMILES symbols의 21-dimensional vector : one-hot vector(binary vector)
        
        single high(1) bit + all the others low(0)
        
- **feature matrix의 length** : 주어진 dataset(400)의 화합물의 최대 SMILES strings 길이로 설정
    
    feature matrix for SMILES strings : all the **blank parts were padded with 0** (→ retain input size)
    
- **Resulting** distributed representation : **2-dimensional feature matrix (400,42)** 고정 크기
- **Table 1**
    - numerical values(degree, charge, chirality) → RDKit 이용(2016.09.4)
        
        atomic substance quantities와 연관
        
    

### CNN

- input : SMILES string의 distributed representation으로 구성
(symbols를 표현하는 sequence of feature vectors 구성)
- Multiple layers : 2 convolutional layers + 2 pooling layers - average(global pooling layer - max이후)
    - 1st CL : SMILES feature matrices와 같은 width의 filters 사용(42)
        
        → convolution은 SMILES string의 방향으로만 수행
        
    - global pooling layer(MAX) : global ‘max’ pooling
        
        **Output : 64-dimensional vector(SCFP)**
        
- **Hyperparameters**(**Table2**) ⇒ **Bayesian optimization(GpyOpt)**에 의해 결정
    
    window size of filters, number of filters 
    
    ![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%204.png)
    
- SCFP를 fully connected layers의 input으로 사용 → prediction model
    - one hidden layer
    - trained by mini-batch stochastic gradient descent
    - Optimization : Adam(learning rate = 0.01)
    - 모든 weights : normal distribution에 의해 초기화(평균 : 0, 표준편차 : 0.01)
- CNN - Python(3.5.2) + Chainer(v1.24.0)

### SMILES convolution fingerprint (SCFP)

Computing a fingerprint 방법

CL에 의해 compute된 64-dimensional vector : SMILES feature matrix의 chemical structure 정보 포함하는 fingerprint

→ 이 vector가 SCFP

network가 train되었다면 training data뿐만 아니라 아무 compound에 대해서나 SCFP computing 가능!

SCFP를 ECFP와 같은 전통적 fingerprints 대체용으로 쓰기를 제안. 

⇒ **ECFP를 넘어선 SCFP의 장점** : **training에서 얻은 중요한 features represent** 가능

(예시 : network가 어떤 단백질의 ligands를 구분하기 위해 훈련되었다면 SCFP는 다른 화합물에서 얻은 ligands를 구별하기 위한 중요한 features를 represent할 것)

**ECFP**는 application context에 상관없이 features의 **고정된 타입** 고려

### Chemical motif

CNN의 또 다른 가치 : interpretability (SCFP에서 얻은 **features**를 input compound의 **substructures**로서 시각화할 수 있게 함.)

SCFP는 global max pooling에 의해 compute → SCFP의 차원 하나:  2nd CL의 filters 중 하나와 대응

각 **dimension** → input compound의 **substructure** 연결 : by tracing back

dimension의 large value → large contribution of 대응되는 filter → 연관된 substructure의 중요성

**“chemical motif”** : 연관된, 중요한 substructure

![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%205.png)

chemical motifs의 분석 : 네트워크에 의한 **예측 결과의 해석(설명. interpretation)** 기능 향상

SCFP의 각 dimension : ~~다른 value scale~~ 가질 수도 있음 → dimension을 large-contribution filters를 식별하는데 비교하기 어려움.

⇒ **SCFP를 정규화**하자! (train, predict에 쓰이지 않고 오직 **chemical motifs detection**에만 사용)

1. 주어진 데이터셋의 모든 compounds에 대한 SCFP computing
2. global max-pooling layer에서 값을 찾고 각 filter의 평균과 variance를 계산.
3. SCFP를 각 dimension에 대해 **Z-scores**로 transform(대응되는 filter의 평균, variance 사용)
    
    chemical motifs detection → SCFP dimensions with Z-socres(2.58 이상)에 주목
    

### Dataset & Performance evaluation

**TOX 21** dataset → CNN의 performance 평가

- TOX 21 Challenge 2014에서 생성(compound classification problems 해결 대회)
    
    → 이전 연구에서 benchmark dataset으로 사용.
    
- 8000 compounds - 12 단백질에 bind될 수도 있다는 정보
- **Table 3-4 (dataset)**
    
    각각 active/inactive compounds 포함 ← 특정 experimental assay에서 획득
    
    Train / Test(validation) / Score(final evaluation) 데이터 타입으로 나뉨.
    
    ![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%206.png)
    
    ![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%207.png)
    

Receiver operating characteristic curve(**ROC-AUC**) 아래 영역 → 모델의 performance 평가

- 주로 **classifiers**의 performance 평가. 측정
- 0~1 사이값. **높은** 값일수록 active/inactive 사이 더 **정확한** classification

## Results

### Cross validation

**Five-fold cross validation**을 이용 → CNN 훈련 및 평가 (computation time, memory usage, convergence speed와 같은 statistics 제공)

TOX21 데이터셋의 Train, Test, Score data를 합쳐서 **하나의** 데이터셋으로 만듦.

→ Five-fold cross validation을 수행

- validation 위해 **ROC-AUC**를 측정하는 동안 300 epochs까지 훈련시킴.
    
    평균적으로 **Giga bytes** memory와 함꼐 36sec/epoch의 훈련 필요
    
    ROC-AUC는 20 epochs 중심으로 수렴
    
    ![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%208.png)
    
- 우리 모델과 Compound classification 문제의 전통적인 방법 비교 (**ROC-AUC**)
    
    input : ECFP + logistic regression / ECFP + random forest / ECFP + DNN / graph convolution보다 뛰어남.
    

### Comparison with the winning model of TOX21 challenge 2014

Fully connected hidden layer이 **SCFP와 output layers 사이**에서 사용되도록 모델 만듦.

Hidden units의 수 → 1st, 2nd convolution layers의 filters의 수와 크기에 **optimize** **(Bayesian optimization, GpyOpt** 이용)

Train/Test data를 **hyperparameter** 결정에 사용 → **ROC-AUC를 ‘Score’**데이터를 이용해 평가.

- **DeepTox**와 모델 비교 : winner method
    
    DeepTox DNN model : hidden layers의 활성화 함수는 ReLU. final output에 sigmoid 함수. 
    mini-batch size는 512, L2 regularization과 dropout은 오버피팅 방지
    
    thousands of features 사용 - 2500 in-house toxicophores features로 구성(substructures 구성)
    
- **ROC-AUC**는 DNN이 ECFP만 사용했을 때보다 평균적으로 낫지만 **ECFP + ElNet을 제외한 ‘DeepTox features’보다는 낮다.**

### Chemical space analysis with SCFP

SCFP가 전통 fingerprints를 대체할 수 있다 → **Chemical space** analysis using **SCFP**

SR-MMP subdataset에서 모든 compounds의 SCFP 계산 → MDS(Multi-Dimensional Scaling)으로 차원 축소 수행

ECFP(length=1024, radius=2)를 사용한 유사 분석

> SCFP에 의해 만들어진 **chemical space** : 명확히 구분 ↔ ECFP
(SCFP의 expressive power : ECFP보다 강함 → SR-MMP subdatasets)
> 

![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%209.png)

SCFP의 **차원 수**(64) : ECFP 차원 수(1024)보다 훨씬 작다.

- ECFP는 high-dimensional vector로 represent되지만, fingerprints간의 **거리는** hash collision때문에 항상 compounds의 **유사성과** ~~비례하지 않는다.~~
- 반면, SCFP의 각 원소는 training으로부터 얻은 대응되는 **substructure**의 contribution 표현
    
    모델은 label classification 문제에 **크게 기여하는** substructure 추출
    

### Detection of chemical motifs

CNN의 **예측 정확도**가 state-of-the-art(최첨단)의 방법보다 뛰어나진 않지만 **chemical motif 형태의 learned feature representation 추출** 가능하다!

**NR-AR subdataset** → chemical motifs 분석

CNN에 active compounds 적용 + chemical motifs detection

motif analysis → chemical motifs를 NR-AR dataset의 중요한 substructures로서 해석(설명) 가능.

![Untitled](SCFP%20bdbb0cdf0b0e467d82cf99ca6f65ef7a/Untitled%2010.png)

## Discussion

chemical compound data를 분석하는 새로운 CNN 제안

- 전통적인 CNN이 image data를 다루는 방식과 유사하게 **SMILES-based feature matrix** 사용
- CNN에서 chemical motif 형태로 얻은 feature representation을 추출하는 방법 개발
- **chemical motifs**의 분석은 예측 결과의 해석을 용이하게 하여 화합물의 중요한 **substructure** 강조
    
    중요 substructure 지식 없이도 representation learning에 의해 motifs 자동 획득
    

CNN을 Classification model로 사용하면 five-fold cross validation 실험에서 존재하는 방법들보다 높은 정확성 (***ROC-AUC*** 참고)

- ECFP만 사용하는 DNN보다는 정확 / ECFP와 **DeepTox features**를 사용하는 모델보다는 덜 정확
- DeepTox features : classification model의 정확도를 향상시키는데 기여

⇒ representation 학습에 의해 자동으로 얻은 SCFP는 이전에 잘 사용된 ECFP의 성능 능가.

아직 handcrafted DeepTox features의 성능에 도달하지 못했음.

**SMILES feature matrices** : 각 atom + atom symbol을 대표하는 **one-hot vector**의 **structural** properties 포함

- one-hot vector가 머신러닝의 string data의 symbols를 대표하는 features로 자주 사용되지만 ~~논문에서는 사용X~~
    
    (atom의 property는 compound의 structural environment에 상당히 의존)
    
    예시) 탄소 원자의 특성은 benzene ring인지, 산소 원자에 결합되었는지에 따라 다름.
    
    원자의 종류 차이는 유사한 property를 가질 수도 있음 ← 같은 family(group of elements in the periodic table)에 속해있다면
    
- SMILES feature matrices : 원자의 structural properties를 사용하여 capture되도록 설계

SMILES convolution의 가치 : 미리 substructures를 input features로 특정지을 필요 없음.

- 중요한 substructure에 대한 사전 지식 없이도 chemical motifs를 representation learning에 의해 자동으로 획득
- SCFP의 크기가 작게 유지될 수 있음. (preferentially obtain) - 논문에서는 64
    
    ⇒ **ECFP**와 반대. (ECFP는 모든 가능한 substructures를 고려하여 large-sized vectors 필요 / hash collision때문에 제한된 expressive power)
    

Chemical motifs의 분석 → CNN은 steroid-like chemical motif 성공적으로 detect

- steroid-like chemical motif : androgen receptors의 결합을 위해 중요한 구조
- 다른 motifs : androgen receptors를 위한 novel skeleton structures를 위한 후보들로 생각할 수 있음.

⇒ classification method뿐만 아니라 drug disscovery에 단서들을 제공하는 수단

[Chemical motifs의 detection : **filters** 기반] ⇒ detectable chemical motifs의 **size는 window size of filters**에 의해 제한.

- maximum motif size : 2k1 + k2
- multiple filters는 특유의 overlapping substructures를 represent할 수 있음.
    
    overlapping substructures의 조합 → entire motif represent할 수 있음.
    

⇒ Large chemical motifs의 detection : filters의 결합 분석으로 가능할 수도!

TOX 21 dataset의 (active compounds - inactive compounds 개수) 간의 **imbalance** 문제

- learning rate가 positive data에 대해서만 상수곱 → positive data could be learned strongly
- active compounds만 non-canonical SMILES에도 described → positive examples 수 증가

2가지 방법 모두 accuracy 향상 실패

**⇒ imbalanced data sampling 기법**

SMOTE → ADASYN ——> CEGAN(GAN을 classification에 적용)

## Conclusions

**SMILES** linear notation of compounds에 기반한 **feature matrix** 설계 ⇒ **CNN**에 적용.

CNN : convolution operation이 SMILES string을 따라 한 방향으로만 수행

- SMILES string 기반 CNN의 성능 : chemical compounds의 virtual screening에 사용되는 conventional fingerprint method보다 뛰어남.

**학습된 filters를 사용한 motif detection** : substructures(protein-binding sites)뿐만 아니라 찾으려는 unknown functional groups의 substructures도 가능

TOX 21 Challenge를 벤치마크로서 사용 → 현재 winning model보다 좋은 성능

Multivariate analysis : SMILES 기반 representation learning에 의해 학습된 features로 구성된 chemical space는 compounds의 정확한 식별이 가능한 **rich feature space**를 적절하게 표현

---

### Reference

- **용어**
    - ECFP : Extended-Connectivity FingerPrint
    - ElNet : Elastic Net
    - MDS : Multi-dimensional scaling
    - RF : Random Forest
    - ROC-AUC : receiver operating characteristic curve 아래 부분
    - OpenSMILES
    - GPyOpt : A Bayesian Optimization Framework in Python