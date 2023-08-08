# FP2VEC

- 참고 github repository
[https://github.com/wsjeon92/FP2VEC](https://github.com/wsjeon92/FP2VEC)

**FP2VEC** : chemical compounds → set of trainable(task-specific, information-rich) embedding vectors로 represent

- new molecular featurizer
- 다른 vector embedding methods처럼 nonvectorial data → numerical vectors로 변환 (Euclidean space)
- additional attractive features 가지고 있음.
- **장점** 3가지
    1. multitask learning tasks에 효율적
    2. consistently achieve competitive prediction accuracy even when trained on small datasets on the classification tasks
    3. simple and easy, training of the model is straightforward and fast

**CNN** : capture important(local) features of data ← from embedding vectors

- grid-like data뿐만 아니라 sequential data(sentences, DNA sequences) 분석에도 효율적
- model graph-like structures → “graph convolution featurizer”
- simple CNN architecture의 QSAR model → NLP classification tasks에도 사용.
    
    ⇒ capture the important features from the embedding vectors

---

### **Introduction**

- **QSAR models**
    
    principle : structurally similar chemicals should have similar properties
    
    - vital role & appplication
        - drug discovery (특히 lead compound generation by VS)
        - drug’s ADME property optimization
        - computational toxicity prediction
    
    prediction accuracy  : DL technology 의해 향상 - QSAR ML challenge(by Merck)에서 Hinton’s group 우승하며 주목
    
    - **DL models + DeepChem**
        - random forest method - set of large diverse QSAR datasets
        - boosting docking-based virtual screening with DL
        - multitask DNN in QSAR studies (Xu, 2017)
        - generating focused chemical libraries using a RNN (Segler, 2018)
        - de novo generation of new molecules using generative models (Kadurin, 2017; Sanchez-Lengeling and Aspuru-Guzik, 2018)
- **NLP**
    
    words→ numerical values(ex. n-dimensional vectors)
    
    - **Word2Vec** model(Mikolov, 2013) : compute the semantic relationship between words (represented by trainable vectors)
    
    biological data에 적용
    
    express biological data in numerical form & calculate the semantic meaning of the data
    
    - **Mol2vec**(Jaeger, 2018) : Word2Vec model을 molecular structure information로부터 molecular properties 예측
        
        molecular structure → vector representation (molecular fingerprint vector가 molecular substructures list 사용하는 것과 유사)
        
    - **SMILES2VEC** model(Goh, 2017) : direct conversion [SMILES representation → embedding vectors]
- **CNN**
    
    convolution operation으로 capture local features
    
    grid-like data뿐만 아니라 sequential data(sentences, DNA sequences)에도 적용 가능 (***Alipanahi 2015, Collobert 2011, Kalchbrenner 2014, Shen 2014, Yih 2014***)
    can be extended to model graph-like structures(ex. protein-protein interaction networks)
    chemical structure can be represented as a graph (**graph convolution featurizer**) (***Duvenaud 2015, Kearnes 2016***)
    Several featurizers based on graph convolutions → **ECFP** feature을 classification, regression tasks에서 outperform
    

> The properties of chemical compounds 예측에 QSAR model 사용
> QSAR prediction model using a simple CNN (sentence classification task에 성공적 사용)
>   critical to develop more effective new featurizers to fully realize the power of DL techniques
> FP2VEC + CNN → QSAR tasks (classification)

---
## Materials and methods

### Benchmark featurizers and datasets

**FP2VEC의 성능을 평가**하기 위해 우리 QSAR model을 다른 Molecular featurizers를 사용한 model들과 **benchmark 결과 비교**

**→  ECFP / Graph Convolution / Weave featurizer** (MoleculeNet & Graph Convolution study)

**Classification** : MoleculeNet의 prediction models

- FCNN : ECFP featurizer 사용
- Bypass (Bypass multitask network model) : ECFP featurizer 사용
- GC (Graph Convolution) : GC featurizer 사용
- Weave : Weave featurizer 사용

**Regression** : MoleculeNet & Graph Convolution study의 models

- MoleculeNet : FCNN / GC / Weave models
- Graph Convolution study
    - GraphConv : 2 GC featurizer models + linear/neural network
    - ECFP : 2 ECFP models + linear/neural network

---

Datasets

- Classification : Tox21 / HIV / BBBP / SIDER
- Regression : Malaria / CEP / ESOL / FreeSolv / Lipophilicity
- 모든 chemical compounds는 **SMILES codes**로 표현.
- **train : validation : test = 8 : 1 : 1** (benchmark studies와 똑같이 준비됨)
- 과정 : train → optimization by choosing the model hyperparameters(validation sets 고려) → test sets의 optimized models에 대해 예측 성능 측정
    
    **각각의 task 5번 반복 + 평균과 표준 편차** 기록
    

### ★★★ Featurizer and QSAR model ★★★

- **Fingerprint embedding featurizer**
    - text(sentence or document) ⇒ words ⇒ 각각 numerical **vector**(word embedding matrix)
    - **chemical compounds ⇒ molecular substructures(fingerprints) ⇒ 각각 fingerpint embedding vector**
    
    NLP의 text representation뿐만 아니라 CNN같은 기술 → utilized to 다양한 분자 특성 예측
    
    ![Untitled](https://github.com/doammii/CADD-study/assets/100724454/895f8bf2-3d2c-4a39-b893-ea7a71c920cb)
    
    ---
    
- First, we extract the **molecular substructures** from a SMILES representation of a chemical compound.
    
    generate the **1024** bit **Morgan (or circular) fingerprint** of a radius of 2 by using the RDKit.
    
    (2048 bit or full-size (‘unfolded’) fingerprints도 시도해봤지만 size of the fingerprint vectors는 모델 성능에 영향X)
    
- After that, we collect the **fingerprint indices** in the fingerprint vector.
- Then express the features of the molecular structure as **a list of integers**
    
    each integer → a specific molecular substructure → word indices of texts
    
- **Build lookup table** to represent **each integer index as a vector of finite size**(embedding size).
    - The lookup table : a two-dimensional matrix (**size =** bit size * embedding size)
        
        Each **row** : unique embedding vector corresponding to each integer of the Morgan fingerprint. 
        
        Initial state with random values. 
        
    - **Training** process : the values of the lookup table are fine-tuned to maximize the specific objective of the training.
        
        예시 : 그림에서 the embedding vector for fingerprint 2 is initialized with random values such as [0.2, 0.5,0.1, . . ., 0.5]. 
        이러한 random fingerprint embedding vectors →  a particular QSAR model → optimized through the training process by adjusting their values to maximize 모델 예측 정확도
        이러한 변화들은 lookup table에도 반영.
        
        ⇒ **Task-specific vector representation of compounds(the fingerprint embedding matrix) 획득**
        
        장점 : conventional circular fingerprint 자체보다 더 유용한 정보 제공
        
    - **Test** process : fingerprint embedding matrix를 예측에 사용
- **Structure of the QSAR model using a simple CNN architecture**

![Untitled 1](https://github.com/doammii/CADD-study/assets/100724454/51bce1ba-0cac-4c7f-8b2e-13ff5633084e)

**2차원 Convolution layers(Conv2d) + Max pooling layer + Dropout layer + FC layer**

- **Before training**
    - SMILES representation → fingerprint embedding matrix
    - x 가 **R의 lk 차원**에 존재 : x는 fingrprint embedding matrix
        - l : fingerprint vecotr의 1bits 수
        - k : embedding size
    - **mini-batch** operation : fingerprint embedding matrix를 fingerprint dimension을 따라 0으로 padding + m (max length)
        - padded fingerprint embedding matrices는 모두 같은 size ⇒ x(pad)가 R의 mk 차원에 존재
        
        > padded fingerprint embedding matrix **x(pad)가 모델의 input data로**!
        > 
- **Conv2d layer**
    
    ![Untitled 2](https://github.com/doammii/CADD-study/assets/100724454/608eb4c7-b986-4b6a-948f-44e073d6c984)
    
    **: convolution 연산자*
    
    w(conv)가 R의 hk 차원에 포함 : **각 filter w(conv)** + **window size(h*k)**
    
    **convolution filter**는 **substructure** 차원만을 따라 움직임. (~~embedding dimension~~)
    
    → convolution 연산은 n개의 필터들에 의해 n개의 feature maps 생성 (c는 R의 (m-h+1,n) 차원 안에 포함됨. → **전체 feature map** 표현)
    
- **Add bias** to the feature maps + **apply ReLU function** for the non-linear activation
    
    ![Untitled 3](https://github.com/doammii/CADD-study/assets/100724454/d412eb9b-d1eb-486e-a872-28d70c495373)
    
- **Max pooling layer**
    - max-over-time pooling operation : feature maps로부터 중요 features 추출
    - MP layer : c(relu)로부터 최댓값 pick up
    
    ![Untitled 4](https://github.com/doammii/CADD-study/assets/100724454/9581075e-7951-427f-ad13-784ba20099d8)

    
- **Dropout** (dropout rate of **0.5 → prevent overfitting** during a training session)
    
    ![Untitled 5](https://github.com/doammii/CADD-study/assets/100724454/52d3dd3a-a6d8-46de-9dbd-5968ea4f96ce)
    

**Evaluation** session : the ~~dropout~~ is not applied to cmax

→ n개의 c(drop)를 1차원 행렬로 concatenate → FC layer이 모델 출력 산출

→ **모델 출력** : QSAR model의 예측 (w(fc) : FC layer의 가중치 & b(fc) : FC layer의 bias)

![Untitled 6](https://github.com/doammii/CADD-study/assets/100724454/71527d31-3f6d-4a8e-9c1f-1795cf6c256d)

**Training** session : lookup table의 값 & Network parameters 조정

**Test** session : 예측을 위해 trained values 사용

---

- **Output & evaluation**
    - Classification
        - ouput : **sigmoid** activation function
        - optimization : Logarithmic loss function & **Adam** optimizer
        - evaluation : **ROC-AUC** scores (5번의 독립 시행에 의해 얻은 ROC-AUC 값들의 **평균**)
    - Regression
        - optimization : mean-sqaured error loss function & **Adam** optimizer
        - evaluation : **RMSE** scores (5번의 독립 시행에 의해 얻은 RMSE 값들의 **평균**)
- **Hyperparameters**
    - **FP2VEC : 1개** (embedding size **k**)
    - **QSAR model : 2개** (filter의 window size **h** & feature map의 size **n**)
    
    k = 100,200,300… / h = 1,2,3,,,7 / n = 512,1024,2048 / learning rate = 1e-3,5e-4,1e-4,,,
    
    모델의 성능은 이러한 hyperparameters의 변화에 그렇게 민감하지 않다.
    
- **Multi-task learning**
    - Tox21 & SIDER : 1 compound ↔ multiple targets
    - single task learning : 각 target은 **분리된 CNN model** 가짐 → 각 CNN model은 **다른 input data에 의해 train**됨.
        
        (Tox21같은 경우 12개의 개별 CNN models for 12 다른 targets)
        
    
    multi task learning : **Single CNN model + Separated FC layer**로 예측 정확도 향상
    
    - 모든 targets에 대해 **1개의 CNN model만** 가짐.
    - Target들은 CNN model architecture의 **parameters 공유**! → chemical compounds의 **general features capture** 가능!
    - 각 Target에 대해 분리된 FC layers → 각 Target에 대해 **specific features** 학습
    
    ![Untitled 7](https://github.com/doammii/CADD-study/assets/100724454/ffe431f0-af6a-44c5-87d5-cdeb9552c0ae)
    

## Results and discussion

FP2VEC featurizer을 사용하는 QSAR model의 예측 성능 ↔ 다른 모델들과 비교

### Classification tasks

비교군 From MoleculeNet

- FCNN / Bypass model + ECFP featurizer
    
    GC model + GC featurizer
    
    Weave model + Weave featurizer
    

> **결론** : GC & Weave는 BBBP, SIDER dataset처럼 상대적으로 작은 크기의 dataset에 안 좋은 성능.
↔ **FP2VEC** **featurizer**은 dataset의 size에 상관없이 reliable 성능 일관적.
> 

**Multiple target datasets(Tox21, SIDER) → Multitask learning**

Test sets의 **ROC-AUC scores** 측정 → QSAR model의 **예측 정확도** 평가

ROC-AUC scores의 **평균 & 표준 편차**는 **5번**의 독립 시행으로 측정

- Tox21 + SIDER : random split method
    - multiple targets → multi task learning methods
    
    > single task learning보다 [FP2VEC + **multi task learning**]이 Tox21&SIDER datasets에 대해 훨씬 높은 ROC-AUC scores 결과 보여줌.
    (ROC-AUC scores of the individual targets in the multitask model 증가)
    > 
    
    PotentialNet과 같은 최근 featurizer보다 높은 ROC-AUC score 보여줌. → FP2VEC이 multitask learning tasks에 매우 효율적!
    
- HIV + BBBP : scaffold split method
    - one target → single task learning model
    - HIV dataset : 2nd-best ROC-AUC score (1st : GC)
        
        BBBP : highest ROC-AUC score
        
    
    > **Scaffold split method: training/validation/test sets가 2차원 분자 구조로 type이 다르기 때문에 더 복잡하다.**
    structural differences 증가 → more difficult evaluation setting
    > 
    
    ⇒ **분자 구조에 따른 characteristics가 다른데도 FP2VEC 방법은 화합물의 general feature 학습 가능!**
    

![Untitled 8](https://github.com/doammii/CADD-study/assets/100724454/233b83e3-58c0-44b1-a01a-7301d5a9f714)

### Regression tasks

비교군

- (Table2) ESOL/FreeSolv/Lipophilicity + **MoleculeNet**(FCNN/GC/Weave model)
    - ECFP featurizer + FCNN model에 비해서 나음.
    - GC & Weave model에 비해 비슷하거나 나쁜 결과 - **graph-based model**이 더 나은 성능
- (Table3) Malaria/CEP + **graph convolution** with NN/linear methods(GC featurizer.GraphConv/ECFP featurizer)
    - 2가지 dataset에 비교해보면 가장 낮은 RMSE score. GC model보다 나음.

Test sets에 대해 **RMSE scores 측정**해 모델의 **예측 정확도** 측정 * **5번 독립 시행**

**random** split method에 의해 평가

> **결론** : FP2VEC의 성능이 classification tasks만큼은 아니어도 regression tasks에서 comparative performance 보여줌!
> 

![Untitled 9](https://github.com/doammii/CADD-study/assets/100724454/1caa4cc8-3e65-4d74-9ab9-6bfa8bc3f6a1)

![Untitled 10](https://github.com/doammii/CADD-study/assets/100724454/7809dd52-da26-4868-99af-a3baf83babcf)

### ★ Analysis of FP2VEC and QSAR model ★

- **[Featurizer 차이]** **Circular** fingerprint featurizer과 비교
    - circular fingerprint을 그 자체로 CNN-based QSAR model의 **input feature**로 테스트
    - Tox21 dataset에 대해 circular fingerprint model 평가 + 같은 CNN-based QSAR model 사용
    - 결과 : circular fingerprint model의 예측 결과가 FP2VEC에 비해 한참 나쁨. (ROC-AUC score↓)
    
    **Circular fingerprint vectors의 문제점 : “sparse” → convolution filters의 작은 window size는 molecular features를 적절히 capture 불가능**
    
    FP2VEC featurizer은 raw circular fingerprint에 비해 QSAR tasks 예측 성능 향상시킴.
    
- QSAR model을 위한 **CNN architecture 분석**
    - convolution filters with various window sizes는 neighboring molecular substructures(→ fingerprint vectors)를 detect하기 위해 디자인됨.
    - 1~7 window sizes 테스트 → **filter의 window size** 최적화 목적
        - regression : window size of 1 (fingerprint vectors를 나타내는 substructure의 임의의 ordering, neighboring 사이에 분명한 관계 없음.)
        - classification : window size of 5
    
    > 특정 분자 특성에 중요한 특정 substructure은 FP2VEC에 가까이 위치할 수도! (유한한 convolution filter size에 의해 captured)
    > 
- **[Fingerprint 차이] MACCS**-based FP2VEC과 비교
    - **MACCS** : 166 bit-size keys를 가진 fingerprint. 각 key들은 특정 분자 구조 표현. **RDKit**에 의해 구현
    - **FP2VEC algorithm : molecular fingerprint를 사용하여 molecular substructures → integer numbers**
        
        Circular fingerprint 대신 MACCS를 FP2VEC algorithm에서 사용해보자!
        
    - classification : circular fingerprint-based FP2VEC 모델에 비해 약간 낮은 예측 정확도
        
        regression : 몇몇 tasks는 circular fingerprint-based FP2VEC 모델보다 나은 결과
        
        (FreeSolv에서 최고, ESOL&Lipophilicity에서는 circular보다 약간 나음, Malaria&CEP에서는 감소)
        
    
    > **결론** : FP2VEC algorithm은 circular fingerprint뿐만 아니라 MACCS와 같은 fingerprint type에도 적용 가능!
    > 

## Reference

- ProtVec : Asgari,E. and Mofrad,M.R. (2015) Continuous distributed representation of biological sequences for deep proteomics and genomics. PLoS One, 10, e0141287
- Mol2vec : Jaeger,S. et al. (2018) Mol2vec: unsupervised machine learning approach with chemical intuition. J. Chem. Inf. Model., 58, 27–35
- SMILES2VEC : Goh,G.B. et al. (2017) SMILES2Vec: an interpretable general-purpose deep neural network for predicting chemical properties. arXiv: 1712.02034 [[stat.ML](http://stat.ml/)]
- PotentialNet

CNN 도입

- Alipanahi,B. et al. (2015) Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning. Nat. Biotechnol., 33, 831–838.
- Kim,Y. (2014) Convolutional neural networks for sentence classification. arXiv: 1408.5882 [[cs.CL](http://cs.cl/)].

FP2EC 도입

- Cadeddu,A. et al. (2014) Organic chemistry as a language and the implications of chemical linguistics for structural and retrosynthetic analyses. Angew. Chem. Int. Ed. Engl., 53, 8108–8112.