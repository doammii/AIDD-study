DL-based toxicity prediction models

공통점 : DeepTox, MOL2VEC보다 좋은 성능 / CNN model로서 DL의 자동화된 feature 추출 능력(**representation learning**) / **Zero-padding** 사용

- zero-padding 사용 효과 : CNN model의 input을 고정 크기로 만들 수 있다. encoding 중 화학적 특성을 유지하면서 minority data를 shifting과 함께 augment 가능.
- zero-padding을 위한 encoding 방법
    - SCFP는 SMILES symbols의 feature matrix로. (단, SMILES strings의 최대 길이보다 symbols의 수가 작을 때 padding)
    - FP2VEC은 Morgan fingerprint ‘1’로 look-up table에 매치된 randomized fingerprint embedding vector로

### CNN

**multi-task learning** 환경에서 sharing multiple tasks를 통해 feature 학습 도움.

- shared hidden layers에서, multiple tasks 사이에 여러 features 공유 → multiple task resemblance

**CNN-based classifiers** : SMILES feature matrix와 Fingerprint embedding vector로부터 자동으로 features 추출(capture)(**representation learning**) → active/inactive로 분류

- CNN에 의한 representation learning : 자동으로 input features의 저차원 representation 획득 가능
- CNN 학습에서 kernel(**filter**) : feature map에서의 **chemical motif**(functional substructure. learned feature representation) 추출 위해 분석
- SCFP : motif는 **resampling**에 상관없이 올바르게 추출
- FP2VEC : **부분적으로** motif 발견 (**Fingerprint** 기반 DL 모델이기 때문)
    - train 결과나 feature map을 해석할 때 추가 변환 과정 필요
    - fingerprints로 변환되는 특정 substructures의 search space가 제한되거나 무시됨 (다른 원자 환경이 같은 bit로 매핑)

### SCFP

각 SMILES sequence의 character가 42-bit vectors로 represent(one-hot encoding) → SMILES feature matrix라는 분산된 representation 계산

- **feature matrix** : CNN input. SMILE character 각각의 one-hot encoding(custom word embedding) vector(42 bit) 집합을 축적함으로써 feature 연산
    
    atom의 **structural properties**를 사용하여 capture되도록 설계
    
    - character 각각을 custom word embedding → activation이 잘 되는 sequence ⇒ ‘**structure-alert**(SMILES가 활성화된 부분 찾기)’
        
        구조에 대한 이해가 힘들어서 **각 character sorting** 작업 필요
        
    - atom symbol + SMILE symbol
    - CNN input 크기는 고정되야 하므로 feature matrix의 고정된 크기를 유지하고자 **zero-padding** 추가됨. (단, SMILES strings의 최대 길이보다 symbols의 수가 작을 때 padding)
- **1D CNN**에서 **learned filters**에 의해 발견 → chemical compounds의 **classification + chemical known motifs(structures)의 추출**
- **CNN-based classifier :** 2 conv layers + 1 FC layer로 구성
    - 1st : SMILES symbols의 feature 추출
    - 2nd : SMILES sequence의 feature 추출
    - maxpooling 결과 **SCFP 생성**
    - SCFP를 input으로 한 FC layer가 active 여부 분류
- 장점 : improved prediction accuracy as well as the motif detection (by conv layers에서 만들어진 feature maps 분석)

### FP2VEC

fingerprint vectors 중에서 selected ECFP(key fingerprint)가 lookup table의 매치된 fix-sized randomized vector로 embedding됨 - encoding 과정

- set of trainable embedding vectors = molecular featurizer = FP2VEC
    
    FP2VEC 역할 : chemical compounds → set of trainable(task-specific, information-rich) **numerical, embedding vectors**로 represent
    
- **Fingerprint embedding featurizer**
    
    **chemical compounds ⇒ molecular substructures(fingerprints) ⇒ 각각 fingerpint embedding vector로 encoding**
    
    molecular fingerprint를 사용하여 molecular substructures → integer numbers
    
    - **<Fingerprint embedding 과정 : FP2VEC featurizer(vector) generation>**
        - SMILES로부터 **molecular substructures(fingerprint vector, fingerprint)** 추출
            
            generate the **1024** bit **Morgan (or circular) fingerprint** of a radius of 2 by using the RDKit.
            
        - Collect **fingerprint indices** in the fingerprint vector.
        - Express the features of the molecular structure as **a list of integers**
            
            each integer → a specific molecular substructure → word indices of texts
            
        - Build **lookup table** to represent **each integer index as a vector of finite size**(embedding size).
            - The lookup table : a two-dimensional matrix (**size =** bit size * embedding size)
                
                Each **row** : unique embedding vector corresponding to each integer of the Morgan fingerprint. 
                
            - **Training** process : the values of the lookup table are fine-tuned to maximize the specific objective of the training
                
                ⇒ **Task-specific vector representation of compounds(the fingerprint embedding matrix) 획득**
                
            - **Test** process : fingerprint embedding matrix를 예측에 사용
    
    화합물마다 Morgan fp의 수가 다르므로 molecular encoding size도 다양 → **zero-padding** 적용
    
- **CNN-based classifier** : 1 conv layer(Conv2d) + 1 FC layer로 구성
    
    **2차원 Convolution layers(Conv2d) +** Max pooling layer + Dropout layer **+ FC layer**
    
    multi-task learning을 사용하는 CNN-based classifier 발견
    
- **multi-task learning :** conv layer가 multiple dataset과 공유됨.
    
    **Single CNN model + Separated FC layer**로 예측 정확도 향상
    
    **1개의 CNN model만** 가짐 → Target들은 CNN model architecture의 **parameters 공유** → chemical compounds의 **general features capture** 가능!
    
- Hyperparameter 1개 **:** embedding size **k**
- **장점**
    1. **multi-task** learning tasks에 효율적
    2. **dataset의 size**에 상관없이 competitive prediction **accuracy** 일관되게 유지
    3. simple and easy + training of the model is straightforward and fast