# cMolGPT: A Conditional Generative Pre-Trained Transformer for Target-Specific De Novo Molecular Generation

### Abstract

Deep generative models → generation of novel compounds in small-molecule drug design

GPT-inspired model for de novo target-specific (protein) molecular design

implementing different **keys, values** - **multi-head attention conditional on a specific target**

drug-like & active compounds에 일치하는 SMILES strings 생성

→ closely match the chemical space of real target-specific molecules

## Introduction

unbounded search space 때문에 optimization task는 다루기 힘들다.

computational methods developed to search vast chemical space *in silico* and *in vitro*

produce new small molecules with **desired biological activity** - **desired locations** within the various  chemical search spaces

특정 화학 구조에 대응되는 SMILES strings로 표현된 drug-like molecules의 conditional generation

화학 구조는 제한된 SMILES strings dataset에서 RNN을 사용하여 sample 가능

- sample target-specific chemical structures by **fine-tuning RNN + small set of active SMILES strings**
- **modify RNN by setting the interval states** of RNN cells(LSTM) → specific target proteins
- transformer, seq2seq structure에 **modified SMILES**를 넣어 condition 아래의 SMILES 출력

conditional graph generative models : molecular graph 생성, 각 층에 hidden state로 conditional representation 항 추가

---

NLP task로서 small-molecule drug design 문제를 **text/SMILES generation** 문제로 보자!

GPT 구조는 conditional generation 지원 + **fine-tuning small-sized supervised data**

- GPT : autoregressive language model to produce human-like text + large unlabeled train data
- large text corpus로 **unsupervised learning** - 이전 단어들로 다음 단어 예측
- well-trained GPT는 synthetic text excerpts 생성 가능

target protein 같이 **predefined conditions**를 바탕으로 small-molecule drug design하는 것 중요

**→ conditional sequential generation + target protein + cMolGPT(auto-regressively)**

1. pre-train a Transformer-based auto-regressive decoder on MOSES dataset (target 정보X)
2. randomness into **sampling** process - more variations & creative
3. different embeddings(k,v) → generative process to be conditioned on the specificed targets
- **fine-tuning** base model on 3 target-specific datasets

generated sequence가 SMILES grammar에 맞는 valid drug-like structures라는 것 보장

Transformer structure의 specified target 맞춤 conditional generation 보장

→ de novo molecule design & molecular optimization cycle time 가속화

## Results and Discussion

generating compound libraries using a pre-trained base model of cMolGPT

- creating novel, diverse virtual compound libraries 능력 평가
- valid, unique, fragment similarity, SNN
- closely matched the real data distribution → generating drug-like molecules 가능성

generating **target-specific** molecules using conditional MolGPT

- representative of 초기 drug discovery
- same MOSES set & fine-tuning on the target set
- outperform cRNN model - valid compounds 생성뿐만 아니라 de novo design에도 좋은 성능
- distribution of predicted activity of compounds가 cRNN보다 3targets 모두에서 좋은(active) 성능
- visualizing chemical space
    - MinHash fingerprint vectors, Tree MAP(TMAP)을 사용하여 2D projection construct
    - generated & real 모두 same sub-chemical space
- evaluate quality of generated compounds
    - generating new active series compounds for target 가능
    - generate compounds with a wide range of physical-chemical properties

## Methods and Materials

### Datasets

MOSES molecular dataset(from Polykovskiy)

- perform unsupervised pre-training of the proposed cMolGPT + target-specific embedding initialized as zero
- ZINC clean Lead Collection에서 추출된 drug molecules → same train/test split
- target-specific molecular datasets : EGFR(1381), S1PR1(795), HTR1A target proteins(3485)

Problem

- Generative molecular design : valid, drug-like, novel chemical structures 집합 생성
- Conditional molecular design : binding affinity의 적절한 **tuning**
- target protein을 condition으로서 embed → target protein에 대해 active한 화합물 생성

Transformer → conditional generator

- RNN의 recurring nature가 병렬화 방해 → Transformer : long input sequence에 강한 **attention**을 사용한 sequential modeling
- encoder(sequence of tokens를 sequence of latent representations z로 변형)
decoder(한 번에 z의 한 원소를 conditioning한 output sequence 생성)
- **sampling sequence** : **auto-regressive generation**  - 이전에 생성된 모든 token 사용
    
    **token-wise generation**에 영향.
    
- attention : querying (k-v) pairs / **multi-head attention** : k,v,q를 h번 project, 병렬화
- pre-trained on a large-scale SMILES dataset → parametric probabilistic distribution 학습
- conditional generation은 target-specific embeddings를 MHA component에 제공하여 강화됨.

### Unsupervised generative pre-training

seq2seq model usually trained end-to-end with a large number of training pairs

Chemical space에서는 limited labeled data 때문에 **unsupervised** learning task 필요

→ **decoder-only generative** model : 이전에 생성된 token들로 다음 token 예측

GPT, GPT2는 뛰어난 language generation → transformer-based model을 small-molecule optimization & hit finding에 사용 (generating drug-like SMILES sequences)

auto-regressive generation → sampling은 이전에 나오지 않았던 variation 만들어내야 한다!

### Conditional Generative pre-trained transformer

decoder-only : pre-training동안 memorize drug-like structures

pre-defined condition → confine the search space & sample drug-like structures

- decoder learns a parametric probabilistic distribution - SMILES vocabulary space
+ target condition
    
    attention to embeddings of 이전에 생성된 tokens, target-specific embedding
    
    multiplying masked positions with negative infinity
    
- maintain the structural consistency + SMILES grammar(drug-like based on memorization)
- target information 통합
    
    → leverage MHA in Transformer decoder & target-specific embeddings to k,v of attention
    
    target-specific embedding을 key, value로 사용 (2nd MHA) 
    
    target-specific design은 decoder와 독립적 → condition embeddings(zero) 같은 세팅으로 제거 가능
    

### Workflow for training and sampling cMolGPT

decoder에서 이전에 생성된 token들은 MHA를 거쳐 encoder의 ‘기억’ 출력

- base model : **pre-train** a model on the MOSES dataset without target conditions
    - model :  Transformer-based auto-regressive decoder
    - drug-like structure 학습에만 집중
        - parametric probabilistic distribution over SMILES vocabulary space + SMILES grammar에 맞도록
    - target-specific embedding을 zero embeddings로 초기화 (target-specific info 제공X)
        
        → generator가 decoder의 MHA에 전달
        
- **fine-tuning** with different target-specific-embeddings
    - 3 target-specific datasets의 <compound, target> pairs 이용
    - target-specific embedding 초기화하여 attention layer에 전달
- **auto-regressively sampling** tokens로 drug-like structure 생성 → desired  target(property) 반영
    - 훈련된 decoder에서 sampling tokens 얻음.
    - randomness into the **sampling** process → make model creative and more variations
        - sampling 과정 중 NLL loss 측정

**Likelihood of Molecular sequential generation**

- **NLL loss**(Negative Log-Likelihood) 사용 - **likelihood estimation of SMILES sampling**
- forcing values & keys of multi-head attention → generative condition(c)

**ML-based QSAR model for Active scoring**

- regression-based QSAR model for each target
- ExCAPE-DB에서 얻은 molecular datasets (active compounds → training target-specific generative models에 사용)
- LightGBM model to predict activity + molecular features(FCFP6 fingerprint, MACCSkeys, Molecular descriptors)

## Conclusions

Transformer-based random molecular generator → generate drug-like structures

target-specific molecular generator 위해 target-specific(protein) embedding을 transformer decoder에 제공

→ 3 target-biased datasets to evaluate(EGFR, HTR1A, S1PR1)

model에서 얻은 sampled compounds가 더 active

visualize chemical space → 원래 sub-chemical space 확장