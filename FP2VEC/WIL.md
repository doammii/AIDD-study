# FP2VEC

[https://github.com/wsjeon92/FP2VEC](https://github.com/wsjeon92/FP2VEC)

**FP2VEC** : chemical compounds → set of trainable(task-specific, information-rich) embedding vectors로 represent

- new molecular featurizer
- 다른 vector embedding methods처럼 nonvectorial data → numerical vectors로 convert (Euclidean space)
- additional attractive features 가지고 있음.
- **장점** 3가지
    1. multitask learning tasks에 효율적
    2. consistently achieve competitive prediction accuracy even when trained on small datasets on the classification tasks
    3. simple and easy, training of the model is straightforward and fast

**CNN** : capture important(local) features of data - from the embedding vectors

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
    

The properties of chemical compounds 예측에 QSAR model 사용

QSAR prediction model using a simple CNN (sentence classification task에 성공적 사용)

critical to develop more effective new featurizers to fully realize the power of DL techniques

FP2VEC + CNN → QSAR tasks (classification)

## Materials and methods

### Benchmark featurizers and datasets

### Featurizer and QSAR model

## Results and discussion

### Classification tasks

### Regression tasks

### Analysis of FP2VEC and QSAR model

## Reference