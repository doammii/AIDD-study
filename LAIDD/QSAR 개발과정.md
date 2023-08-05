# QSAR 개발과정

- **Reference**
    
    [QSAR 모델 개발 과정](https://velog.io/@ssumannb/QSAR-모델-개발-과정)
    

### QSAR란?

**정의** : Quantitative structure–activity relationships

Construction of a mathematical model relating a molecular structure to a chemical property or biological effect by means of statistical techniques

**목적** : Lead-compound 찾기 + Optimization

**virtual screening**(컴퓨터가 대신 HTS. **컴퓨터로 target에 가장 강하게 binding하는 hit compound 탐색**) 방법 중 하나.

**Fingerprint**-based / machine learning algorithm

- **fingerprint는 분자들간 구조의 유사도(Tanimoto coefficient)를 빠르게 측정**하기 위해 사용
- 2D diagram을 직접 비교하지 않고, **분자 구조의 특징을 뽑아내어** 이를 통해 비교하는 것
- Molecular fingerprint : **bit string** representations of molecular structure & properties
- **2D structure features**는 전형적으로 **이진값 벡터**들로 encode됨.

### QSAR 모델링 과정

1. **dataset** 준비 (화합물 + activity)
2. 수치/벡터화
    
    X : molecular **descriptors (화합물 데이터)**
    
    Y : response variable. acivity data. (실수값 또는 y/n)
    
3. 정량적 함수 / 통계 모델 구하기 (statistical analysis)
4. Validation

### Components/Dataset

- 화합물 데이터
- Activity data
    - 관찰된 activities
    - experimental observation 형태 → numerical(regression) 또는 categorical labels(active 여부. soluble 여부.)
- Statistical modeling method
    - identify the key relationships (molecular descriptors - activities)
    - linear regression, SVM, Random forest, Deep learning

### Assessing model performance

- regression : MSE / Spearman Rank Correlation(순서에만 관심있다면)
- classification : accuracy / precision / recall / F1 score / AUC

---

model descriptor

주로 2D descriptor (2D-QSAR) 이용

**p.25 ~ Fragment codes / keys / Molecular Fingerprint**

### PaDEL Descriptor

software to calculate molecular descriptors and fingerprints

convert : SMILES / Mol file → PaDEL descriptor