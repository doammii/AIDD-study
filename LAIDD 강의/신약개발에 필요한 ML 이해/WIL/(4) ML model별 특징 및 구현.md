## **ML model 특징**

- 지도학습(예측모델) : 선형 계열, 신경망, 트리 계열, 기타
- 비지도학습(데이터 처리) : 클러스터링, 데이터 변환, 차원 축소

[다시 살펴보는 머신러닝 주요 개념 3편 - 주요 머신러닝 모델 - 골든래빗 (goldenrabbit.co.kr)](https://goldenrabbit.co.kr/2023/08/09/%eb%8b%a4%ec%8b%9c-%ec%82%b4%ed%8e%b4%eb%b3%b4%eb%8a%94-%eb%a8%b8%ec%8b%a0%eb%9f%ac%eb%8b%9d-%ec%a3%bc%ec%9a%94-%ea%b0%9c%eb%85%90-3%ed%8e%b8-%ec%a3%bc%ec%9a%94-%eb%a8%b8%ec%8b%a0%eb%9f%ac%eb%8b%9d/)

| 머신러닝 유형 | 알고리즘 | 특징 |
| --- | --- | --- |
| 선형 계열 | 선형 모델, SVM, 로지스틱회귀 | 곱셈, 덧셈으로 score 계산 → 분류 및 회귀 예측 |
| 신경망 | MLP, CNN, RNN, Transformer | Matrix 연산을 기반으로 scores 계산, 활성화 함수 도입 |
| 트리 계열 | decision tree, random forest, gradient boosting | T/F 선택을 반복하여 예측 수행
Scaling 불필요 |
| 기타 | kNN, Bayes | 특성 공간상의 거리를 기준, 또는 조건부 확률을 기준으로 예측 |
| 클러스터링 | k-means, DBSCAN | 특성 공간상 거리와 유사도를 기준으로 sample grooping |
| 데이터 변환 | Scaling, 로그 변환, 카테고리 인코딩 | 효과적인 데이터 전처리 |
| 차원 축소 | PCA, t-SNE | 계산량과 모델 성능 향상, 의미 있는 시각 |

### Feature-Enginnering

- Column을 추가하면 성능이 좋아지지 않을까? → 분자 특성 추가 [rdkit.Chem.Descriptors](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.rdkit.org%2Fdocs%2Fsource%2Frdkit.Chem.Descriptors.html)

## 데이터 준비

소분자 유기화합물의 **lipophilicity(log10P)** 데이터

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn import tree

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 데이터 다운로드 (컬럼명을 'smiles'와 'logP'로 지정)
logP_data = pd.read_csv('https://raw.githubusercontent.com/StillWork/data/master/logP_dataset.csv', 
            names=['smiles', 'logP'])

# csv 파일로 저장
logP_data.to_csv('logP.csv')

# 내용 보기
print(logP_data.shape)
logP_data[:3]
```

### SMILES로부터 Mol 객체 얻기

```python
# 사본 데이터프레임 사용
df = logP_data.copy()

df['mol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x)) 

# 다른 방법
# df['mol'] = [Chem.MolFromSmiles(x) for x in df['smiles']] 
print(type(df['mol'][0]))
```

### 시각화

```python
# rdkit.Chem.Draw를 사용하여 mol 객체를 시각화할 수 있다.

# 16개 이미지를 그리드 형태로 그리기
mols = df['mol'][:16]
Draw.MolsToGridImage(mols, molsPerRow=4, useSVG=True, legends=list(df['smiles'][:16].values))
```

### 분자 정보 보기

```python
df['mol'] = df['mol'].apply(lambda x: Chem.AddHs(x))
df['num_of_atoms'] = df['mol'].apply(lambda x: x.GetNumAtoms())
df['num_of_heavy_atoms'] = df['mol'].apply(lambda x: x.GetNumHeavyAtoms())
```

### 특성 추가

```python
# 탄소 패턴을 지정한다
c_patt = Chem.MolFromSmiles('C')

# 이 패턴이 들어있는 곳을 찾는다. 패턴의 수를 세면 탄소 원자가 몇개 들어있는지 알 수 있다
print(df['mol'][0].GetSubstructMatches(c_patt))

# 임의의 패턴(원자)를 몇개 포함하고 있는지를 얻는 함수
def number_of_atoms(atom_list, df):
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['mol'].apply(lambda x: len(x.GetSubstructMatches(Chem.MolFromSmiles(i))))

number_of_atoms(['C','O', 'N', 'Cl'], df)
```

## 회귀 모델

### ★ 훈련/검증 데이터 나누기

R-squared / MSE

```python
# 특성 컬럼을 선택하여 X를 만들고 목적변수를 정의한다
train_df = df.drop(columns=['smiles', 'mol', 'logP'])
y = df['logP'].values

print(train_df.columns)

# 훈련과 검증 데이터를 나눈다 (검증 데이터로 10% 할당. shuffle=False 옵션 지정하면 순서 지키면서 split)
# random split 가정
X_train, X_test, y_train, y_test = **train_test_split**(train_df, y, test_size=.1)

# 회귀 모델 성능 지표 확인 함수 -> N 조
def show_reg_result(y_test, y_pred, N=50):
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    R2 = r2_score(y_test, y_pred)
    max_err = np.abs(y_test - y_pred).max()
    
    print('R2:', round(R2,4))
    print('MAE:', round(mae, 4))
    print('RMSE:', round(rmse,4))
    print('Max error:', round(max_err, 4))

    # 일부 실제값과 예측값 샘플을 plot으로 비교하여 그려본다 (N 개)
    
    if N > 0:
      plt.figure(figsize=(10, 6))
      plt.plot(y_pred[:N], ".b-", label="prediction", linewidth=1.0)
      plt.plot(y_test[:N], '.r-', label="actual", linewidth=1.0)
      plt.legend()
      plt.ylabel('logP')
      plt.show()
```

### ★ 선형 모델

**weight**를 보고 +, - 요소 추측 가능 → 선형 관계일 때 유용

```python
# 모델 학습 및 성능 평가
lin = LinearRegression() # y = ax + b
lin.fit(X_train, y_train) # 훈련
y_pred = lin.predict(X_test) # logP의 예측값
show_reg_result(y_test, y_pred)

'''
R2: 0.6569
MAE: 0.5691
RMSE: 0.7408
Max error: 4.839
'''

**# 선형 모델 가중치(분자 특성 추가 후)**
# 선형 모델 가중치를 보는 함수 정의

def plot_feature_weight(feature, weight):
    # plt.figure(figsize=(5,8)) # 특성수가 많은 경우
    W = pd.DataFrame({'feature':feature,'weight':weight})
    W.sort_values('weight', inplace=True)
    plt.barh(W.feature, W.weight)

plot_feature_weight(train_df.columns, lin.coef_)
```

### 회귀 모델

- 회귀 모델의 성능을 평가하는 척도로 기본적으로 R-Squared를 사용하며 MAE, RMSE 등을 참고.
    - MAE: mean absolute error
    - MSE: mean squared error
    - RMSE: root MSE

```markdown
$MAE = $$1\over{N}$$ \sum |y_{i} - \hat{y}|$

$MSE = $$1\over{N}$$ \sum (y_{i} - \hat{y})^{2}$

$RMSE = \sqrt{MSE}$

$R^{2} = 1$-$\sum(y_{i}-\hat{y})^{2}\over{\sum(y_{i}-\bar{y})^{2}}$ $= 1$-$MSE\over{Variance}$  
 >  $y$ :  실제값, $\hat{y}$ :  예측치, $\bar{y}$ :  평균치
 
-  R-Squared
 - 회귀 성능의 기본적인 평가 지표
 - MSE를 분산으로 정규화한 값을 사용한다
 - R-sqaured 값은 1에 가까울수록 완벽한 예측을 수행한 것이고, 0 근처이면 오차 평균치가 표준편차 정도인 경우이다.
```

### 분자 특성 추가

- [rdkit.Chem.Descriptors](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Fwww.rdkit.org%2Fdocs%2Fsource%2Frdkit.Chem.Descriptors.html)이 제공하는 함수들을 사용하면 다양한 분자특성들을 알 수 있다. 아래의 특성을 추가하겠다
    - TPSA() - the surface sum over all polar atoms or molecules also including their attached hydrogen atoms;
    - ExactMolWt() - 정확한 몰 중량
    - NumValenceElectrons() - number of valence electrons (may illustrate general electronic density)
    - NumHeteroatoms() - general number of non-carbon atoms
- 이외에도 분자의 링정보를 보거나, 원자별로 결합 정보를 볼 수 있다
    - GetRingInfo(), GetAtoms(), GetBonds() 등을 사용

**→ lin.coef_**

## 트리 모델

- 선형 모델로 부족한 경우 : 카테고리 변수가 많다. / 입출력 관계가 비선형적

### decision-tree

- depth가 너무 깊으면 overfitting될 수 있음 → **dtr.score()값**이 떨어지면 멈추자! 제일 높은 값의 depth 기준
- feature_importances 모두 **양수 →** 좋은 결정 트리 분류 기준

```python
# 최적의 트리 깊이 (max_depth)를 실험으로 찾는다

for depth in range(1,30,2):
    dtr = DecisionTreeRegressor(max_depth=depth) 
    dtr.fit(X_train, y_train) 
    print(depth, dtr.score(X_test, y_test).round(3))

# 결정 트리 회귀 모델

dtr = DecisionTreeRegressor(max_depth=17) 
dtr.fit(X_train, y_train) 
y_pred = dtr.predict(X_test)
show_reg_result(y_test, y_pred)

# 트리 모델이 제공하는 특성 중요도 (feature_importances_) 보기

plot_feature_weight(train_df.columns, dtr.feature_importances_)
```

### ★ random-forest

- tree 모델 개선 - 앙상블 기법 (부분적인 데이터를 많이 보여줌 - 데이터 적게 주는 weak 모델화. 다양한 상황을 조합해 확인하는 효과.)
- tree 모델보다 feature를 골고루 사용.

```python
rfr = RandomForestRegressor() 
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
show_reg_result(y_test, y_pred)

# 랜덤 포레스트 모델이 제공하는 특성 중요도 (feature_importances_) 보기
plot_feature_weight(train_df.columns, rfr.feature_importances_)
```

### 트리 그리기 → max_depth 설정

feature_importances 계속 변화

```python
from sklearn import tree
import matplotlib
plt.figure(figsize=(26,12))

tree.**plot_tree**(dtr, fontsize=14,
              feature_names=train_df.columns,
              filled=True,
              impurity=True,
              max_depth=3)
plt.show()

# value = 평균치 -> variance가 작아지도록 학습
```

## Fingerprint 표현형

- mol 객체로부터 ECFP Fingerprint를 구하는 함수 정의
- GetMorganFingerprintAsBitVect() 함수를 사용한다.
- 분자별 Fingerprint 정보를 2차원 어레이로 만들기
- 여러 어레이를 합치기 위해서 np.**vstack**()을 사용한다.

```python
def mol_2_fp(mol):
  fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
  # fp_arr = np.zeros((1, ), dtype=np.int8)
  # DataStructs.ConvertToNumpyArray(fp, fp_arr)
  # return fp_arr
  return fp

list_fp = df['mol'].apply(mol_2_fp)   # 분자별 Fingerprint 정보를 2차원 어레이로 만들기
ecfp = np.vstack(list_fp)
print(ecfp.shape)
ecfp[:3]

# 성능 확인
X_train, X_test, y_train, y_test = train_test_split(ecfp, y, test_size=.1)

lin.fit(X_train, y_train)
y_pred = lin.predict(X_test)
show_reg_result(y_test, y_pred)
```

### ★ **Light GBM**

- **부스팅** 방식(순차적, 이전 모델 결과 활용하여 개선)의 앙상블 모델인 Light GBM 사용하기
- **랜덤 포레스트**와 성능이 비슷하며 학습 속도가 **빠르다**

```python
from lightgbm import LGBMRegressor
lgbm_r = LGBMRegressor()
lgbm_r.fit(X_train, y_train)
y_pred = lgbm_r.predict(X_test)
show_reg_result(y_test, y_pred)
```

### **Scaffold Splitter**

- 훈련과 검증 데이터를 나눌때 ~~train_test_split를 사용하여 랜덤하게~~ 나누지 않고 **분자의 Scaffold를 고려**하여 훈련과 검증 데이터에 유사한 Scaffold가 섞이지 않게 한다.
    - random split을 사용했을 때 문제 : 화합물의 scaffold(구조, 골격이 비슷) - 훈련과 검증 데이터에 유사한 것이 있으면 성능이 좋은 것처럼 보이게 할 수 있음.

```python
# Scaffold Splitter를 확인하기 위한 샘플 분자 정의
data_test= ["CC(C)Cl" , "CCC(C)CO" ,  "CCCCCCCO" , "CCCCCCCC(=O)OC" , "c3ccc2nc1ccccc1cc2c3" , "Nc2cccc3nc1ccccc1cc23" , "C1CCCCCC1" ]
Xs = np.zeros(len(data_test))
Ys = np.ones(len(data_test))

# deepchem dataset를 정의하고 deepchem이 제공하는 **ScaffoldSplitter**()를 수행한 결과 보기
dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(data_test)),ids=data_test)
scaffoldsplitter = dc.splits.ScaffoldSplitter()
train, test = **scaffoldsplitter.train_test_split**(dataset)   # 앞의 train_test_split과 다르게 scaffoldsplitter에서 제공된 것
train, test

# 앞의 예제에 Scaffold Splitter를 적용한 경우의 성능 보기
Xs = df[['num_of_atoms', 'num_of_heavy_atoms',
       'num_of_C_atoms', 'num_of_O_atoms', 'num_of_N_atoms', 'num_of_Cl_atoms',
       'tpsa', 'mol_w', 'num_valence_electrons', 'num_heteroatoms']]
Ys = df['logP']
ids = df['smiles']

dataset = dc.data.DiskDataset.from_numpy(X=Xs,y=Ys,w=np.zeros(len(Xs)),ids=ids)
scaffoldsplitter = dc.splits.ScaffoldSplitter()
train,test = **scaffoldsplitter.train_test_split**(dataset)   # dataset 수가 줄어든다.

lin.fit(train.X, train.y)
y_pred = lin.predict(test.X)
show_reg_result(test.y, y_pred)
```

## Graph 사용

• 분자 표현형으로 그래프 객체인 ConvMol을 사용하고 머신러닝 모델로 GraphConvModel을 사용한다.