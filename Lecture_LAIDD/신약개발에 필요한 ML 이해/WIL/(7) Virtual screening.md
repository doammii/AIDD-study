# Virtual screening

- 어떤 조건을 만족하는 분자를 찾아내는 작업을 **실험을 수행하지 않고 머신러닝 모델로** 찾는 것을 말한다
- 강의 예제 : ERK2 단백질을 억제하는 분자를 선별하는 모델을 만들기
    
    DUD-E 데이터베이스에서 활성 및 비활성 분자 데이터를 다운받아 사용
    
    - Extracellular signal-regulated kinase (다수의 생화학적 신호에 관여하고 세포의 증식 분화 전사조절에 관여)
    - 비소세포성 폐암 non-small cell lung cancer과 흑색종(피부암)에 대해 임상시험중
- **그래프 컨볼류션 모델**을 사용

### 데이터 준비

- 활성 및 비활성 데이터를 다운로드 받는다
- 분자 데이터는 SMILES 형식이다. 이를 데이터프레임에 불러온다
- AddMoleculeColumnToFrame()를 사용해 SMILES에 해당하는 Mol 객체를 데이터프레임에 추가할 수 있다.

```python
active_rows, active_cols = active_df.shape
active_df.columns = ["SMILES","ID","ChEMBL_ID"]
active_df["label"] = ["Active"]*active_rows
PandasTools.AddMoleculeColumnToFrame(active_df,"SMILES","Mol")
```

|  | SMILES | ID | ChEMBL_ID | label | Mol |
| --- | --- | --- | --- | --- | --- |
| 0 | Cn1ccnc1Sc2ccc(cc2Cl)Nc3c4cc(c(cc4ncc3C#N)OCCC... | 168691 | CHEMBL318804 | Active | <rdkit.Chem.rdchem.Mol object at 0x7fab8b8e6090> |
| 1 | C[C@@]12[C@@H]([C@@H](CC(O1)n3c4ccccc4c5c3c6n2... | 86358 | CHEMBL162 | Active | <rdkit.Chem.rdchem.Mol object at 0x7fab8b8e6030> |
| 2 | Cc1cnc(nc1c2cc([nH]c2)C(=O)N[C@H](CO)c3cccc(c3... | 575087 | CHEMBL576683 | Active | <rdkit.Chem.rdchem.Mol object at 0x7fab8b8e60f0 |

**특성값 추가 → active/Decoy 데이터 label 구분 → 데이터 탐색(violin plot)**

특성값

- 분자량 molecular weight
- 분배 계수 partition coefficeint(LogP)
- 전하 charge (양인지 음인지)

```python
def add_property_columns_to_df(df_in):
    df_in["mw"] = [Descriptors.MolWt(mol) for mol in df_in.Mol]
    df_in["logP"] = [Descriptors.MolLogP(mol) for mol in df_in.Mol]
    df_in["charge"] = [rdmolops.GetFormalCharge(mol) for mol in df_in.Mol]

# 비활성
decoy_df = pd.read_csv("https://raw.githubusercontent.com/deepchem/DeepLearningLifeSciences/master/Chapter11/mk01/decoys_final.ism",
                       header=None,sep=" ")
decoy_df.columns = ["SMILES","ID"]
decoy_rows, decoy_cols = decoy_df.shape
decoy_df["label"] = ["Decoy"]*decoy_rows
PandasTools.AddMoleculeColumnToFrame(decoy_df,"SMILES","Mol")
add_property_columns_to_df(decoy_df)

tmp_df = active_df.append(decoy_df)   
# active/inactive 데이터 불균형. 특히 charge가 문제 - 전하 차이에 의해 모델의 bias 발생하므로 화학 구조를 수정하여 charge 제거
# 업데이트 이후 특성 다시 계산. revised_decoy_df
tmp_df.shape

```

### 분류 모델

GCN 모델 사용

1. **GCN model 생성**
- RandomSplitter() 사용
- 화학 구조를 기반으로 데이터셋을 나누는 ScaffoleSplit(), 데이터를 군집화 한 후에 데이터셋을 분리하는 **ButinaSplitter**()도 있다
- 불균형 데이터에 대한 성능 평가시에는 매튜 상관계수 Matthews Correlarions Coefficients(MCC)를 사용한다
    - +1이면 완벽한 예측, 0이면 랜덤 예측, -1이면 완전히 반대되는 예측을 의미한다

```python
def generate_graph_conv_model():
    batch_size = 128
    model = **GraphConvModel**(1, batch_size=batch_size, 
             mode='classification', model_dir="./model_dir")
    return model

dataset_file = "dude_erk1_mk01.csv"
tasks = ["is_active"]
featurizer = dc.feat.**ConvMolFeaturizer**()
loader = dc.data.CSVLoader(tasks=tasks, feature_field="SMILES", featurizer=featurizer)
dataset = loader.create_dataset(dataset_file, shard_size=8192)

splitter = dc.splits.**RandomSplitter**()
metrics = [dc.metrics.Metric(dc.metrics.matthews_corrcoef, np.mean)]

training_score_list = []
validation_score_list = []
transformers = []
**cv_folds** = 5

for i in tqdm(range(0,cv_folds)):
    **model** = generate_graph_conv_model()
    train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(dataset)   # default : 8:1:1
    model.fit(train_dataset)
    train_scores = model.evaluate(train_dataset, metrics, transformers)
    training_score_list.append(train_scores["mean-**matthews_corrcoef**"])
    validation_scores = model.evaluate(valid_dataset, metrics, transformers)
    validation_score_list.append(validation_scores["mean-matthews_corrcoef"])

**# 5개의 metric 값이 비슷한지 확인**
print(training_score_list)
print(validation_score_list)
```

1. **검증 데이터 분류 예측**

```python
pred = [x.flatten() for x in model.predict(valid_dataset)]

pred_df = pd.DataFrame(pred,columns=["neg","pos"])
pred_df["active"] = [int(x) for x in valid_dataset.y]
pred_df["SMILES"] = valid_dataset.ids

pred_df.sort_values("pos",ascending=False).head(5)

sns.boxplot(x=pred_df.active,y=pred_df.pos)

# false 샘플 찾기
false_negative_df = pred_df.query("active == 1 & pos < **0.5**").copy()
PandasTools.AddMoleculeColumnToFrame(false_negative_df,"SMILES","Mol")
false_negative_df # (실제로는 active한데 pos는 낮음.)

false_positive_df = pred_df.query("active == 0 & pos > **0.5**").copy()
PandasTools.AddMoleculeColumnToFrame(false_positive_df,"SMILES","Mol")
false_positive_df # (낮은데 실제로 inactive)
```

1. **모든 데이터로 학습 → for 성능 향상**
2. **rd_filters 적용**
- Pat Walters가 만든 스크립트로 잠재적으로 문제가 있는 분자를 걸러낸다
- 생물학적 분석이 어려운 분자들을 제거하기 위해서 rd_filters.py를 사용
- [rd_filters github](https://github.com/PatWalters/rd_filters)
1. **Zinc 데이터에 필터 적용**

FILTER의 경우의 수를 파악하기 위해서 collection의 Counter를 사용

1. **모델 사용**
- GCN 모델을 불러오고, 피처화기를 만든후, 분자 데이터를 피처화
- 예측값을 확인하고 가장 높은 점수를 얻은 화학구조를 확인
- 분자 데이터를 CSV로 저장
- CSV 파일을 읽어온다
- dataset을 만든후 모델을 사용하기 위해서 피처화를 수행한다
    - ConvMolFeaturizer 사용
- 활성화와 비활성화를 구분하기 위해서 점수를 시각화 한다
- SMILES와 예측값이 있는 데이터프레임을 합친다
- 예측값이 높은 분자를 살펴본다 → 예측값이 높은 분자들의 모양(scaffold)가 비슷한 것을 알 수 있다
- 중복되는 것을 피하기 위해서 **군집화를** 수행한다
    - 화학적 유사성을 사용하는 **Butina 군집화**를 사용하겠다
    - 군집화를 하려면 기준값을 주어야 한다 (아래에서 **Tanimoto** 유사도 > **0.35**)
    
    ```python
    def butina_cluster(mol_list,cutoff=0.35):
        fp_list = [rdmd.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for m in mol_list]
        dists = []
        nfps = len(fp_list)
        for i in range(1,nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fp_list[i],fp_list[:i])
            dists.extend([1-x for x in sims])
        mol_clusters = Butina.ClusterData(dists,nfps,cutoff,isDistData=True)
        cluster_id_list = [0]*nfps
        for idx,cluster in enumerate(mol_clusters,1):
            for member in cluster:
                cluster_id_list[member] = idx
        return cluster_id_list
    ```