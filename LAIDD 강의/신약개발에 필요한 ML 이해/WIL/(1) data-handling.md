Digital biology : 예전엔 logic-driven => 최근에 data-driven(end-to-end 방식)

<Bio data issue>

- phenotype data : variant, noisy, hard to 설명 (다양한 형태의 data)
- hard to understand : needs end-to-end models
- real world data sharing : privacy
- multimodality : 다양한 format의 data

![Untitled](https://github.com/doammii/CADD-study/assets/100724454/e877f68e-2b26-409f-8317-e16f11ce3845)

중요한 건 **“data representation”** : 입력 data의 feature을 잘 표현해야 AI 잘 학습 가능

AIDD의 장점 : predictive task + generative task

[colab.research.google.com/github/](http://colab.research.google.com/github/)

---

### Public Data

```python
!pip install deepchem

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import AllChem
from rdkit import DataStructs

IPythonConsole.ipython_useSVG=True
%config InlineBackend.figure_format = 'retina'
```

---

- **ChEMBL**
    - target&activity type이 dictionary → `targets = pd.DataFrame.from_dict(target_query)`
    - 결측치가 많은 column 삭제 : `data= data.dropna(axis=1)`
    - 목적 변수 범주화
        - 목적 변수로 standard_value를 선택
        - 등급을 표시하기 위해서 activity column 추가
    - Lipinski’s Rule(Rule of five) : ADME에 기반한 프로파일
        - 일부 Descriptor(분자 특성 상세하게 표현), Lipinski data 추가하기 `from rdkit.Chem import Descriptors, Lipinski`
        - MW, LogP, NumHDonors, NumHAcceptors 특성값
    - IC50 : 저해농도, 로그값 pIC50으로 표현 후 nM에서 M단위로 데이터 변환
    - **데이터 탐색** : activity값 분포 확인 - box plot, scatter plot/pair plot
    - **데이터 저장** : 인덱스 여부 고려
- **PubChem**
    
    ```python
    !pip install pubchempy
    import pubchempy as pcp
    ```
    
    - get_properties function 이용
- **MoleculeNet** - DeepChem library에 포함되어있음.
    - AUC-ROC, AUC-PRC, RMSE, MAE metric을 사용하여 성능 비교 평가 수행 가능(benchmark)
    - deepchem.molnet module에서 데이터 다운로드하는 함수 제공 : `**deepchem.molnet.load_XXX()**`
        - **return**값 : **tasks**(타겟 작업/하나 또는 복수) + **datasets**(deepchem.data.Dataset 객체) + **transformers**(deepchem.trans.Transformer 객체로 **전처리** 방법 알려줌.)
        - 다운로드시 원하는 feature 선택 가능
    - **load_XXX() 옵션**
        - featurizer : ECFP, GraphConv, Weave, smiles2img
        - splitter : None, index, random, scaffold, stratified
    - **dataset type**
        
        각 행은 분자 구분 - features(x), labels(y), weights, ID(unique identifier/SMILES)
        
    
    ### 데이터 읽기
    
    - `iterbatches(batch_size=100, epochs=10, deterministic=False)`
    - epoch를 지정할 수 있으며, 읽을 때마다 순서를 랜덤하게 바꾼다
    - `to_dataframe()`을 사용하면 데이터프레임으로 읽는다
    - TensorFlow 타입, 즉 `tensorflow.data.Dataset`을 얻으려면 `make_tf_dataset()`를 사용한다
    - **Torch** 타입의 `torch.utils.data.IterableDataset`을 얻으려면 `make_pytorch_dataset()`를 사용한다
    
    ### Dataset 생성
    
    ndarray data → DeepChem dataset과 호환성을 갖는 dataset으로 만들기 위해 **NumpyDataset**을 사용!
    
    `dataset = dc.data.NumpyDataset(X=X, y=y)`