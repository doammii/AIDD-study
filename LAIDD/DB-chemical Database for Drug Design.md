# RDkit + 화합물DB

## RDkit

Open-Source Cheminformatics Software

[The RDKit Documentation — The RDKit 2023.03.1 documentation](https://www.rdkit.org/docs/)

**QED**

```python
from rdkit import Chem
m = Chem.MolFromSmiles('Cc1ccccc1’)

from rdkit.Chem import QED
qed = QED.qed(m)
print(qed)
```

## 화합물 DB

![chem-DB](https://github.com/doammii/CADD-study/assets/100724454/c26e1443-a9da-46ef-b416-9e1b25f4bc63)


- **ChEMBL** : 정리 good (sequence-based)
- **PubChem** (ligand-based)
- **DrugBank**
- 화합물 → **ZINC**
- 약물 및 단백질 → **PDB** (structure-based)

---

- **PubChem** : 다양하고 큰 규모
    - DB 구성
        
        ![PubChem](https://github.com/doammii/CADD-study/assets/100724454/6d56b536-c3a5-4277-a257-2fd4997afaa7)

        
    
    components
    
    - compound : unique → QSAR
    - substance : 실험 → entity, substance가 compound 포함, only combination
    - bioassay
    - bioactivity
- **ChEMBL** : 사람이 잘 정리해놓았고, 주기적으로 update되는 버전.
    - QSAR는 ChEMBL에서 다운로드받는게 적합
    - 이용하는 주 목적 : drug target에 어떤 compound → active ligand?
        - contents : drug(대부분 단백질 target)
        - assay : functional, binding, ADME
        - compounds는 ‘.mol’ 파일, uni-compound는 InCHI로 저장.
- **DrugBank** : 약물 compound
    
    absorption, pharmacodynamics, mechanism of action
    
- **ZINC**
    
    virtual screening(docking)기반. (library in SMILES format for ligand-based vs)
    
- **PDB**(Protein Data Bank)
    
    단백질 target → structure-based
    PDB ID는 4-letter code
    

## Dataset 종류

*FP2VEC paper 참고*

- **Tox21** dataset
    - 8014 compounds/toxicity(label) ↔ 12 targets
        
        1 compound와 multiple targets 연결 → single-task learning과 multi-task learning으로 나뉨.
        - 801 "dense features" that represent chemical descriptors, such as molecular weight, solubility or surface area
        - 272,776 "sparse features" that represent chemical substructures (ECFP10, DFS6, DFS8; stored in Matrix Market Format)     
        - Machine learning methods can use sparse or dense data or combine them. 
        - For each sample there are 12 binary labels that represent the outcome (active/inactive) of 12 different toxicological experiments.
        
    - random split method (train, valid, test sets로 random하게 나눔.)
- **SIDER** dataset
    - marketed drugs & ADR(adverse drug reactions/binary label) against 27 System-Organs Class 포함.
    - 1427 data points ↔ 27 targets
    - random split method + single/multi-task
- **HIV** dataset
    - experimental measurement of the ability to inhibit HIV replication
    - 41,127 compounds ↔ ability of inhibition with binary labels
    - scaffold split method (two-dimensional molecular structures에 따라 3개의 sets로 나눔.)
        
        **RDkit**에 의해 구현.
        
- **BBBP** dataset
    - blood-brain barrier penetration with binary labels
    - 2050 compounds
    - scaffold split method
- **ESOL** dataset
    - measurements of the water solubility of small compounds
    
        water solubility는 measured log solubility in moles/liter로 측정
        
    - 1128 compounds ↔ water solubility
    - random split method
- **FreeSolv** dataset
    - hydrogen-free energy of small compounds in a water environment
    - 642 molecules + hydrogen-free energy
    - random split method
- **Lipophilicity** dataset
    - octanol/water distribution coefficient at pH7.4
    - 4200 compounds ↔ lipophilicity values
    - random split method
- **Malaria** dataset
    - half-maximal effective concentration(EC50) values of a sulfide-resistant strain of Plasmodium falciparum(malaria source)
    - 9998 compounds ↔ EC50 values
    - random split method
- **CEP** dataset
    - Clean Energy Project - solar cell materials에 맞는 candidate molecules 포함
    - 29,978 compounds ↔ CEP values
    - random split method