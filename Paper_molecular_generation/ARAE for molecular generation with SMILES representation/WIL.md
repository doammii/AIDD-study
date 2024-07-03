# Molecular Generative Model Based on an Adversarially Regularized Autoencoder

### Abstract

Deep generative models for molecular design

VAE(variational autoencoder) or GAN(generative adversarial network) has a limitation : low validity & uniqueness

→ ARAE(adversarially regularized autoencoder) 제안

- VAE처럼 latent variables를 이용
- GAN과 같은 adversarial training에 의해 latent variables의 distribution 추정

→ avoid insufficiently flexible approximation of posterior distribution (VAE)

→ avoid difficulty in handling discrete variables (GAN)

생성된 분자마다 validity, uniqueness, novelty 성능이 기존 모델들보다 뛰어남.

single & multiple properties에 대해 성공적인 conditional generation of drug-like molecules

주어진 조건들을 동시에 만족하면서 active molecules의 scaffolds도 공유하는 EGFR(epidermal growth factor receptor) inhibitors도 생성 가능


## Introduction

Design novel(de novo) molecules + desired(target) properties → Molecular generative models

- **key idea** : estimate the distribution of molecules → sample unseen molecules with target properties
- GAN + latent variable models
    - GAN : implicitly estimate the distribution of input data by training adversarially
    - Latent variable models : estimate the distribution of (input data에 대응되는) latent variables + generate new molecules by decoding the sampled latent variables
        
        VAE(variational autoencoder), AAE(adversarial training)
        
- molecular generation을 probabilistic autoregressive process로 보면 RNN 사용

Conditional generative models : promote targeted molecular generations

- ex) ChemicalVAE : autoencoder, classifier networks를 train시켜 molecules를 latent space와 유사한 properties로.
- **Molecules with desired(targeted) properties 장점**
    - can be searched by advanced Bayesian optimization
    - can directly incorporate target properties into the generation process to estimate a molecular distribution or manipulate a latent space

VAE & GAN 기반 model의 한계

- VAE
    - low **validity** : often produce unnatural molecules or even invalid outputs
    - 원인 : posterior distribution이 surrogate(variational) distribution에 의해 추정되어 insufficiently flexible distribution이 추정의 질을 악화
- GAN (ORGAN, MolGAN)
    - higher validity (adversarial training 덕분에)
    - low **diversity**
        
        **원인** : **discrete** (categorical) representation of atomic symbols로 표현되는 분자 구조
        
        → train시킬 때 같은 instance가 반복 생성됨.
        
        - “**mode-collapse problem**”이 원인이 아님. (**Wasserstein distance**를 probability 측정에 사용했는데도 low diversity 문제를 해결하지 못했음.)
    - low **uniqueness** : novel molecular generation을 위한 policy networks를 도입했는데도
1. **ARAE**(adversarially regularized autoencoder) framework 제안 → high validity & uniqueness
    - latent variable models : discretized molecular structure inputs → continuous latent representation으로 변형
    - key distinct feature : GAN의 adversarial training을 latent variables의 distribution을 추정하는데 사용
        
        → 너무 단순한 prior에 의한 variational posterior approximation으로 발생하는 문제피할 수 있음.
        
2. **Conditional ARAE** 소개 : desired properties에 따른 분자 생성
    - variational mutual information minimization framework → manipulate latent variables + target properties
    - designated molecular properties를 가지는 unseen molecules 생성 가능
- **LatentGAN**(by Prykhodoko) : estimate the distribution of latent variables by adversarial training
    
    차별점 1 : ARAE가 VAE, GAN의 한계를 어떻게 극복했는지에 대한 이론적 디테일 제공 → justify
    
    차별점 2 : conditional molecular generations를 desired properties를 가진 분자를 생성하는데 적용하는 방법 소개
    

3가지 Usefulness of ARAE model

- high performance for estimating the latent vector distribution
    
    validity, uniqueness, novelty
    
    test the smoothness of the latent space ← interpolating between 2 vectors in the latent space
    
- simulatenous control of multiple properties + high success rate의 실현가능성
- ARAE model이 drug discovery에서 hit compounds의 de novo design으로 사용될 수 있음을 보여줌. + EGFR inhibitors → possible practical application

## Previous works

( * objective mathematical expression : pdf file 참고)

### GAN

- estimate distribution of **input samples ← adversarial training** of generator network & discriminator network
- learning ~~discrete~~ representation → low-diversity problem ⇒ 분자의 distribution을 ~~data space~~보다는 **continuous latent space**로 estimate하자!

### VAE

- latent variable model - distribution of latent variables(**posterior**)
    
    generate instances by decoding latent variables
    
- exact posterior distribution 대신 approximating with variational distribution(output of encoder)
    - decoder : reconstruct inputs from the latent variables
    - minimizing the 2nd term → 2 distribution similar → posterior distribution can be approximated by **prior distribution(predefined prior, surrogate distribution)**
    
    → perform explicit density estimation
    
- Why VAES often generate unrealistic instances?
    - true posterior distribution may not be well approximated by a given prior(ex: multivariate Gaussian distribution)
    - minimizing the KL divergence is not suitable - if the posterior distribution is ‘multimodal’
    
    **‘dead regions’** in the latent space : latent variables are decoded to invalid data points
    

### ARAE

- latent variable model : encoder-decoder architecture
    - encoder : output the distribution of true latent variables z from the given inputs
    - decoder : reconstruct the input from the latent variable drawn from the posterior
- posterior distribution ← adversarial training. minimizing the 1-Wasserstein distance between true&generated latent variables

## Methods

- 분자 구조의 SMILES representation → **discrete** random variable input
    
    train/test dataset으로 QM9 & ZINC dataset 활용.
    
- latent representation으로 변형, distribution ← adversarial training
    - avoid posterior collapse problem (VAEs) → high valid rate
    - estimating the distribution of molecules in the continuous latent space → low-diversity problem
- efficient conditional generation scheme of molecules (conditional generative model) : **conditional ARAE**
    1. **encoder** : SMILES sequences → latent variables
    2. **generator** : produce new samples by taking random variables
    3. distributions of 2 variables → becomes similar by minimizing 1st, 2nd term + **gradient descent optimization**
    - **predictor** network : estimate variational mutual information (VMI)
    - training : **decoder** reconstructs input molecular structures & property information of input molecules
    - inference : sample new molecules by tuning the latent vector & specifying the desired property


### Implementation details

encoder & decoder : single LSTM + dimensionality of ouputs : 300

- LSTM encoder : sequential SMILES strings → latent vectors로 변형

generator & discriminator adversarial training : 2 FC layers + 300 hidden dimensions for ZINC(200 for QM9) - predictor network도 똑같음.

Bayesian graph convolutional network + **EGFR** subset of the **DUD-E dataset**

- hp : number of graph convolution layers, dropout rate, weight decay coefficient
- train : test = 8 : 2
- train : batch size, training epoch, learning rate
- inference : predictive distribution with the number of Monte Carlo sampling of 50

## Results and discussion

### Metrics

- **Validity** : (number of valid molecules) / (number of generated samples) ← RDKit
- **Uniqueness** : (number of unrepeated molecules) / (number of valid molecules)
- **Novelty** : (number of molecules - not included in the training set) / (number of unique molecules)
- **Novel/sample** : (number of valid, unique, novel molecules) / (number of generated samples)
- **Diversity** : all N molecule pairs in test set에 대해 측정. 2 molecules’ similarity ← Tanimoto similarity(radius : 4, 2048 bits)
- 3 molecular properties for conditional generations : log P, TPSA, SAS

### Performance of ARAE on Molecular Generation

- training model with **QM9** dataset
    - each epoch : 10,000 samples generated
    - evaluation metrics **smoothly converged** : common difficulties in training GANs(mode collapse, diminished gradient) are less problematic
- ChemicalVAE, GrammarVAE : SMILES
    
    GraphVAE, MolGAN : molecular graphs
    
    → ARAE가 **novelty**를 제외하고 outperform
    
- validity & uniqueness are correlated
    - if there are many dead zones in latent space → only latent vectors from very restricted regions(can be correctly decoded to valid molecules) → low diversity & igh rebundancy
    - wide region of latent space → higher diversity & lower rebundancy

**ARAE’s low novelty : low chemical diversity of QM9 dataset 때문**

- limited number of heavy atoms → novel molecules 생성 기회 제한
- **ZINC** dataset에서는 high novelty 가능 ← ZINC dataset spans a huge chemical space
- **MolGAN**도 adversarial training 적용 → high novelty BUT low uniqueness → **novelty/sample** 이 더 나은 metric
    
    Graph representation > SMILES : higher novel/sample values
    

Approximating a posterior distribution + insufficiently flexible prior → low validity(VAE)

→ unrealistic molecules at interpolation points 생성 가능하다는 문제 → **latent space with the adversarial training**

high membered ring molecules(from VAE-based molecular generative models) not produced by our model

### Conditional Generation of Molecules + CARAE

Generate molecules + high validity, uniqueness, novelty, diversity

- **designated property values**에 의해 multiple property control 성능 결정
    - 3 target properties : log P, SAS, TPSA → 동시에
    - ARAE와 비교했을 때 높은 **SAS** 값 빼고 유의미한 결과
        
        SAS : synthetic accessibility & structural stability와 연관 → 높은 SAS값에서 valid molecules의 빈도가 낮아짐. 
        
- simultaneous control of 3 target properties → each distribution was well localized about a given designated point & separated
    
    → high accuracy of CARAE model for multiple property control
    

target values(given designed points/properties)에 대해 각 latent space distribution이 localized(separated) → high accuracy of multiple property control

### De Novo Design of EGFR Inhibitors

designing the novel inhibitors of an EGFR

- **DUD-E** dataset(active, decoy molecules)
- CARAE model + 4 target properties(activity against EGFR, log P, TPSA, SAS)

Inhibitors generation scenarios

- one activity condition - generate EGFR-active molecules only
- 4 condiitons - molecules satisfying Lipinski’s rule of five & synthesizability

evaluate EGFR inhibition activity of the generated molecules - **Bayesian graph convolutional network(Bayesian GCN)**

- Bayesian GCN은 DUD-E dataset의 EGFR subset에 의해 처음으로 훈련&테스트
- successful molecules 수는 4 conditions scenario에서 감소 ← 추가 조건들이 더 엄격한 제한사항 만듦.

## Conclusions

Molecular generative models : VAE or GAN

- VAE : often produce invalid molecules (insufficiently flexible approximation of posterior distribution 때문)
- GAN
    - adversarial training → estimate the distribution implicitly from the input data → validity enhanced
    - training discrete variables → low uniqueness of generated molecules

**ARAE** for AI-based molecular design in various chemical applications

- latent variable model
    
    distribution of latent variables is obtained by adversarial training from training data, ~~approximating with a predefined function~~
    
- continuous latent vectors
    
    ~~discrete molecular structures in the adversarial training~~
    

Benchmark studies → evidence of the high performance of ARAE model

- uniqueness, novel/sample-ratio + high validity for QM9 dataset - outperform
- allow smooth interpolation between 2 molecules in the latent space
    
    → feasibility of the successful modeling of latent space + adversarial training
    
- conditional generation - single, multiple properties → design EGFR inhibitors
