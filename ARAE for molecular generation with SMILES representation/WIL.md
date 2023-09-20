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

### 사전 지식

- EGFR(epidermal growth factor receptor) : hit compounds의 de novo design에 사용 가능
- posterior distribution - VAE
- GAN의 “**mode-collapse problem**” 문제 : **Wasserstein distance를** 도입해서 해결 가능
- Bayesian graph convolutional network + **EGFR** subset of the **DUD-E dataset**
- ZINC & QM9 dataset

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

### VAE

### ARAE

## Methods

분자 구조의 SMILES representation → discrete random variable input

train/test dataset으로 QM9 & ZINC dataset 활용.

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

### Performance of ARAE on Molecular Generation

### Conditional Generation of Molecules + CARAE

### De Novo Design of EGFR Inhibitors

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
