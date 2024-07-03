## SA-MTL

**code repository for ICPR 2020 poster paper** : Predicting Chemical Properties using Self-Attention Multi-task Learning based on SMILES Representation

**Original transformer code from** : [https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb)

**Tox21 data(same preprocessed version)** : Convolutional neural network based on SMILES representation of compounds for detecting chemical motif.

## Dependency

(latest versions will suffice)

- rdkit
- numpy
- sklearn
- tensorflow version 2

## Running code

To run the singleTask.py for a specific task such as NR-AR:   

`python3 singleTask.py -p NR-AR`  

 
To run the multiclass code:  

`python3 selfAttMulticlass.py`