# Terpene-former

This is the directory of the work 

## Dependency

Follow the below steps for dependency installation:
'''
conda create -n terpene python=3.7.0
conda activate terpene
pip install -r requirements.txt
'''
The rdchiral package can be taken from [here](https://github.com/connorcoley/rdchiral) (no need to install it).

## Data
Retro-tero has been provided.
The raw USPTO-50K dataset can be taken from [here](https://github.com/Hanjun-Dai/GLN).
The raw BioNavi data can be taken from [here](https://github.com/prokia/BioNavi-NP).

## Train & Translate
Simply run ./train.sh for training and ./translate.sh for testing. One can specify different model by changing configurations in scripts. 

## Reference
If you want to refer to our work, please cite by:
'''
'''