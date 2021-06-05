# Event-Based-Modality
## The Possible, the Plausible, and the Desirable: Event-Based Modality Detection for Language Processing

Table of Contents
=================
* [Introduction](#introduction)
* [Main Features](#main-features)
* [Requirements](#requirements)
* [Setup](#setup)
* [Basic Usage](#basic-usage)
* [Important Notes](#important-notes)
* [Training Your Own Model](#training-your-own-model)
* [Evaluation](#evaluation)
* [Citations](#citations)

## Introduction
Code for event-based modality detection, as described in the ACL paper ["*The Possible, the Plausible, and the Desirable: Event-Based Modality Detection for Language Processing"*](to-be-updated). The code allows training, data processing, and includes configuration files for the experiments described in the paper. 

## Main Features


## Requirements
1. `python>=3.6`
1. `torch=1.5.1`
1. `spacy==2.1.9`
1. `allennlp==1.0.0`

## Data
You can find the data for training your own model in the following repository:
https://github.com/OnlpLab/Modality-Corpus .

To convert from the conll format to the format used for training use the following script:
data_processing/convert_formats.py

## Training Your Own Model
We used the code from AllenNLP (https://github.com/allenai/allennlp).
If you want to train your own model you need to specify the train and dev files in the jsonnet configuration files
in the following folder: Roberta/experiments.
Afterwards you can run the model with the following command: 
allennlp train experiments/my_config.jsonnet --include-package my_library -s output_directory

## Evaluation
To evaluate your predictions you first need to convert them to a different format using this script:
Roberta/scripts/format.py
Afterwards you can use the conll evaluation script: Roberta/scripts/conlleval_orig.py .

## Citations
If you use code from repository or our models, please cite the Event-Based-Modality paper:
```bibtex
@InProceedings{eventbasedmodality,
  author    = {Pyatkin, Valentina and Sadde, Shoval and Rubinstein, Aynat and Portner, Paul and Tsarfaty, Reut},
  title     = {The Possible, the Plausible, and the Desirable: Event-Based Modality Detection for Language Processing},
  year      = {2021},
}
```
