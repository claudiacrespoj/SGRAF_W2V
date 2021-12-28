# SGRAF_W2V
*Python 3.6 and pytorch implementation of paper paper of [**“Similarity Reasoning and Filtration for Image-Text Matching”**](https://arxiv.org/pdf/2101.01368.pdf).*  with an added embedding layer 

* the authors of the original paper build the code based of [SCAN](https://github.com/kuanghuei/SCAN) 


## Introduction

**The framework of SGRAF with an added embedding layer:**

<img src="./fig/model.png" width = "100%" height="50%">

## Requirements 
To run the code in your local we recommend  **python 3.6**.
* cd to the directory where requirements.txt is located.
* activate your virtualenv.
* run: pip install -r requirements.txt in your shell.

## Download data and vocab
We follow [SCAN](https://github.com/kuanghuei/SCAN) to obtain image features, proccessed vobularies with gensim can be download from GDrive link, to train please place all files in data folder

* The preprocessed training images [here](https://drive.google.com/file/d/1-K5S2EN5juOUXG8JwDoJL2FrJ_kg4w5-/view?usp=sharing) 
* The preprocessed test images [here](https://drive.google.com/file/d/18zTyLNt6Zu0iVQObnAQcJ9_ygkkZh8lf/view?usp=sharing)

* The pretrained word2vec model can be download from [here](https://drive.google.com/file/d/1okujwj6TlOUypavjSgQN0lNx_lNZfop5/view?usp=sharing)



* Vocab preprocessed can be download [here](https://drive.google.com/file/d/1KnDtgoOlVnk0M9x-w2gS_1FmogaQaYji/view?usp=sharing) 

## Pre-trained models and evaluation
the pretrained SGRAF model can be downloaded from [here](https://drive.google.com/file/d/1rn4oQmXJpPwzYbvui4WsXMfjv1TvP1xH/view?usp=sharing)
Modify the **model_path**, **data_path**, **vocab_path** in the `evaluation.py` file. Then run `evaluation.py`:

```bash
python evaluation.py
```

## Training new models from scratch
Modify the **data_path**, **vocab_path**, **model_name**, **logger_name** in the `opts.py` file. Then run `train.py`:



For Flickr30K:

```bash
(For SGR) python train.py --data_name f30k_precomp --num_epochs 80 --lr_update 30 --module_name SGR
```

to train your own word2vec model 
```bash
(For SGR) python vocab.py --w2vec True
```

## Google Colab notebook
A pre-exectued Google Colab notebook of the evaluation and vocabulary can be found in the following link
```bash
https://colab.research.google.com/drive/1tXYCZRuGev_VmlKd-Ptt2orlRKjDodYz?usp=sharing
```
---
**NOTE**

In order to execute the google collab notebook independetly you must create a data folder in your drive and mount it.
for any issues downloading files and models please contact ccrespojulio@ryerson.ca

---