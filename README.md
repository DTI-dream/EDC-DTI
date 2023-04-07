![](C:\Users\12624\Desktop\EDC-DTI\assets\logo.png)

# EDC-DTI

[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## An end-to-end deep collaborative learning model based on multiple information for drug-target interactions prediction.

### 1.Introduction🦄

In this study, we propose an **end-to-end** deep **collaborative** learning model for DTI prediction, called EDC-DTI, designed to identify new targets for existing drugs by leveraging multiple drug-target-related information types, **including homogeneous and heterogeneous data**, through deep learning techniques.

Our end-to-end model comprises **a feature builder and a classifier**. The feature builder consists of two collaborative feature construction algorithms that extract molecular properties and network topology properties. The classifier is composed of a feature **encoder** and a feature **decoder**, which are designed for feature integration and DTI prediction, respectively. The feature encoder, primarily based on the **improved graph attention network (GAT)**, integrates heterogeneous information into drug features and target features separately. The feature decoder consists of multiple neural networks for making predictions.

### 2.code overview🤖

#### code structure

* Data
* src
  * core: The core implements for EDC-DTI based on encoder decoder and collaborative calculation.
  * configs:Hyperparameters used to train the model
  * dti_dataset:Feature fusion and processing before input to the encoder
  * loss_curve/utils: utils for training
  * main_ran: Enable EDC-DTI structure
  * model_trainer:Enable training EDC-DTI
  * train: Training EDC-DTI on different datasets for DEMO

### 3.Train in our Datasets🚀

* First you need to clone the project locally, or you can run it using google colab

  ```shell
  git clone https://github.com/DTI-dream/EDC-DTI.git
  ```

* After cloning to local, install the requirements

  ```shell
  pip install -r requirements.txt
  ```

* You can choose to run our model by simply running

  ```shell
  python train.py
  ```

### 4.Train in your Datasets

Our model is based on a variety of heterogeneous information, please be prepared to process good drug features, disease features and protein features, and modify the following parameters for the location of the data, if you need to get the preprocessing code from raw chemical information to features, please contact neound986@gmail.com.

```shell
python train.py --drug_feature_path "your drug feature path" \
                           --protein_feature_path "your protein feature path" \
                           --disease_feature_path "your disease feature path" \
                           --set_path "your binary data path"
```

**Attention:** In order for the model to work properly, set the correct feature dimensions.



