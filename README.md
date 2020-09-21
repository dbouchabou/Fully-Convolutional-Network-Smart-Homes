# Fully Convolutional Network Bootstrapped by Word Encoding and Embedding for Activity Recognition in Smart Homes

![Framework Architecture](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/png/fcn_framework.png)

## Code

The code is divided as follows: 
* The [main.py](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/main.py) python file contains the necessary code to run an experiement.
* The [classifiers](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/classifiers) folder contains deep neural network models tested in our paper.
* The [pre_proccessing](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/pre_processing) folder contains scripts to preprocess datasets.

## Data 
The data used in this project comes from Center for Advanced Studies in Adaptive Systems CASAS : 

* The [Aruba](http://casas.wsu.edu/datasets/aruba.zip)
* The [Milan](http://casas.wsu.edu/datasets/milan.zip)

* [datasets](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/datasets) contain a copy of the original datasets and an exemple allready pre-processed with a sliding windows of size 100 and 25.

## Setup

To quickly run a training, extract the pre-processed datasets in datasets folder and see the section "Train" below

## Pre-processing

First step is to read raw dataset pre-process it with [pre_process.py](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/pre_processing/pre_process.py).

* STEP 1: Load dataset
* STEP 2: prepare dataset
* STEP 3: segment dataset in sequence of activity
* STEP 4: transform sequences of activity in sentences
* STEP 5: sentences indexization
* STEP 6: split indexed sentences in sliding windows
* STEP 7: pad sliding windows
* STEP 8: save sliding windows and labels


```
python pre_process.py --i ../datasets/original_datasets/milan/data --o MILAN_activity_sequences_windowed --w 25  
```

Options
* --i input file dataset
* --o output file
* --w windows size

In order to train models on equal bases, uses [subsets_generator.py](https://github.com/dbouchabou/Fully-Convolutional-Network-Smart-Homes/blob/master/pre_processing/subsets_generator.py). This script generate train, validation and test subsets.

```
python subsets_generator.py --i MILAN_activity_sequences_windowed_25_padded --p ../datasets/MILAN    
```

Options
* --i input file dataset
* --p path where subsets will be save

## Train

To run models on one dataset you should issue the following command: 
```
python main.py --i MILAN_activity_sequences_windowed_25_padded --d datasets/pre_processed_datasets/MILAN --c batch_1024_epoch_400_early_stop_mask_0.0 --m FCN_Embedded FCN 
``` 
Options
* --i input file dataset
* --d path to the dataset
* --c comments ex: choosen hyperparameters, batch size, ...
* --m models one ore more


## Results

The following table contains the averaged balanced accuracy and average weighted F1-score over 3 runs of each implemented models on Aruba and Milan datasets from CASAS depends on the sliding windows size used.


|                             |                 |           |ARUBA     |          |          |   |           |MILAN     |          |          |
|-----------------------------|-----------------|-----------|----------|----------|----------|---|-----------|----------|----------|----------|
|**Metrics**                  |**Models**       |**100**    |**75**    |**50**    |**25**    |   |**100**    |**75**    |**50**    |**25**    |
|**Weighted avg F1 Score (%)**|LSTM             |96.67      |94.67     |90.67     |85.00     |   |84.00      |85.67     |75.33     |64.00     |
|                             |FCN              |99.00      |98.00     |97.67     |92.33     |   |77.33      |93.67     |88.33     |83.67     |
|                             |LSTM + Embedding |**100.00** |99.67     |98.00     |90.00     |   |98.00      |97.00     |93.00     |73.67     |
|                             |FCN + Embedding  |**100.00** |**100.00**|**100.00**|**99.00** |   |**99.00**  |**98.00** |**97.00** |**94.33** |
|                             |                 |           |          |          |          |   |           |          |          |          |
|**Balanced Accuracy (%)**    |LSTM             |81.45      |76.09     |71.05     |83.30     |   |62.15      |64.95     |55.70     |43.29     |
|                             |FCN              |88.85      |87.41     |87.08     |80.32     |   |42.24      |76.41     |71.82     |71.34     |
|                             |LSTM + Embedding |94.55      |93.61     |90.20     |74.81     |   |**88.52**  |**86.77** |82.05     |59.35     |
|                             |FCN + Embedding  |**95.37**  |**95.07** |**94.89** |**92.44** |   |84.23      |86.64     |**87.83** |**90.86** |

## Reference

If you re-use this work, please cite:

```
@article{dbouchabou2020fcnsmarhomehar,
  Title                    = {Fully Convolutional Network Bootstrapped by Word Encoding and Embedding for Activity Recognition in Smart Homes},
  Author                   = {Damien, Sao Mai, Christophe, Benoit, Ioannis},
  journal                  = {},
  Year                     = {2020},
  volume                   = {},
  number                   = {},
  pages                    = {},
}
```
