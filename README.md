# Data-Driven EEG Band Discovery with Decision Trees

Software implementation of the EEG band discovery method described in the (unpusblished) paper: Data-Driven EEG Band Discovery with Decision Trees. The method determinines the best EEG bands for a given dataset using a two-part approach. First, a decision tree-based technique is used to estimate the optimal frequency band boundaries for reproducing the signal's power spectrum for a every possible number of bands. Then, the optimal number of bands is determined using an AIC-inspired quality score that balances goodness-of-fit with a small band count.

## Getting Started
To get started clone this repo and install the requirements via the following command.
```
pip install -r requirements.txt
```

To reproduce the case study figures in the paper run:
```
python main.py
```

To use the method on your own data use the getOptimalBands() function in getOptimalBands.py. Example code is available in the example.py script.


## Decision Tree Binning
Visual overview of how decision trees can be used to derive frequnecy bands. A single predictor variable (frequency) to estimate a single target variable (natural logarithm of the power spectral density). The decision tree then splits frequency values into subgroups and assigns each subgroup a single target value prediction. A greedy search of the decision tree parameter space yields frequency splits that best reproduce target values. Thus, through this optimization process we automatically obtain the optimal member-adjacent frequency bands for a predefined number of bands.

![Overview](/visuals/other_figures/method_overview.jpeg)

## How to cite

If you find value in this work please use the following citation: 

`Talebi S., et al. decisiontreeBinning. 2022. https://github.com/mi3nts/decisiontreeBinning`

__Bibtex__:
```
@misc{decisiontreeBinning,
authors={Shawhin Talebi, John Waczak, Bharana Fernando, Arjun Sridhar, David J. Lary},
title={decisiontreeBinning},
howpublished={https://github.com/mi3nts/decisiontreeBinning}
year={2022}
}
```
