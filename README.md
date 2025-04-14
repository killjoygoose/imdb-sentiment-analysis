# Project overview

This project aims at evaluating the quantized Qwen2.5 SMLs:
[bartowski/Qwen2.5-1.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-1.5B-Instruct-GGUF/blob/main/Qwen2.5-1.5B-Instruct-Q5_K_M.gguf)
and [bartowski/Qwen2.5-0.5B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/blob/main/Qwen2.5-0.5B-Instruct-Q5_K_M.gguf)
on a sentiment analysis task.

This readme contains all information about the project - from exploratory data analysis, through results exploration to dependency installation guidelines.

# Dataset

The dataset provided can be found on [ajaykarthick/imdb-movie-reviews](https://huggingface.co/datasets/ajaykarthick/imdb-movie-reviews). In consists
of 40k examples in the training set and a test set of 10k examples. The first task here was, since we are going to solve the sentiment analysis task
via prompt engineering to compute a suitable subsample size, which to optimize the prompts on. To this end, we need to conduct power analysis in 
the context of paired samples - as we will be comparing pairs of models on the same sample. To calculate the sample size needed for a desired 
significance level and power we apply the following formula (screenshot from the ./notebooks/exploratory_data_analysis.ipynb). [formula](./figs/formula.png). 
We assume a minimum detectable difference in accuracy of 3% at confidence level of .05 and power of .9, the estimated variance we set to .1 - thus the number of 
needed datapoints is estimated to __1122__. We later empirically assert that the accuracy on our development set of 1122 samples correctly estimates the accuracy on the 
full 10k test set.

The dataset consists only of English texts and is fully balances - 20k datapoints for each class as seen here: [balance](./figs/label_balance.png)
Also the dataset is balanced in terms of text length distribution accross the two classes: [balance_in_lengths](./figs/balance_in_lengths.png)
Additionally, one could try to find meaningful clusters among the data to further assure unbiased subsampling. In our case, we quickly experimented with
LDA+k-means on verb-adjective-adv-noun bag-of-words representation of the texts, however, even at the peak of the silhouette scores the distribution
of the clusters was way too skewed and the distribution among the classes was to make any sense to use (i.e. it was almost as if most of the documents belong to the same class). Due to
this mishap, we finally subsampled the above stated number of datapoints, making sure that we preserve the class balance.

# Experiments




