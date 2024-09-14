# This is the official pytorch implementation of "Debiasing the Conversion Rate Prediction Model in the Presence of Delayed Implicit Feedback" paper published in *Entropy*.

We use two public real-world datasets (Coat, Yahoo!) for real-world experiments 

## Run the real-world experiments code

Please refer to real_coat.ipynb, real_yahoo.ipynb  for the results of the corresponding datasets (for **Table 1** in our manuscript).

## Run the sensitivity analysis

Please refer to *real_coat_sen[].ipynb* for the results of the sensitivity analysis. The *real_coat_sen[10, 15, 20, 25, 30].ipynb* generate the results of our proposals and some competing models with different mislabel ratios. The *real_coat_sen20_[weibull, exponential, lognormal].ipynb* generate the results of our proposals and some competing models with different models for delayed feedback. **Figures** 1 and 2 are generated with these results and *figure.ipynb*.


If you find this code useful for your work, please consider to cite our work as

```

```
