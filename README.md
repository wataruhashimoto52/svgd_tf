# Stein Variational Gradient Descent

## Description
Implementation of Stein Variational Gradient Descent with TensorFlow 2.0

  
## Requirements
* `tensorflow>=2.0.0-rc0`
* `tensorflow-probability>=0.8.0-rc0`


## Contents
* `gaussian_mixture.py` - Transformation initial samples to Gaussian mixture using SVGD.
* `bayesian_logistic_regression.py` - Bayesian Logistic Regression with SVGD.
* `tfp_bayesian_logistic_regression.py` - Bayesian Logistic Regression with Mean-Field Variational Inference.


## Reference
* Q. Liu, and D. Wang, "Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm", in NIPS, 2017.  https://papers.nips.cc/paper/6338-stein-variational-gradient-descent-a-general-purpose-bayesian-inference-algorithm 