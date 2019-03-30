# Dataset
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

# Performance

# Parameter
- lamda for get label correlation matrix : $\frac { 1 } { 100 } \left\| \mathbf { Y } _ { j } ^ { \top } \mathbf { Y } _ { - j } \right\| _ { \infty }$
- alpha: [0, 0.1, 0.2,...,1]
- lamda1: 1
- lamda2: [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1]
- rho(ADMM the augmented lagrangian parameter): not mention in the paper
- 5 fold cross validation

# Requrements
- scikit-learn 0.19.1
- numpy 1.16.2
- cupy-cuda100 5.3.0

# Usage
- prepare data
- run main

# Thanks
ADMM tutorial(The alternating direction method of multipliers)交替方向乘子法 [https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html](https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html)

# Reference
Collaboration based Multi-Label Learning 2019 AAAI