# Dataset
|name | domain | instances |nominal	|numeric|labels|cardinality	|density|distinct|
| ------ | ------ | ------ |------ |------ |------ |------ |------ |------ |
| yeast| biology | 2417	 |0|103	|14|4.237|0.303	|198|

# Performance
| evaluate           | paper        | this code    |
| ------------------ | ----------- | ------------- |
|  **image data**              |             |               |
| One-error          | 0.242±0.033 | 0.2490±0.0310 |
| Hamming loss       | 0.144±0.012 | 0.1487±0.0120 |
| Coverage           | 0.156±0.016 | 0.1603±0.0083 |
| Ranking loss       | 0.128±0.013 | 0.1324±0.0133 |
| Average precision  | 0.843±0.018 | 0.8374±0.0172 |
| Macro-averaging F1 | 0.660±0.030 | 0.6528±0.0281 |
| Micro-averaging F1 | 0.659±0.031 | 0.6495±0.0285 |
|       **yeast data**             |             |               |
| One-error          | 0.218±0.027 | 0.2321±0.0259 |
| Hamming loss       | 0.190±0.005 | 0.1926±0.0074 |
| Coverage           | 0.446±0.010 | 0.4475±0.0129 |
| Ranking loss       | 0.162±0.007 | 0.1639±0.0089 |
| Average precision  | 0.775±0.013 | 0.7698±0.0159 |
| Macro-averaging F1 | 0.411±0.018 | 0.4041±0.0167 |
| Micro-averaging F1 | 0.655±0.010 | 0.6503±0.0167 |
|       **scene data**             |             |               |
| One-error          | 0.175±0.027 | 0.1803±0.0121 |
| Hamming loss       | 0.072±0.009 | 0.0755±0.0038 |
| Coverage           | 0.062±0.006 | 0.0654±0.0042 |
| Ranking loss       | 0.058±0.005 | 0.0608±0.0043 |
| Average precision  | 0.897±0.012 | 0.8926±0.0070 |
| Macro-averaging F1 | 0.787±0.023 | 0.7712±0.0115 |
| Micro-averaging F1 | 0.780±0.026 | 0.7642±0.0141 |
|        **enron data**            |             |               |
| One-error          | 0.207±0.038 | 0.2203±0.0127 |
| Hamming loss       | 0.045±0.003 | 0.0456±0.0009 |
| Coverage           | 0.239±0.028 | 0.2489±0.0131 |
| Ranking loss       | 0.079±0.028 | 0.0837±0.0060 |
| Average precision  | 0.718±0.025 | 0.7115±0.0066 |
| Macro-averaging F1 | 0.325±0.044 | 0.2479±0.0185 |
| Micro-averaging F1 | 0.580±0.023 | 0.5705±0.0065 |

# Parameter
- lamda for get label correlation matrix : $\frac { 1 } { 100 } \left\| \mathbf { Y } _ { j } ^ { \top } \mathbf { Y } _ { - j } \right\| _ { \infty }$
- alpha for CAMEL: [0, 0.1, 0.2,...,1]
- alpha ban for ADMM: [0, 0.1, 0.2,...,1]
- lamda1: 1
- lamda2: [0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 1]
- rho(ADMM the augmented lagrangian parameter): not mention in the paper, set 1
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
the author's code

# Reference
Collaboration based Multi-Label Learning 2019 AAAI