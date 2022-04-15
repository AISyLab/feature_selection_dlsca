### Neural networks for different feature selection scenarios for different AES128 datasets

This repository contains the source code used to generate the results from the paper: 
"Exploring Feature Scenarios in Deep Learning-based Profiled Side-Channel Analysis".

To run experiments, the user needs to set the dataset paths in:

```experimets/ASCADf/paths.py```

```experimets/ASCADr/paths.py```

```experimets/CHESCTF/paths.py```

```experimets/DPAV42/paths.py```

Ex: to run best found models for ASCADf, key byte 0, NOPOI with resampling window of 20 
(without desynchronization - False):

```python experiments/ASCADf/test_best_models.py HW mlp NOPOI 0 20 False```

```python experiments/ASCADf/test_best_models.py ID mlp NOPOI 0 20 False```

```python experiments/ASCADf/test_best_models.py HW cnn NOPOI 0 20 False```

```python experiments/ASCADf/test_best_models.py ID cnn NOPOI 0 20 False```


