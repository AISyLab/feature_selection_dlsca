## Exploring Feature Selection Scenarios for Deep Learning-based Side-Channel Analysis

This repository contains the source code used to generate the results from the paper: 
"Exploring Feature Scenarios in Deep Learning-based Profiled Side-Channel Analysis".

### Setting up

##### 1. Configure dataset paths
Go to ```experimets/paths.py``` file and set all **dataset path folders** for all feature selection
scenarios (RPOI, OPOI, NOPOI, NOPOI_DESYNC) and for all datasets (ASCADf, ASCADr, DPAV42, CHESCTF).


##### 2. Configure results folder paths
Go to ```experimets/paths.py``` file and set all **result path folders** for all feature selection
scenarios (RPOI, OPOI, NOPOI, NOPOI_DESYNC) and for all datasets (ASCADf, ASCADr, DPAV42, CHESCTF).

##### 3. Download raw datasets

Download ASCADf raw traces: https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_fixed_key
Download ASCADr raw traces: https://github.com/ANSSI-FR/ASCAD/tree/master/ATMEGA_AES_v1/ATM_AES_v1_variable_key
Download DPAV42 raw traces: https://www.dpacontest.org/v4/42_traces.php
Download CHESCTF raw traces: https://zenodo.org/record/3733418#.Yc2iq1ko9Pa

##### 4. Set sys.path.append in scripts (to avoid import issues)

Set the following line:

```sys.path.append('/project_root_folder')```

in the following files:

ASCADf:

```experiments/ASCADf/generate_dataset.py```

```experiments/ASCADf/grid_search_gta_mlp_cnn.py```

```experiments/ASCADf/grid_search_mlp_cnn_no_lda.py```

```experiments/ASCADf/random_search.py```

```experiments/ASCADf/random_search_da.py```

```experiments/ASCADf/test_best_models.py```

ASCADr:

```experiments/ASCADr/generate_dataset.py```

```experiments/ASCADr/grid_search_gta_mlp_cnn.py```

```experiments/ASCADr/grid_search_mlp_cnn_no_lda.py```

```experiments/ASCADr/random_search.py```

```experiments/ASCADr/random_search_da.py```

```experiments/ASCADr/test_best_models.py```

DPAV42:

```experiments/DPAV42/generate_dataset.py```

```experiments/DPAV42/grid_search_gta_mlp_cnn.py```

```experiments/DPAV42/grid_search_mlp_cnn_no_lda.py```

```experiments/DPAV42/random_search.py```

```experiments/DPAV42/random_search_da.py```

```experiments/DPAV42/test_best_models.py```

CHESCTF:

```experiments/CHESCTF/generate_dataset.py```

```experiments/CHESCTF/random_search.py```

```experiments/CHESCTF/random_search_da.py```

```experiments/CHESCTF/test_best_models.py```


##### 5. Generate RPOI, OPOI, NOPOI and NOPOI_DESYNC datasets

To prepare datasets, run following python scripts:

ASCADf: 

```experiments/ASCADf/generate_dataset.py```

ASCADr: 

```experiments/ASCADr/generate_dataset.py```

CHESCTF: 

```experiments/CHESCTF/generate_dataset.py```

DPAV42:

```experiments/DPAV42/convert_to_h5.py```
```experiments/DPAV42/generate_dataset.py```

### Executing random search 

For random search, you have to run ```random_search.py``` (or ```random_search_da.py```, for data augmentation) 
files for each dataset. There are seven parameters to pass with the python file call:

1. Leakage Model: **HW** or **ID**
2. Model type: **mlp** or **cnn** 
3. Feature selection type: **RPOI**, **OPOI**, **NOPOI** or **NOPOI_DESYNC**
4. Number of POIS: (e.g. **700** for ASCADf and OPOI) 
5. Target key byte: **0** to **15**
6. Regularization: **True** or **False**
7. Resampling Window: **10**, **20**, **40** or **80** (for NOPOI and NOPOI_DESYNC. For RPOI and OPOI the value is ignored, but it has to be provided.)

### Testing best models

To test best models, you have to run ```test_best_models.py``` files for each dataset.
There are six parameters to pass with the python file call:

1. Leakage Model: **HW** or **ID**
2. Model type: **mlp** or **cnn** 
3. Feature selection type: **RPOI**, **OPOI**, **NOPOI** or **NOPOI_DESYNC**
4. Number of POIS: (e.g. **700** for ASCADf and OPOI) 
5. Target key byte: **0** to **15**
6. Resampling Window: **10**, **20**, **40** or **80** (for NOPOI and NOPOI_DESYNC. For RPOI and OPOI the value is ignored, but it has to be provided.)

##### Examples:

#### Testing best OPOI models for ASCADf

```python experiments/ASCADf/test_best_models.py HW mlp OPOI 700 2 20```

```python experiments/ASCADf/test_best_models.py ID mlp OPOI 700 2 20```

```python experiments/ASCADf/test_best_models.py HW cnn OPOI 700 2 20```

```python experiments/ASCADf/test_best_models.py ID cnn OPOI 700 2 20```

#### Testing best NOPOI models for ASCADf

To run best found models for **ASCADf**, **key byte 0**, **10000** POIs, **NOPOI** with resampling window of **20** and 
without **regularization:**

```python experiments/ASCADf/test_best_models.py HW mlp 10000 NOPOI 0 20 False```

```python experiments/ASCADf/test_best_models.py ID mlp 10000 NOPOI 0 20 False```

```python experiments/ASCADf/test_best_models.py HW cnn 10000 NOPOI 0 20 False```

```python experiments/ASCADf/test_best_models.py ID cnn 10000 NOPOI 0 20 False```

#### Testing best NOPOI models for ASCADr

To run best found models for **ASCADr**, **key byte 0**, **10000** POIs, **NOPOI** with resampling window of **20** and 
without **regularization:**

```python experiments/ASCADr/test_best_models.py HW mlp 25000 NOPOI 0 20 False```

```python experiments/ASCADr/test_best_models.py ID mlp 25000 NOPOI 0 20 False```

```python experiments/ASCADr/test_best_models.py HW cnn 25000 NOPOI 0 20 False```

```python experiments/ASCADr/test_best_models.py ID cnn 25000 NOPOI 0 20 False```


