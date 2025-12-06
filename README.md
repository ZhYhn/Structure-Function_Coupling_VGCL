# Structure-Function_Coupling_VGCL

This is the code for the bioRxiv preprint: ***[Quantifying Structure-Function Coupling in the Human Brain using Variational Graph Contrastive Learning](https://doi.org/10.1101/2025.10.26.684597)***.

including: 
- Code for calculating structural connectivity matrix and functional connectivity matrix
- Code for all models used in the paper

The data for the paper can be found here: https://doi.org/10.5281/zenodo.17828069

## Dependencies

The core environment for implementing graph neural network is:
- **Python** 3.12.11
- **PyTorch** 2.8.0+cu129
- **PyTorch Geometric (PyG)** 2.6.1

## `step_01_preprocess`
- `compute_fc.py`: Calculate the average time series within the brain region and compute the Pearson correlation coefficient.
- `fslr_to_fsaverage.sh`: Convert the rs-fMRI data from the fs_LR space to the fsaverage space (refer to [HCP Users FAQ](https://wiki.humanconnectome.org/docs/HCP%20Users%20FAQ.html) item 9: *How do I map data between FreeSurfer and HCP?*).
- `generateFC.sh`: Run the scripts `compute_fc.py` and `fslr_to_fsaverage.sh` to generate functional connectivity matrices.
- `generateSC.py`: Convert .mat files to numpy arrays to generate structural connectivity matrices (The structural connectivity matrix in .mat format come from the dataset: https://doi.org/10.5281/zenodo.3928848).

## `step_02_modeling`
- `create_dataset.py`: Perform sparsification and create dataset.
- `model_xxx.py`: Including the implementation of Variational Graph Contrastive Learning (VGCL) model and Predictive GCN (pGCN) model. Other models are used for ablation study.
- `cross_validation_xxx.py`: Perform cross-validation using the corresponding model and output the results.
- `correlation_approach.ipynb`: Calculating structure-function coupling using correlation approach.
- `regression_approach.ipynb`: Calculating structure-function coupling using regression approach.
- `statistical_testing.ipynb`: Including the implementation of mixed-effects model and the calculation of correlation coefficients (Pearson, Spearman, and intraclass correlation coefficient).
