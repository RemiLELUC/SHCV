# Sliced-Wasserstein Estimation with Spherical Harmonic as Control Variates (ICML2024)

This is the code associated to the ICML2024 paper: "Sliced-Wasserstein Estimation with Spherical Harmonic as Control Variates", Rémi Leluc, Aymeric Dieuleveut, François Portier, Johan Segers and Aigerim Zhuman.

> @inproceedings{
leluc2024slicedwasserstein,
title={Sliced-Wasserstein Estimation with Spherical Harmonics as Control Variates},
author={Leluc, R{\'e}mi and Dieuleveut, Aymeric and Portier, Fran{\c{c}}ois and Segers, Johan and Zhuman, Aigerim},
booktitle={Proceedings of the 41th International Conference on Machine Learning},
year={2024}
}
> 

## Description 

The different folders contain the code of the experimental results, the code is written in Python3
and relies on the github repository (https://github.com/vdutor/SphericalHarmonics) to build the spherical harmonics. Please follow the installation instructions of this repository to be able to build the matrix of control variates.

## Folder code/

- requirements.txt : python dependencies
- results/         : files .npy of the results of the experiments
- scripts/         : python scripts to perform experiments 
- graphs/          : figures of the paper

All the different Sliced-Wasserstein estimates are implemented in the script 'sliced_wasserstein.py'.

- To reproduce the results for the synthetic gaussian experiments, run 'python synthetic_gaussian.py' in the folder 'scripts/' and change the parameter of dimension d=10 and number of samples n_samples=1000 to the setting you want.
- To reproduce the results for the 3D point clouds experiments, run 'python 3D_point_cloud.py' in the folder 'scripts/' and change the parameter 'ind_source=8' and 'ind_target=32' to select the 3D point cloud you want.
- To compute the Kernel matrix for the SVM experiments, run 'python build_kernel.py' in the folder 'scripts/'.

- To reproduce the figures for the synthetic gaussian experiments, run "python figure_gaussian.py --d=3 --gauss_type='exact' " in the folder 'scripts/' and change to --d=6 or gauss_type='sampled' accordingly to plot the Figures in dimension d=6 or to change the exact gaussian case to sampled case with empirical distributions.
- To reproduce the figures for the 3D point clouds experiments, just run 'python figure_3D_point_cloud.py' in the folder 'scripts/'

