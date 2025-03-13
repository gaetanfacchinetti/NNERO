# NNERO

[![Build Status](https://github.com/gaetanfacchinetti/NNERO/actions/workflows/python-package.yml/badge.svg?branch=torch)](https://github.com/gaetanfacchinetti/NNERO/actions?query=branch%3Atorch)


[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-green.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![Static Badge](https://img.shields.io/badge/physics-cosmology-darkblue)](https://en.wikipedia.org/wiki/Cosmology)
[![Static Badge](https://img.shields.io/badge/physics-21cm-yellow)](https://en.wikipedia.org/wiki/Hydrogen_line)

This is **NNERO** (**N**eural **N**etwork **E**mulator for **R**eionization and **O**ptical depth), a fast adaptative tool to emulate reionization history using a simple neural network architecture. 

The current default networks implemented have been trained on data generated with **[21cmCLAST](https://github.com/gaetanfacchinetti/21cmCLAST)**. 

---
> This package is part of a set of codes which can be combined together to produce forecast or constraints from late-time Universe observables (such as 21cm) on exotic scearios of dark matter and more. Some of these packages are forks of previously existing repositories, some have been written from scratch
- [21cmCLAST](https://github.com/gaetanfacchinetti/21cmCLAST) forked from [21cmFAST](https://github.com/21cmfast/21cmFAST)
- [HYREC-2](https://github.com/gaetanfacchinetti/HYREC-2) forked from [this repository](https://github.com/nanoomlee/HYREC-2)
- [MontePython](https://github.com/gaetanfacchinetti/montepython_public) forked from [this repository](https://github.com/brinckmann/montepython_public)
- [21cmCAST](https://github.com/gaetanfacchinetti/21cmCAST)



## How to install NNERO?

NNERO can be installed using pip with the following command
```bash
pip install nnero
```
For a manual installation or development you can clone this repository and install it with
```bash
git clone https://github.com/gaetanfacchinetti/NNERO.git 
pip install -e .
```

## How to use NNERO?

A detailed documentation is under construction [here](https://gaetanfacchinetti.github.io/docs/NNERO/html/index.html).

### Simple case

NNERO combines two neural networks, a classifier that identifies if a model leads to a reionization that is early enough, and a regressor that predict the evolution of the free-electron fraction and the associated optical depth to reionization.

```python
from nnero import predict_Xe
from nnero import predict_tau

# load classifier and regressor at <path_*>, if no <path> given, the defaults are loaded
classifier=nnero.Classifier.load(<path_c>)
regressor=nnero.Regressor.load(<path_r>)

# print general information
# - structure of the network
# - input parameters name and training range
regressor.info()

# get Xe from loaded classifier and regressor
# **kwargs can be any parameter that is printed calling the info() function above
Xe=predict_Xe(classifier, regressor, **kwargs)
z=regressor.z

# get tau similarly
tau=predict_tau(classifier, regressor, **kwargs)
```


### Run simple MCMC on astrophysical parameters

With NNERO it is possible to run simple MCMC using emcee on the astrophysical packages, combining the UV-Luminosity function likelihood, the likelihood on the reionzation history and a constraint on the optical depth to reionization. A simple example is given below.

```python
import nnero
import emcee

classifier = nnero.Classifier.load()
regressor  = nnero.Regressor.load()

filename = "output.h5"

# varying parameters
p_theta = ['F_STAR10', 'ALPHA_STAR', 'F_ESC10', 'ALPHA_ESC', 'M_TURN', 't_STAR', 'L_X', 'NU_X_THRESH']

# fixed parameters and their value
p_xi = ['Ombh2', 'Omdmh2', 'hlittle', 'Ln_1010_As', 'POWER_INDEX']
xi = np.array([0.0224, 0.12, 0.677, 3.05, 0.965])

# save the parameters used
nnero.analysis.save_sampling_parameters(filename, p_theta, p_xi, xi)

# define the Likelihoods
uv_lkl   = nnero.UVLFLikelihood(parameters=p_theta+p_xi, 
    parameters_xi=p_xi, xi = xi, precompute=True, k = k, pk = pk)
tau_lkl  = nnero.mcmc.OpticalDepthLikelihood(parameters=p_theta+p_xi, 
    classifier=classifier, regressor=regressor, 
    median_tau=0.0557, sigma_tau=np.array([0.0075, 0.0067]))
reio_lkl = nnero.mcmc.ReionizationLikelihood(parameters=p_theta+p_xi, 
    classifier=classifier, regressor=regressor)

# get the name and the range of the parameters on which the NN have been trained
p_range = regressor.parameters_range
p_names = list(regressor.parameters_name)

# definen the prior range for the varying parameters
theta_min  = np.array([p_range[p_names.index(param), 0] for param in p_theta])
theta_max  = np.array([p_range[p_names.index(param), 1] for param in p_theta])

# initialise the walkers ar random values
pos = nnero.initialise_walkers(theta_min, theta_max, xi, 
    likelihoods = [uv_lkl, tau_lkl, reio_lkl],  n_walkers = 32)
nwalkers, ndim = pos.shape

# define the emcee backend
backend = emcee.backends.HDFBackend(filename)

# sample over the distribution
sampler = emcee.EnsembleSampler(nwalkers, ndim, nnero.log_probability, args = (xi, theta_min, theta_max, [tau_lkl, reio_lkl, uv_lkl]), backend=backend, vectorize=True)
sampler.run_mcmc(pos, 200000, progress=True);
```

### Use analysis / plotting tools in NNERO

NNERO has a built-in set of function to perform MCMCs and 
In the example below we show how to plot the result of a MCMC performed with emcee as described above or with MontePython


```python
import nnero

classifier = nnero.Classifier.load()
regressor  = nnero.Regressor.load()

# import the samples from generated files
# assume that both samples are for the p_theta parameters above
samples_MP = nnero.MPSamples('<path>/chains/test/2025-03-12_500000_')
samples_EM = nnero.analysis.EMCEESamples('output.h5', add_tau=True)
parameters = p_theta

# prepare the data for plotting                 
data_MP = nnero.prepare_data_plot(samples_MP, data_to_plot=parameters)
data_EM = nnero.prepare_data_plot(samples_EM, data_to_plot=parameters,
    thin=20, classifier = classifier, regressor = regressor)

# process the data to generate the contours and all statistics
c_data_MP = nnero.generate_contours(data_MP, bins=25, 
    smooth_1D=True, smooth_2D=True, sigma_smooth=1.5)
c_data_EM = nnero.generate_contours(data_EM, bins=25, 
    smooth_1D=True, smooth_2D=True, sigma_smooth=1.5)

# get the labels associated to the parameters
labels = nnero.latex_labels(parameters)

# prepare the grid for the triangle plot
grid = nnero.AxesGrid(c_data.size, labels = labels, names = data, scale=1.4)

# plot the contours on the grid
nnero.plot_data(grid, c_data_MP, show_contour=True, show_points = False)
nnero.plot_data(grid, c_data_EM, show_contour=True, show_points = False)
```

## Contributions

Any comment or contribution to this project is welcome.

## Credits

If you use **NNERO** or the default classifiers / regressor trained using **21cmCLAST** please cite at least one of the following paper that is relevant to your usage:

- G. Facchinetti, *Teaching reionization history to machines:  \\
constraining new physics with early- and late-time probes* (in prep.)
- V. Dandoy, C. Doering, G. Facchinetti, L. Lopez-Honorez, J. R. Schwagereit (in prep.)
- G. Facchinetti, A. Korochkin, L. Lopez-Honorez (in prep.)
