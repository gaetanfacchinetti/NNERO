# MCMC

### *class* nnero.mcmc.Likelihood(parameters: list[str])

Bases: `ABC`

#### get_x(theta: ndarray, xi: ndarray) → ndarray

#### get_x_dict(x: ndarray) → dict

#### get_x_dicts(x: ndarray) → list[dict]

#### loglkl(theta: ndarray, xi: ndarray, \*\*kwargs) → ndarray

### *class* nnero.mcmc.OpticalDepthLikelihood(parameters: list[str], \*, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier), regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor), median_tau: float = 0.0544, sigma_tau: float = 0.0073)

Bases: [`Likelihood`](#nnero.mcmc.Likelihood)

### *class* nnero.mcmc.ReionizationLikelihood(parameters: list[str], \*, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier), regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor))

Bases: [`Likelihood`](#nnero.mcmc.Likelihood)

### *class* nnero.mcmc.UVLFLikelihood(parameters: list[str], \*, parameters_xi: ndarray | None = None, xi: ndarray | None = None, k: ndarray | None = None, pk: ndarray | None = None, precompute: bool = False)

Bases: [`Likelihood`](#nnero.mcmc.Likelihood)

Likelihood for the UV luminosity functions.

* **Parameters:**
  * **k** (*np.ndarray*) – Array of modes on which the matter power spectrum is given (in 1/Mpc).
  * **pk** (*np.ndarray*) – Matter power spectrum.

#### get_k_max(x) → None | float

### nnero.mcmc.initialise_walkers(theta_min: ndarray, theta_max: ndarray, xi: ndarray, likelihoods: list[[Likelihood](#nnero.mcmc.Likelihood)], n_walkers: int = 64, \*\*kwargs)

### nnero.mcmc.log_likelihood(theta: ndarray, xi: ndarray, likelihoods: list[[Likelihood](#nnero.mcmc.Likelihood)], \*\*kwargs) → ndarray

Compute the log Likelihood values.

* **Parameters:**
  * **theta** (*np.ndarray*) – Varying parameters.
  * **xi** (*np.ndarray*) – Extra fixed parameters.
  * **likelihoods** (*list* *[*[*Likelihood*](#nnero.mcmc.Likelihood) *]*) – The likelihoods to evaluate for the fit.
* **Returns:**
  Values of the log Likelihood for each chain.
* **Return type:**
  np.ndarray

### nnero.mcmc.log_prior(theta: ndarray, theta_min: ndarray, theta_max: ndarray, \*\*kwargs) → ndarray

Natural logarithm of the prior

assume flat prior except for the parameters for which
a covariance matrix and average value are given

## Parameters:

- theta: (n, d) ndarray
  : parameters
    d is the dimension of the vector parameter
    n is the number of vector parameter treated at once
- theta_min: (d) ndarray
  : minimum value of the parameters allowed
- theta_max:
  : maximum value of the parameters allowed

## kwargs:

- mask: optional, (d) ndarray
  : where the covariance matrix applies
    the mask should have p Trues and d-p False
    with p the dimension of the covariance matrix
    if cov and my given with dim d then mask still optional
- mu: optional, (p) ndarray
  : average value of the gaussian distribution
- cov: optional, (p, p) ndarray
  : covariance matrix

### nnero.mcmc.log_probability(theta: ndarray, xi: ndarray, theta_min: ndarray, theta_max: ndarray, likelihoods: list[[Likelihood](#nnero.mcmc.Likelihood)], \*\*kwargs) → ndarray
