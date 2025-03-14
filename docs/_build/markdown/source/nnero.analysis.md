# Analysis

### *class* nnero.analysis.AxesGrid(n: int, \*, scale: float = 2.0, wspace: float = 0.05, hspace: float = 0.05, labels: list[str] | None = None, names: list[str] | None = None, \*\*kwargs)

Bases: `object`

#### add_title(axis: int, new_titles: str, color: str | None = None)

#### get(i: int | str, j: int | str)

#### index_1D(i: int, j: int)

#### index_from_name(name: str | list[str])

#### indices_2D(k)

#### set_label(i: int, name: str)

#### show()

#### update_edges(axis: int | str, min: float, max: float)

#### update_labels(\*\*kwargs)

#### update_titles(height=1.05, spacing=1.9, fontsize=None)

### *class* nnero.analysis.EMCEESamples(path: str, add_tau: bool = False)

Bases: [`Samples`](#nnero.analysis.Samples)

Daughter class of Samples for emcee chains.

#### convergence()

#### flat(discard: ndarray | None = None, thin: None | int = None, \*\*kwargs) → ndarray

#### load_chains()

### *class* nnero.analysis.GaussianInfo(mean: numpy.ndarray | None = None, cov: numpy.ndarray | None = None, param_names: numpy.ndarray | list[str] | None = None)

Bases: `object`

#### compatible_with(other: Self) → bool

#### cov *: ndarray | None* *= None*

#### mean *: ndarray | None* *= None*

#### param_names *: ndarray | list[str] | None* *= None*

### *class* nnero.analysis.GaussianSamples(gaussians: list[[GaussianInfo](#nnero.analysis.GaussianInfo) | str] | [GaussianInfo](#nnero.analysis.GaussianInfo) | str, add_tau: bool = False, params: list[str] | None = None, \*, n: int = 200000)

Bases: [`Samples`](#nnero.analysis.Samples)

Daughter class of Samples for gaussian generated chains.

#### flat(discard: ndarray | None = None, thin: None | int = None, \*\*kwargs) → ndarray

#### load_data(filename) → [GaussianInfo](#nnero.analysis.GaussianInfo)

### *class* nnero.analysis.MPChain(filename: str)

Bases: `object`

Class MPChain reading chains from MontePython output files

* **Parameters:**
  **filename** (*str*) – Path to the the file where the chain is stored.

#### compute_stats() → None

Compute the mean and standard deviation within the chain.
Should be called after remove_burnin().

#### load() → None

Load the chain and automatically remove the non-markovian points.

#### remove_burnin(global_max_lnlkl: float) → None

Remove the burnin points according to the value of the maximum
log likelihood over all chains. Only points of the chain that are
after its overcrossing of global_max_lnlkl - 3 are kept.

* **Parameters:**
  **global_max_lnlkl** (*float*) – Global maximum log likelihood over all chains

#### values(discard: int = 0, thin: int = 1) → ndarray

Get the values of the chain.

* **Parameters:**
  * **discard** (*int* *,* *optional*) – Number of initial points to discard, by default 0.
  * **thin** (*int* *,* *optional*) – Thining factor (taking only one value every value of thin), by default 1.
* **Return type:**
  np.ndarray with dimension (number of parameters, length of chain)

### *class* nnero.analysis.MPSamples(path: str, ids: list[int] | ndarray | None = None)

Bases: [`Samples`](#nnero.analysis.Samples)

Daughter class of Samples for MontePython chains.

#### convergence() → ndarray

Gelman-Rubin criterion weighted by the length of the chain as implemented
in MontePython.

* **Returns:**
  R-1 for all parameters
* **Return type:**
  np.ndarray

#### covmat(discard: ndarray | None = None, params_in_cov: list[str] | None = None) → ndarray

Covariance matrix.

* **Parameters:**
  * **discard** (*np.ndarray* *|* *None* *,* *optional*) – Number of points to discard at begining of the chain, by default None (0).
  * **data_to_cov** (*list* *[**str* *]*  *|* *None* *,* *optional*) – List of parameters to put in the covariance matrix (in the order of that list).
    If None consider all parameters available.
* **Returns:**
  Covariance matric (n, n) array
* **Return type:**
  np.ndarray

#### flat(discard: ndarray | None = None, thin: None | int = None, \*\*kwargs) → ndarray

Flatten samples of all chains.

* **Parameters:**
  * **discard** (*np.ndarray* *|* *None* *,* *optional*) – Number of points to discard at begining of the chain, by default None (0).
  * **thin** (*None* *|* *int* *,* *optional*) – Reduce the size of the sample by taking one point every thin, by default None.
    If Nont compute the reduced size such that the total length of the sample is 10000.
* **Returns:**
  Sample in a 2 dimensional array of shape (# of parameters, # of points)
* **Return type:**
  np.ndarray

#### load_chains() → None

#### load_paramnames()

#### load_scaling_factor() → None

#### print_best_fit(discard: ndarray | None = None, \*\*kwargs)

### *class* nnero.analysis.ProcessedData(hists_1D: numpy.ndarray | None = None, hists_2D: numpy.ndarray | None = None, edges: numpy.ndarray | None = None, centers: numpy.ndarray | None = None, levels: numpy.ndarray | None = None, q: numpy.ndarray | None = None, mean: numpy.ndarray | None = None, median: numpy.ndarray | None = None, bestfit: numpy.ndarray | None = None, quantiles: numpy.ndarray | None = None, samples: numpy.ndarray | None = None, size: int | None = None)

Bases: `object`

#### bestfit *: ndarray | None* *= None*

#### centers *: ndarray | None* *= None*

#### edges *: ndarray | None* *= None*

#### hists_1D *: ndarray | None* *= None*

#### hists_2D *: ndarray | None* *= None*

#### levels *: ndarray | None* *= None*

#### mean *: ndarray | None* *= None*

#### median *: ndarray | None* *= None*

#### q *: ndarray | None* *= None*

#### quantiles *: ndarray | None* *= None*

#### samples *: ndarray | None* *= None*

#### size *: int | None* *= None*

### *class* nnero.analysis.Samples(path: str, ids: list[int] | ndarray | None = None)

Bases: `ABC`

Class containing all chains of a MCMC analysis

* **Parameters:**
  * **path** (*str*) – Path to the chains.
  * **ids** (*list* *|* *np.ndarray* *|* *None* *,* *optional*) – List of chains to take into accoung. If none all possible found chains are added. By default None.

#### *abstract* flat(discard: ndarray | None = None, thin: None | int = None, \*\*kwargs) → ndarray

### nnero.analysis.compute_parameter(flat_chain: ndarray, param_names: list[str] | ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, interpolator: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, parameter: str | None = None) → ndarray

### nnero.analysis.compute_quantiles(sample: ndarray, q: float, bins: int | ndarray = 30) → tuple[ndarray, ndarray]

Give the q-th quantile of the input sample.

* **Parameters:**
  * **sample** (*np.ndarray*) – 1D array of data points.
  * **q** (*float*) – Quantile value.
  * **bins** (*int* *|* *np.ndarray* *,* *optional*) – Binning of the histogram that is used
    for a first approximation of the quantile
    edges, by default 30.
* **Returns:**
  * *tuple[np.ndarray, np.ndarray]*
  * *min,max bounds*

### nnero.analysis.compute_tau(flat_chain: ndarray, param_names: list[str] | ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None) → ndarray

### nnero.analysis.generate_contours(samples: ndarray, bins: int = 20, q=[0.68, 0.95], smooth_2D: bool = False, smooth_1D: bool = False, sigma_smooth: float = 1.5) → [ProcessedData](#nnero.analysis.ProcessedData)

### nnero.analysis.get_Xe_stats(samples: [Samples](#nnero.analysis.Samples), data_to_plot: list[str] | ndarray, nbins: int = 100, discard: int = 0, thin: int = 100, \*, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, smooth: bool = False, sigma_smooth: float = 1.5, \*\*kwargs)

### nnero.analysis.get_Xe_tanh_stats(samples: [Samples](#nnero.analysis.Samples), nbins: int = 100, discard: int = 0, thin: int = 100, x_inf: float = 0.0002, \*, smooth: bool = False, sigma_smooth: float = 1.5, \*\*kwargs)

### nnero.analysis.neutrino_masses(mnu_1, mnu_2=0.0, mnu_3=0.0, hierarchy='NORMAL')

### nnero.analysis.plot_2D_marginal(ax: Axes, data: [ProcessedData](#nnero.analysis.ProcessedData), i: int, j: int, show_hist: bool = False, show_surface: bool = True, show_contour: bool = False, show_points: bool = False, colors: list[str] = 'orange', alphas: list[float] = 1.0)

### nnero.analysis.plot_data(grid: [AxesGrid](#nnero.analysis.AxesGrid), data: [ProcessedData](#nnero.analysis.ProcessedData), show_hist: bool = False, show_surface: bool = True, show_contour: bool = False, show_quantiles: list[bool] = [False, False], show_mean: bool = False, show_title: bool = True, show_points: bool = False, redefine_edges: bool = True, q_in_title: int = 0.68, colors: list[str] = 'orange', axes: list[int] | ndarray | None = None, exclude_quantiles: int | str | list[int] | list[str] = [], exclude_mean: int | str | list[int] | list[str] = [], exclude_title: int | str | list[int] | list[str] = [], alphas: list[float] = 1.0)

### nnero.analysis.prepare_data_Xe(samples: [Samples](#nnero.analysis.Samples), data_to_plot: list[str] | ndarray, discard: int = 0, thin: int = 100, \*, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None)

### nnero.analysis.prepare_data_plot(samples: [Samples](#nnero.analysis.Samples), data_to_plot, discard=0, thin=1, \*\*kwargs)

### nnero.analysis.save_sampling_parameters(filename: str, parameters_theta: list[str], parameters_xi: list[str], xi: ndarray)

### nnero.analysis.to_21cmFAST_names(array: list[str] | ndarray)

### nnero.analysis.to_CLASS_names(array: list[str] | ndarray)
