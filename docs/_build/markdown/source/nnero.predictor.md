# Predictor

### nnero.predictor.check_values(vals, metadata: [MetaData](nnero.data.md#nnero.data.MetaData))

### nnero.predictor.input_values(metadata: [MetaData](nnero.data.md#nnero.data.MetaData), default: str = {'ALPHA_ESC': 0.3, 'ALPHA_STAR': 0.5, 'FRAC_WDM': 0.0, 'F_ESC10': -1.0, 'F_STAR10': -1.5, 'INVERSE_M_WDM': 0.0, 'LOG10_PMF_SB': -5.0, 'L_X': 40.0, 'Ln_1010_As': 3.047, 'M_TURN': 8.7, 'M_WDM': '20.0', 'NEUTRINO_MASS_1': 0.0, 'NU_X_THRESH': 500, 'Ombh2': 0.02242, 'Omdmh2': 0.11933, 'PMF_NB': -2.0, 'POWER_INDEX': 0.9665, 'hlittle': 0.6736, 't_STAR': 0.5}, \*\*kwargs)

### nnero.predictor.predict_Xe(classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, default: dict | None = None, \*\*kwargs) → ndarray | bool

Prediction of the free electron fraction X_e, taking into account
the selection from the classifier. See predict_Xe_numpy for a more
advanced but faster method (working with numpy arrays).

* **Parameters:**
  * **classifier** ([*Classifier*](nnero.classifier.md#nnero.classifier.Classifier) *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **regressor** ([*Regressor*](nnero.regressor.md#nnero.regressor.Regressor) *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
  * **default** (*dict* *|* *None* *,* *optional*) – Dictionnary of default values, by default None.
    The default dictionnaly should have keys corresponding to those of the trained dataset
    as accessible from regressor.info() or classifier.info().
  * **\*\*kwargs** – Any value for a parameter the classifier and regressor have been trained on.
    Keys should corresponds to regressor.parameters_name or classifier.parameters_name.
* **Returns:**
  Returns False if the classifier returns False. Returns the prediction
  from the regressor for the free electron fraction X_e otherwise.
* **Return type:**
  np.ndarray | bool

### nnero.predictor.predict_Xe_numpy(theta: ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None) → ndarray

Prediction of the free electron fraction X_e, taking into account
the selection from the classifier. Same as predict_Xe but much faster
as it work with numpy arrays.

* **Parameters:**
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    regressor for training. These parameters and their order are accesible
    calling regressor.info().
  * **classifier** ([*Classifier*](nnero.classifier.md#nnero.classifier.Classifier) *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **regressor** ([*Regressor*](nnero.regressor.md#nnero.regressor.Regressor) *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
* **Returns:**
  2D array of shape (n, q) for the values for X_e
  with q the number for parameters.
  When the classifier outputs false, the array is filled with -1.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_classifier(classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, default: dict | None = None, \*\*kwargs) → bool

Prediction of the classifier. See predict_classifier_numpy for a more
advanced but faster method (working with numpy arrays).

* **Parameters:**
  * **classifier** (*nnero.Classifier* *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **\*\*kwargs** – Any value for a parameter the classifier has been trained on.
    Keys should corresponds to classifier.parameters_name.
* **Returns:**
  True for an early reionization.
  False for a late reionization.
* **Return type:**
  bool

### nnero.predictor.predict_classifier_numpy(theta: ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None) → ndarray

Prediction of the classifier. Same as predict_classifier but much faster
as it work with numpy arrays.

* **Parameters:**
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    classifier for training. These parameters and their order are accesible
    calling classifier.info().
  * **classifier** (*nnero.Classifier* *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
* **Returns:**
  Array of booleans.
  True for an early reionization.
  False for a late reionization.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_interpolator(interpolator: Interpolator | None = None, default: dict | None = None, parameter: str | None = None, \*\*kwargs) → ndarray

Prediction of the interpolator. See predict_interpolator_numpy for a more
advanced but faster method (working with numpy arrays).

* **Parameters:**
  * **interpolator** (*nnero.Interpolator* *|* *None* *,* *optional*) – Interpolator object already trained. Default is None. If None, the default
    interpolator DefaultInterpolator_<parameter> is used.
  * **default** (*dict* *|* *None* *,* *optional*) – Dictionnary of default values, by default None.
    The default dictionnaly should have keys corresponding to those of the trained dataset
    as accessible from interpolator.info().
  * **parameter** (*str* *|* *None* *,* *optional*) – If no interpolator given in input, must be specified to know which interpolator to load.
    Default is None.
  * **\*\*kwargs** – Any value for a parameter the interpolator has been trained on.
    Keys should corresponds to interpolator.parameters_name.
* **Returns:**
  Interpolated value.
* **Return type:**
  float
* **Raises:**
  **ValueError** – If no interpolator given, need to specify on which parameter we cant to interpolate
      in order to load the associated default interpolator.

### nnero.predictor.predict_interpolator_numpy(theta: ndarray, interpolator: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, parameter: str | None = None) → ndarray

Prediction of the interpolator. Same as predict_interpolator but much faster
as it work with numpy arrays.

* **Parameters:**
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    regressor for training. These parameters and their order are accesible
    calling regressor.info().
  * **interpolator** (*nnero.Interpolator* *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
  * **parameter** (*str* *|* *None* *,* *optional*) – If no interpolator given in input, must be specified to know which interpolator to load.
    Default is None.
* **Returns:**
  1D array of shape n for the inteprolated values.
* **Return type:**
  np.ndarray
* **Raises:**
  **ValueError** – If no interpolator given, need to specify on which parameter we cant to interpolate
      in order to load the associated default interpolator.

### nnero.predictor.predict_parameter(classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, interpolator: Interpolator | None = None, default: dict | None = None, parameter: str | None = None, \*\*kwargs) → ndarray | bool

Prediction of the interpolated parameter, taking into account
the selection from the classifier. See predict_parameter_numpy for a more
advanced but faster method (working with numpy arrays).

* **Parameters:**
  * **classifier** ([*Classifier*](nnero.classifier.md#nnero.classifier.Classifier) *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **interpolator** (*Interpolator* *|* *None* *,* *optional*) – Interpolator object already trained. Default is None. If None, the default
    interpolator DefaultInterpolator_<parameter> is used.
  * **default** (*dict* *|* *None* *,* *optional*) – Dictionnary of default values, by default None.
    The default dictionnaly should have keys corresponding to those of the trained dataset
    as accessible from interpolator.info() or classifier.info().
  * **\*\*kwargs** – Any value for a parameter the classifier and interpolator have been trained on.
    Keys should corresponds to regressor.parameters_name or classifier.parameters_name.
* **Returns:**
  Returns False if the classifier returns False. Returns the prediction
  from the interpolator for the parameter otherwise.
* **Return type:**
  np.ndarray | bool

### nnero.predictor.predict_parameter_numpy(theta: ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, interpolator: Interpolator | None = None, parameter: str | None = None) → ndarray

Prediction of the interpolated parameter, taking into account
the selection from the classifier. Same as predict_parameter but much faster
as it work with numpy arrays.

* **Parameters:**
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    regressor for training. These parameters and their order are accesible
    calling regressor.info().
  * **classifier** ([*Classifier*](nnero.classifier.md#nnero.classifier.Classifier) *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **interpolator** (*Interpolator* *|* *None* *,* *optional*) – Interpolator object already trained. Default is None. If None, the default
    interpolator DefaultInterpolator is used.
* **Returns:**
  1D array of shape (n) for the value of the parameter
  When the classifier outputs false, the array is filled with -1.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_regressor(regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, default: dict | None = None, \*\*kwargs) → ndarray

Prediction of the regressor. See predict_regressor_numpy for a more
advanced but faster method (working with numpy arrays).

* **Parameters:**
  * **regressor** (*nnero.Regressor* *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
  * **default** (*dict* *|* *None* *,* *optional*) – Dictionnary of default values, by default None.
    The default dictionnaly should have keys corresponding to those of the trained dataset
    as accessible from regressor.info().
  * **\*\*kwargs** – Any value for a parameter the regressor has been trained on.
    Keys should corresponds to regressor.parameters_name.
* **Returns:**
  Array of values for X_e.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_regressor_numpy(theta: ndarray, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None) → ndarray

Prediction of the regressor. Same as predict_regressor but much faster
as it work with numpy arrays.

* **Parameters:**
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    regressor for training. These parameters and their order are accesible
    calling regressor.info().
  * **regressor** (*nnero.Regressor* *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
* **Returns:**
  2D array of shape (n, q) for the values for X_e
  wit q the number of redshift bins.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_tau(classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, default: dict | None = None, \*\*kwargs) → float

Predict the optical depth to reionization from a trained classifier and regressor
as well as parameters passed in kwargs or default. See predict_tau_numpy
for a more advanced but faster method (working with numpy arrays).

* **Parameters:**
  * **classifier** ([*Classifier*](nnero.classifier.md#nnero.classifier.Classifier) *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **regressor** ([*Regressor*](nnero.regressor.md#nnero.regressor.Regressor) *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
  * **default** (*dict* *|* *None* *,* *optional*) – Dictionnary of default values, by default None.
    The default dictionnaly should have keys corresponding to those of the trained dataset
    as accessible from regressor.info() or classifier.info().
  * **\*\*kwargs** – Any value for a parameter the classifier and regressor have been trained on.
    Keys should corresponds to regressor.parameters_name or classifier.parameters_name.
* **Return type:**
  float

### nnero.predictor.predict_tau_from_Xe(xe: ndarray, metadata: [MetaData](nnero.data.md#nnero.data.MetaData), default: dict | None = None, \*\*kwargs) → float

Predict the optical depth to reionization from an array of X_e.
See predict_tau_from_Xe_numpy for a more advanced but faster method
(working with numpy arrays).

* **Parameters:**
  * **xe** (*np.ndarray*) – Array for the free electron fraction X_e = xHII(1+db).
  * **metadata** ([*MetaData*](nnero.data.md#nnero.data.MetaData)) – Metadata object (attached to the networks) describing properties
    of the data the networks have been trained on.
  * **default** (*dict* *|* *None* *,* *optional*) – Dictionnary of default values, by default None.
    The default dictionnaly should have keys corresponding to those of the trained dataset
    as accessible from metadata.parameters_name.
  * **\*\*kwargs** – Any value for a parameter in metadata.
    Keys should corresponds to metadata.parameters_name.
* **Return type:**
  float

### nnero.predictor.predict_tau_from_Xe_numpy(xe: ndarray, theta: ndarray, metadata: [MetaData](nnero.data.md#nnero.data.MetaData)) → ndarray

Predict the optical depth to reionization from an array of X_e.
Same as predict_tau_from_Xe but much faster as it work with numpy arrays.

* **Parameters:**
  * **xe** (*np.ndarray*) – Array for the free electron fraction X_e = xHII(1+db).
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    regressor for training. These parameters and their order are accesible
    calling metadata.parameters_name.
  * **metadata** ([*MetaData*](nnero.data.md#nnero.data.MetaData)) – Metadata object (attached to the networks) describing properties
    of the data the networks have been trained on.
* **Returns:**
  Array of values of the optical depth to reionization.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_tau_from_xHII(Xe, metadata: [MetaData](nnero.data.md#nnero.data.MetaData), default: dict | None = None, \*\*kwargs) → float

#### Deprecated
Deprecated since version See: predict_tau_from_Xe instead.

### nnero.predictor.predict_tau_from_xHII_numpy(Xe, theta: ndarray, metadata: [MetaData](nnero.data.md#nnero.data.MetaData)) → ndarray

#### Deprecated
Deprecated since version See: predict_tau_from_Xe_numpy instead.

### nnero.predictor.predict_tau_numpy(theta: ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None) → ndarray

Predict the optical depth to reionization from a trained classifier and regressor
as well as parameters passed in kwargs or default. Same as predict_tau but much faster
as it work with numpy arrays.

* **Parameters:**
  * **theta** (*np.ndarray*) – Array of shape (n, p) of parameters, where p is the number of parameters.
    The p parameters values should be given in the order they were fed to the
    regressor for training. These parameters and their order are accesible
    calling regressor.info() or classifier.info().
  * **classifier** ([*Classifier*](nnero.classifier.md#nnero.classifier.Classifier) *|* *None* *,* *optional*) – Classifier object already trained. Default is None. If None, the default
    classifier DefaultClassifier is used.
  * **regressor** ([*Regressor*](nnero.regressor.md#nnero.regressor.Regressor) *|* *None* *,* *optional*) – Regressor object already trained. Default is None. If None, the default
    regressor DefaultRegressor is used.
* **Returns:**
  Array of values of the optical depth to reionization.
* **Return type:**
  np.ndarray

### nnero.predictor.predict_xHII(classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None, default: dict | None = None, \*\*kwargs) → ndarray

#### Deprecated
Deprecated since version See: predict_Xe instead.

### nnero.predictor.predict_xHII_numpy(theta: ndarray, classifier: [Classifier](nnero.classifier.md#nnero.classifier.Classifier) | None = None, regressor: [Regressor](nnero.regressor.md#nnero.regressor.Regressor) | None = None) → ndarray

#### Deprecated
Deprecated since version See: predict_Xe_numpy instead.

### nnero.predictor.uniform_input_array(theta: ndarray, metadata: [MetaData](nnero.data.md#nnero.data.MetaData))

### nnero.predictor.uniform_input_values(metadata: [MetaData](nnero.data.md#nnero.data.MetaData), default: dict = {'ALPHA_ESC': 0.3, 'ALPHA_STAR': 0.5, 'FRAC_WDM': 0.0, 'F_ESC10': -1.0, 'F_STAR10': -1.5, 'INVERSE_M_WDM': 0.0, 'LOG10_PMF_SB': -5.0, 'L_X': 40.0, 'Ln_1010_As': 3.047, 'M_TURN': 8.7, 'M_WDM': '20.0', 'NEUTRINO_MASS_1': 0.0, 'NU_X_THRESH': 500, 'Ombh2': 0.02242, 'Omdmh2': 0.11933, 'PMF_NB': -2.0, 'POWER_INDEX': 0.9665, 'hlittle': 0.6736, 't_STAR': 0.5}, \*\*kwargs)
