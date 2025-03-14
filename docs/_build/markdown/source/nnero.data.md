# Data

### *class* nnero.data.DataPartition(early_train: ndarray, early_valid: ndarray, early_test: ndarray, total_train: ndarray, total_valid: ndarray, total_test: ndarray)

Bases: `object`

DataPartition class.

Partitioning of the data into a training set, a testing set and a validation set.

* **Parameters:**
  * **early_train** (*np.ndarray*) – indices of the data array with an early enough reionization used for training
  * **early_valid** (*np.ndarray*) – indices of the data array with an early enough reionization used for validation
  * **early_test** (*np.ndarray*) – indices of the data array with an early enough reionization used for testing
  * **total_train** (*np.ndarray*) – all indices of the data array used for training
  * **total_valid** (*np.ndarray*) – all indices of the data array used for validation
  * **total_test** (*np.ndarray*) – all indices of the data array used for testing

#### *classmethod* load(path: str) → Self

Load a previously saved data partition.

* **Parameters:**
  **path** (*str*) – path to the data partition saved file.
* **Return type:**
  [DataPartition](#nnero.data.DataPartition)

#### save(name: str) → None

Save the data partition.

* **Parameters:**
  **name** (*str*) – name of the data partition file

### *class* nnero.data.DataSet(file_path: str, z: ndarray | None = None, \*, frac_test: float = 0.1, frac_valid: float = 0.1, seed_split: int = 1994, extras: list[str] | None = None)

Bases: `object`

DataSet class

Compile the data necessary for training.

* **Parameters:**
  * **file_path** (*str*) – path to the file that contains the raw data
  * **z** (*np.ndarray*) – array of the redshits of interpolation of the nn
  * **use_PCA** (*bool* *,* *optional*) – prepare the data to perform the regression in the principal component basis, default is True
  * **precision_PCA** (*float* *,* *optional*) – if use_PCA is True, select the number of useful eigenvectors from this coefficient
    – only the eigenvectors with eigenvalues larger than precision_PCA \* the largest eigenvalue
    are considered as useful
  * **frac_test** (*float* *,* *optional*) – fraction of test data out of the total sample, default is 0.1
  * **frac_valid** (*float* *,* *optional*) – fraction of validation data out of the total sample, default is 0.1
  * **seed_split** (*int* *,* *optional*) – random seed for data partitioning, default is 1994

#### init_principal_components(pca_precision: float = 0.001) → int

Initialise the principal component analysis decomposition

* **Parameters:**
  **pca_precision** (*float* *,* *optional*) – precision for the principal analysis reconstruction, by default 1e-3
* **Returns:**
  number of necessary eigenvectors to reach the desired precision
* **Return type:**
  int

### *class* nnero.data.MetaData(z: ndarray, parameters_name: list | ndarray, parameters_min_val: ndarray, parameters_max_val: ndarray)

Bases: `object`

MetaData class

Metadata that is saved with the neural network for predictions.

* **Parameters:**
  * **z** (*np.ndarray*) – array of redshifts
  * **parameters_name** (*list* *|* *np.ndarray*) – name of the parameters (input features)
  * **parameters_min_val** (*np.ndarray*) – minimum value of the parameters (input features)
  * **parameters_max_val** (*np.ndarray*) – maximum value of the parameters (input features)

#### *classmethod* load(path: str) → Self

Load a previously saved metadata file.

* **Parameters:**
  **path** (*str*) – path to the metadata saved file.
* **Return type:**
  [MetaData](#nnero.data.MetaData)

#### save(name: str) → None

Save the metadata.

* **Parameters:**
  **name** (*str*) – name of the metadata file

### *class* nnero.data.TorchDataset(x_data: ndarray, y_data: ndarray)

Bases: `Dataset`

Wrapper of torch Dataset.

* **Parameters:**
  * **x_data** (*np.ndarray*) – input features
  * **y_data** (*np.ndarray*) – output labels

### nnero.data.latex_labels(labels: list[str]) → list[str]

### nnero.data.preprocess_raw_data(file_path: str, \*, random_seed: int = 1994, frac_test: float = 0.1, frac_valid: float = 0.1, extras: list[str] | None = None) → None

Preprocess a raw .npz file.
Creates another numpy archive that can be directly used to create a [`DataSet`](#nnero.data.DataSet) object.

* **Parameters:**
  * **file_path** (*str*) – Path to the raw data file. The raw data must be a .npz file with the following information.
    - z (or z_glob): redshift array
    - features_run:  Sequence of drawn input parameters for which there is a value for the ionization fraction
    - features_fail: Sequence of drawn input parameters for which the simulator failed because reionization was too late
    - …
  * **random_seed** (*int* *,* *optional*) – Random seed for splitting data into a training/validation/testing subset, by default 1994.
  * **frac_test** (*float* *,* *optional*) – Fraction of the total data points in the test subset, by default 0.1.
  * **frac_valid** (*float* *,* *optional*) – Fraction of the total data points in the validation subset, by default 0.1

### nnero.data.true_to_uniform(x: float | ndarray, min: float | ndarray, max: float | ndarray) → float | ndarray

Transforms features uniformely distributed along [a, b] into features
uniformely distributed between [0, 1] as fed to the neural networks.

* **Parameters:**
  * **x** (*float* *|* *np.ndarray*) – input featurs distributed uniformely on [a, b]
  * **min** (*float* *|* *np.ndarray*) – minimum value a
  * **max** (*float* *|* *np.ndarray*) – maximum value b
* **Return type:**
  float | np.ndarray
* **Raises:**
  **ValueError** – min should be less than max

### nnero.data.uniform_to_true(x: float | ndarray, min: float | ndarray, max: float | ndarray) → float | ndarray

Inverse transformation of true_to_uniform.

* **Parameters:**
  * **x** (*float* *|* *np.ndarray*) – input featurs distributed uniformely on [0, 1]
  * **min** (*float* *|* *np.ndarray*) – minimum value a
  * **max** (*float* *|* *np.ndarray*) – maximum value b
* **Return type:**
  float | np.ndarray
* **Raises:**
  **ValueError** – min should be less than max
