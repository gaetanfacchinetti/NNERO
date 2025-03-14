# Regressor

### *class* nnero.regressor.Regressor(\*, n_input: int = 16, n_output: int = 50, n_hidden_features: int = 80, n_hidden_layers: int = 5, alpha_tau: float = 0.5, model=None, name: str | None = None, use_pca: bool = True, pca_precision: float = 0.001, dataset: [DataSet](nnero.data.md#nnero.data.DataSet) | None = None)

Bases: [`NeuralNetwork`](nnero.network.md#nnero.network.NeuralNetwork)

Daughter class of `NeuralNetwork` specialised for regressors.

* **Parameters:**
  * **model** (*torch.nn.Module* *|* *None*) – If not None, the model that will be used for classifier.
    Otherwise, a new model is constructed from n_input, n_hidden_features
    and n_hidden_layers. Default is None.
  * **n_input** (*int* *,* *optional*) – Number of input on the neural network
    (corresponds to the number of parameters).
    Default is 16.
  * **n_input** – Number of output on the neural network
    (corresponds to the number of redshift bins).
    Default is 50. Overriden if dataset is specified.
  * **n_hidden_features** (*int* *,* *optional*) – Number of hidden features per layer. Default is 80.
  * **n_hidden_layers** (*int* *,* *optional*) – Number of layers. Default is 5.
  * **name** (*str* *|* *None*) – Name of the neural network. If None, automatically set to DefaultClassifier.
    Default is None.
  * **dataset** (*Dataset* *|* *None*) – Dataset on which the model will be trained.
    If provided, gets n_input and n_output from the data and overrides the user input value.
  * **use_pca** (*bool* *,* *optional*) – If True, decompose the interpolated function on the principal component eigenbasis.
    Default is True.
  * **pca_precision** (*float* *,* *optional*) – If use_pca is True sets how many eigenvalues needs to be considered.
    Only consider eigenvectors with eigenvalues > precision^2 \* max(eigenvalues).
    Default is 1e-3.
  * **alpha_tau** (*float* *,* *optioal*) – Weighting of the relative error on X_e and optical depth in the cost function.
    Default is 0.5

### - name

the name of the model

* **Type:**
  str

#### forward(x)

Forward evaluation of the model.

* **Parameters:**
  **x** (*torch.Tensor*) – input features

#### *classmethod* load(path: str | None = None)

Loads a regressor.

* **Parameters:**
  **path** (*str* *|* *None*) – Path to the saved files containing the regressor data.
  If None automatically fetch the DefaultRegressor.
* **Return type:**
  [Regressor](#nnero.regressor.Regressor)

#### loss_tau(tau_pred, target)

#### loss_xHII(output, target)

#### tau_ion(x, y)

Optical depth to reionization.

* **Parameters:**
  * **x** (*torch.Tensor*) – Input features.
  * **y** (*torch.Tensor*) – Output of the Regressor. Corresponds to X_e(z).

#### test_tau(dataset: [DataSet](nnero.data.md#nnero.data.DataSet)) → ndarray

Test the efficiency of the regressor to reconstruct the optical depth to reionization.

* **Parameters:**
  **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet)) – DataSet containing the training partition and the test partition.
* **Returns:**
  Distribution of relative error between the predicted and true optical depth.
  Array with the size of the test dataset.
* **Return type:**
  np.ndarray

#### test_xHII(dataset: [DataSet](nnero.data.md#nnero.data.DataSet)) → tuple[ndarray, ndarray]

Test the efficiency of the regressor to reconstruct the free electron fraction X_e.

* **Parameters:**
  **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet)) – DataSet containing the training partition and the test partition.
* **Returns:**
  Prediction for X_e and true values.
* **Return type:**
  tuple(np.ndarray, np.ndarray)

### nnero.regressor.train_regressor(model: [Regressor](#nnero.regressor.Regressor), dataset: [DataSet](nnero.data.md#nnero.data.DataSet), optimizer: Optimizer, \*, epochs=50, learning_rate=0.001, verbose=True, batch_size=64, \*\*kwargs)

Trains a given regressor.

* **Parameters:**
  * **model** ([*Regressor*](#nnero.regressor.Regressor)) – Regressor model to train.
  * **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet)) – Dataset on which to train the regressor.
  * **optimizer** (*torch.optim.Optimizer*) – Optimizer used for training.
  * **epochs** (*int* *,* *optional*) – Number of epochs, by default 50.
  * **learning_rate** (*float* *,* *optional*) – Learning rate for training, by default 1e-3.
  * **verbose** (*bool* *,* *optional*) – If true, outputs a summary of the losses at each epoch, by default True.
  * **batch_size** (*int* *,* *optional*) – Size of the training batches, by default 64.
