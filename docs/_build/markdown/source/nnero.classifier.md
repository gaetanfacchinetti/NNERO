# Classifier

### *class* nnero.classifier.Classifier(\*, n_input: int = 16, n_hidden_features: int = 32, n_hidden_layers: int = 4, model: Module | None = None, name: str | None = None, dataset: [DataSet](nnero.data.md#nnero.data.DataSet) | None = None)

Bases: [`NeuralNetwork`](nnero.network.md#nnero.network.NeuralNetwork)

Daughter class of `NeuralNetwork` specialised for classifier.

* **Parameters:**
  * **model** (*torch.nn.Module* *|* *None*) – If not None, the model that will be used for classifier.
    Otherwise, a new model is constructed from n_input, n_hidden_features
    and n_hidden_layers. Default is None.
  * **n_input** (*int* *,* *optional*) – Number of input on the neural network
    (corresponds to the number of parameters).
    Default is 16.
  * **n_hidden_features** (*int* *,* *optional*) – Number of hidden features per layer. Default is 32.
  * **n_hidden_layers** (*int* *,* *optional*) – Number of layers. Default is 4.
  * **name** (*str* *|* *None*) – Name of the neural network. If None, automatically set to DefaultClassifier.
    Default is None.
  * **dataset** (*Dataset* *|* *None*) – Dataset on which the model will be trained.
    If provided, gets n_input from the data and overrides the user input value.

### - name

the name of the model

* **Type:**
  str

#### forward(x: Tensor) → Tensor

Forward evaluation of the model.

* **Parameters:**
  **x** (*torch.Tensor*) – Input features.

#### *classmethod* load(path: str | None = None) → Self

Loads a classifier.

* **Parameters:**
  **path** (*str* *|* *None*) – Path to the saved files containing the classifier data.
  If None automatically fetch the DefaultClassifier.
* **Return type:**
  [Classifier](#nnero.classifier.Classifier)

#### test(dataset: [DataSet](nnero.data.md#nnero.data.DataSet) | None = None, x_test: ndarray | None = None, y_test: ndarray | None = None) → tuple[ndarray, ndarray, ndarray]

Test the efficiency of the classifier.

* **Parameters:**
  * **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet) *|* *None*) – DataSet containing the training partition and the test partition.
  * **x_test** (*np.ndarray*)
  * **y_test** (*np.ndarray*)
* **Returns:**
  y_pred, y_test, and array of true if rightly classifier, false otherwise
* **Return type:**
  tuple(np.ndarray, np.ndarray, np.ndarray)
* **Raises:**
  **ValueError** – Either the dataset or both x_test and y_test must be provided.

#### validate(dataset: [DataSet](nnero.data.md#nnero.data.DataSet)) → tuple[ndarray, ndarray, ndarray]

Validate the efficiency of the classifier.

* **Parameters:**
  **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet)) – DataSet containing the training partition and the test partition.
* **Returns:**
  y_pred, y_test, and array of true if rightly classifier, false otherwise
* **Return type:**
  tuple(np.ndarray, np.ndarray, np.ndarray)

### nnero.classifier.train_classifier(model: [Classifier](#nnero.classifier.Classifier), dataset: [DataSet](nnero.data.md#nnero.data.DataSet), optimizer: Optimizer, \*, epochs: int = 50, learning_rate: float = 0.001, verbose: bool = True, batch_size: int = 64, x_train: ndarray | None = None, y_train: ndarray | None = None, x_valid: ndarray | None = None, y_valid: ndarray | None = None, \*\*kwargs) → None

Trains a given classifier.

* **Parameters:**
  * **model** ([*Classifier*](#nnero.classifier.Classifier)) – Classifier model to train.
  * **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet)) – Dataset on which to train the classifier.
  * **optimizer** (*torch.optim.Optimizer*) – Optimizer used for training.
  * **epochs** (*int* *,* *optional*) – Number of epochs, by default 50.
  * **learning_rate** (*float* *,* *optional*) – Learning rate for training, by default 1e-3.
  * **verbose** (*bool* *,* *optional*) – If true, outputs a summary of the losses at each epoch, by default True.
  * **batch_size** (*int* *,* *optional*) – Size of the training batches, by default 64.
