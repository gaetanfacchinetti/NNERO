# Network

### *class* nnero.network.NeuralNetwork(name: str)

Bases: `Module`

A class wrapping py:class:torch.nn.Module for neural network models

* **Parameters:**
  **name** (*str*) – name of the neural network

#### name

the name of the model

* **Type:**
  str

#### metadata

metadata on which the model is trained

* **Type:**
  Metadata

#### partition

partitioning of the data on which the model is trained

* **Type:**
  [DataPartition](nnero.data.md#nnero.data.DataPartition)

#### train_loss

1D array training loss for each training epoch

* **Type:**
  np.ndarray

#### valid_loss

1D array validation losses for each training epoch

* **Type:**
  np.ndarray

#### train_accuracy

1D array training accuracy for each training epoch

* **Type:**
  np.ndarray

#### valid_accuracy

1D array validation accuracy for each training epoch

* **Type:**
  np.ndarray

#### info()

#### load_weights_and_extras(path: str) → None

loads the network weights and extra information

* **Parameters:**
  **path** (*str*) – path to the network to load
* **Raises:**
  **ValueError** – If not all necessary files exists where path points.

#### print_structure()

prints the list of parameters in the model

#### save(path: str = '.', save_partition: bool = True) → None

Save the neural network model in a bunch of files.

* **Parameters:**
  * **path** (*str* *,* *optional*) – path where to save the neural network
    – default is the current directory “.”
  * **save_partition** (*bool* *,* *optional*) – if save_partition is false the partitioning of the data into
    train, valid and test is not saved (useless for instance once
    we have a fully trained model that we just want to use)
    – default is True

#### set_check_metadata_and_partition(dataset: [DataSet](nnero.data.md#nnero.data.DataSet), check_only: bool = False) → None

set and check the medatada and partition attributes

* **Parameters:**
  * **dataset** ([*DataSet*](nnero.data.md#nnero.data.DataSet)) – dataset to compare or to assign to the object
  * **check_only** (*bool* *,* *optional*) – option to only compare the compatibility
    – default is False
* **Raises:**
  **ValueError** – if the dataset is incompatible with the current metadata or partition
