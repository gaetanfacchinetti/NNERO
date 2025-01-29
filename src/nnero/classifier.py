##################################################################################
# This file is part of NNERO.
#
# Copyright (c) 2024, Gaétan Facchinetti
#
# NNERO is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by 
# the Free Software Foundation, either version 3 of the License, or any 
# later version. NNERO is distributed in the hope that it will be useful, 
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.
#
# You should have received a copy of the GNU 
# General Public License along with NNERO. 
# If not, see <https://www.gnu.org/licenses/>.
#
##################################################################################

import numpy as np
import torch
import torch.nn as nn

from .data    import TorchDataset, DataSet
from .network import NeuralNetwork

import os
import pkg_resources

from typing import Self

DATA_PATH = pkg_resources.resource_filename('nnero', 'nn_data/')

class Classifier(NeuralNetwork):
    """
    Daughter class of :py:class:`NeuralNetwork` specialised for classifier.
    
    Parameters
    ----------
    model: torch.nn.Module | None
        If not None, the model that will be used for classifier. 
        Otherwise, a new model is constructed from `n_input`, `n_hidden_features` 
        and `n_hidden_layers`. Default is None.
    n_input: int, optional
        Number of input on the neural network 
        (corresponds to the number of parameters).
        Default is 16.
    n_hidden_features: int, optional
        Number of hidden features per layer. Default is 32.
    n_hidden_layers: int, optional
        Number of layers. Default is 4.
    name: str | None
        Name of the neural network. If None, automatically set to DefaultClassifier.
        Default is None.
    dataset: Dataset | None
        Dataset on which the model will be trained. 
        If provided, gets `n_input` from the data and overrides the user input value.

    Attributes
    ----------
    - name : str
        the name of the model
    """

    def __init__(self, 
                *,
                n_input: int = 16, 
                n_hidden_features: int = 32, 
                n_hidden_layers: int = 4, 
                model: torch.nn.Module | None = None, 
                name: str | None = None,
                dataset: DataSet | None = None) -> None:

        # if no name, give a default
        if name is None:
            name = "DefaultClassifier"

        if dataset is not None:     
            n_input = len(dataset.metadata.parameters_name)


        # give a default empty array for the structure
        # stays None if a complex model is passed as input
        struct = np.empty(0)
    
        # if no model defined in input give a model
        if model is None:
            
            # define a list of hidden layers
            hidden_layers = []
            for _ in range(n_hidden_layers):
                hidden_layers.append(nn.Linear(n_hidden_features, n_hidden_features))
                hidden_layers.append(nn.ReLU())

            # create a sequential model
            model  = nn.Sequential(nn.Linear(n_input, n_hidden_features), *hidden_layers, nn.Linear(n_hidden_features, 1), nn.Sigmoid())
            
            # save the structure of this sequential model
            struct = np.array([n_input, n_hidden_features, n_hidden_layers])

        # call the (grand)parent init function
        super(Classifier, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        # structure of the model
        self._struct = struct

        # define the model
        self._model = model

        # define the loss function (here binary cross-entropy)
        self._loss_fn = nn.BCELoss()

        # if the dataset is already given, set it as the dataset of the network
        if dataset is not None:
            self.set_check_metadata_and_partition(dataset)



    @classmethod
    def load(cls, path: str | None = None) -> Self:
        """
        Loads a classifier.

        Parameters
        ----------
        path: str | None
            Path to the saved files containing the classifier data.
            If None automatically fetch the DefaultClassifier.

        Returns
        -------
        Classifier
        """

        if path is None: 
            path = os.path.join(DATA_PATH, "DefaultClassifier")

        name = path.split('/')[-1]
        
        if os.path.isfile(path  + '_struct.npy'):

            with open(path  + '_struct.npy', 'rb') as file:
                struct  = np.load(file)

                if len(struct) == 3:

                    classifier = Classifier(n_input=struct[0], n_hidden_features=struct[1], n_hidden_layers=struct[2])
                    classifier.load_weights_and_extras(path)
                    classifier.eval()

                    print('Model ' + str(name) + ' sucessfully loaded')

                    return classifier
        
        # if the struct read is not of the right size
        # check for a pickled save of the full class
        # (although this is not recommended)
        if os.path.isfile(path  + '.pth') :
            classifier = torch.load(path + ".pth")
            classifier.eval()

            print('Model ' + str(name) + ' sucessfully loaded from a .pth archive')

            return classifier
        
        raise ValueError("Could not find a fully saved classifier model at: " + path)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation of the model.

        Parameters
        ----------
        x: torch.Tensor
            Input features.
        """
        return torch.flatten(self._model(x))
    
    @property
    def loss_fn(self):
        return self._loss_fn
    
    def test(self, dataset: DataSet | None = None, x_test:np.ndarray | None = None, y_test: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Test the efficiency of the classifier.

        Parameters
        ----------
        dataset: DataSet | None
            DataSet containing the training partition and the test partition.
        x_test: np.ndarray
        y_test: np.ndarray

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            y_pred, y_test, and array of true if rightly classifier, false otherwise 

        Raises
        ------
        ValueError
            Either the dataset or both x_test and y_test must be provided.
        """

        if dataset is not None:
            self.set_check_metadata_and_partition(dataset, check_only = True)
            x_test = torch.tensor(dataset.x_array[dataset.partition.total_test],      dtype=torch.float32)
            y_test = torch.tensor(dataset.y_classifier[dataset.partition.total_test], dtype=torch.float32)
        elif (x_test is not None) and (y_test is not None):
            x_test = torch.tensor(x_test, dtype=torch.float32)
            y_test = torch.tensor(y_test, dtype=torch.float32)
        else:
            raise ValueError("Either the dataset or both x_test and y_test must be provided.")
        
        self.eval()
        
        with torch.no_grad():
            y_pred  = self.forward(x_test)
            print(f"The accuracy is {100*(y_pred.round() == y_test).float().mean():.4f}%")
            return y_pred.numpy(), y_test.numpy(), (y_pred.round() == y_test).numpy()
        

    def validate(self, dataset: DataSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Validate the efficiency of the classifier.

        Parameters
        ----------
        dataset: DataSet
            DataSet containing the training partition and the test partition.

        Returns
        -------
        tuple(np.ndarray, np.ndarray, np.ndarray)
            y_pred, y_test, and array of true if rightly classifier, false otherwise 
        """

        self.set_check_metadata_and_partition(dataset, check_only = True)
        x_valid = torch.tensor(dataset.x_array[dataset.partition.total_valid],      dtype=torch.float32)
        y_valid = torch.tensor(dataset.y_classifier[dataset.partition.total_valid], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred  = self.forward(x_valid)
            print(f"The accuracy is {100*(y_pred.round() == y_valid).float().mean():.4f}%")
            return y_pred.numpy(), y_valid.numpy(), (y_pred.round() == y_valid).numpy()


    
def train_classifier(model: Classifier, 
                     dataset: DataSet, 
                     optimizer:torch.optim.Optimizer, 
                     *, 
                     epochs: int = 50, 
                     learning_rate: float = 1e-3, 
                     verbose: bool = True, 
                     batch_size: int = 64, 
                     x_train: np.ndarray | None = None,
                     y_train: np.ndarray | None = None,
                     x_valid: np.ndarray | None = None,
                     y_valid: np.ndarray | None = None,
                     **kwargs)-> None:
    """
    Trains a given classifier.

    Parameters
    ----------
    model : Classifier
        Classifier model to train.
    dataset : DataSet
        Dataset on which to train the classifier.
    optimizer : torch.optim.Optimizer
        Optimizer used for training.
    epochs : int, optional
        Number of epochs, by default 50.
    learning_rate : float, optional
        Learning rate for training, by default 1e-3.
    verbose : bool, optional
        If true, outputs a summary of the losses at each epoch, by default True.
    batch_size : int, optional
        Size of the training batches, by default 64.
    """
    # set the metadata and parition object of the model
    model.set_check_metadata_and_partition(dataset)

    if x_train is None:
        x_train = dataset.x_array[dataset.partition.total_train]
    
    if y_train is None:
        y_train = dataset.y_classifier[dataset.partition.total_train]

    if x_valid is None:
        x_valid = dataset.x_array[dataset.partition.total_valid]

    if y_valid is None:
        y_valid = dataset.y_classifier[dataset.partition.total_valid]
        
    # format the data for the classifier
    train_dataset = TorchDataset(x_train, y_train)
    valid_dataset = TorchDataset(x_valid, y_valid)
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    
    # we have only one param_group here
    # we modify the learning rate of that group
    optimizer.param_groups[0]['lr'] = learning_rate

    # start loop on the epochs
    for epoch in range(epochs):
        
        train_loss     = np.array([])
        valid_loss     = np.array([])
        train_accuracy = np.array([])
        valid_accuracy = np.array([])

        # training mode
        model.train()
        
        for batch in train_loader:
            x_batch, y_batch = batch

            optimizer.zero_grad()
            y_pred = model.forward(x_batch)
            loss   = model.loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss     = np.append(train_loss, loss.item())
            train_accuracy = np.append(train_accuracy, (y_pred.round() == y_batch).float().mean())


        # evaluation mode
        model.eval()
    
        with torch.no_grad():
            
            for batch in valid_loader:
                x_batch, y_batch = batch
            
                y_pred = model(x_batch)             # forward pass
                loss = model.loss_fn(y_pred, y_batch) # loss function averaged over the batch size
                
                valid_loss     = np.append(valid_loss, loss.item())
                valid_accuracy = np.append(valid_accuracy, (y_pred.round() == y_batch).float().mean())
        
        # get the mean of all batches
        model._train_loss     = np.append(model._train_loss, np.mean(train_loss))
        model._valid_loss     = np.append(model._valid_loss, np.mean(valid_loss))
        model._train_accuracy = np.append(model._train_accuracy, np.mean(train_accuracy))
        model._valid_accuracy = np.append(model._valid_accuracy, np.mean(valid_accuracy))

        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], loss: ({model.train_loss[-1]:.4f}, {model.valid_loss[-1]:.4f}), accuracy = ({model.train_accuracy[-1]:.4f}, {model.valid_accuracy[-1]:.4f})')   

