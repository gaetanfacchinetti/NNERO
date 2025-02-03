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

from .data      import TorchDataset, DataSet
from .network   import NeuralNetwork

import os
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('nnero', 'nn_data/')


class Interpolator(NeuralNetwork):
    """
    Daughter class of :py:class:`NeuralNetwork` specialised for interpolators.
    
    Parameters
    ----------
    model: torch.nn.Module | None
        If not None, the model that will be used for classifier. 
        Otherwise, a new model is constructed from `n_input`, `n_hidden_features` 
        and `n_hidden_layers`. Default is None.
    dataset: Dataset
        Dataset on which the model will be trained. 
    n_input: int, optional
        Number of input on the neural network 
        (corresponds to the number of parameters).
        Default is 16.
    n_input: int, optional
        Number of output on the neural network 
        (corresponds to the number of redshift bins).
        Default is 50. Overriden if `dataset` is specified.
    n_hidden_features: int, optional
        Number of hidden features per layer. Default is 80.
    n_hidden_layers: int, optional
        Number of layers. Default is 5.
    name: str | None, optional
        Name of the neural network. If None, automatically set to DefaultClassifier.
        Default is None.
    parameter: str | None, optional 
        Parameter which we want to interpolate. Default is None, which only consider the first
        parameter in the extras of the dataset.
        
    Attributes
    ----------
    name: str
        the name of the model

    Raises
    ------
    ValueError
        When the parameter that the network needs to learn is not in the dataset. 
    """

    def __init__(self,
                 dataset: DataSet | None = None,
                 *, 
                 n_input: int = 15,
                 n_hidden_features: int = 80, 
                 n_hidden_layers: int = 5, 
                 model = None, 
                 parameter: str | None = None,
                 name: str | None = None):

        
        if dataset is not None:

            n_input = len(dataset.metadata.parameters_name)

            if (parameter is not None) and (parameter not in dataset.extras_name):
                raise ValueError("Need to pass a parameter that is in the dataset.")
            
            if len(dataset.extras_name) == 0:
                raise ValueError("No extra parameters in the dataset.")
            
            self._id_param  = list(dataset.extras_name).index(parameter) if parameter is not None else 0

         
        self._parameter = parameter

        if name is None:
            name = "DefaultInterpolator_" + parameter


        # give a default empty array for the structure
        # stays None if a complex model is passed as input
        struct = np.empty(0)

        # definition of the model if none is given as input
        if model is None:
            
            hidden_layers = []
            for _ in range(n_hidden_layers):
                hidden_layers.append(nn.Linear(n_hidden_features, n_hidden_features))
                hidden_layers.append(nn.ReLU())

            # create a sequential model
            model = nn.Sequential(nn.Linear(n_input, n_hidden_features), *hidden_layers, nn.Linear(n_hidden_features, 1))
        
            # save the structure of this sequential model and more
            struct = np.array([n_input, n_hidden_features, n_hidden_layers, self.parameter])
        
        # call the (grand)parent constructors
        super(Interpolator, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        # structure of the model
        self._struct = struct
        
        # define the model
        self._model = model

        # set the dataset of the network
        if dataset is not None:
            self.set_check_metadata_and_partition(dataset)

    
    @property
    def id_param(self) -> int:
        return self._id_param
    
    @property
    def parameter(self) -> str:
        return self._parameter


    @classmethod
    def load(cls, path: str | None = None):
        """
        Loads an interpolator.

        Parameters
        ----------
        path: str | None
            Path to the saved files containing the interpolator data.
            If None automatically fetch the DefaultInterpolator.

        Returns
        -------
        Inteprolator
        """

        if path is None:
            path = os.path.join(DATA_PATH, "DefaultInterpolator")

        name = path.split('/')[-1]
        
        if os.path.isfile(path  + '_struct.npy'):

            with open(path  + '_struct.npy', 'rb') as file:
                struct  = np.load(file)

                if len(struct) == 4:

                    interpolator = Interpolator(n_input=int(struct[0]), 
                                          n_hidden_features=int(struct[1]), 
                                          n_hidden_layers=int(struct[2]),
                                          parameter=str(struct[3]),
                                          name=name)
                    interpolator.load_weights_and_extras(path)
                    interpolator.eval()

                    print('Model ' + str(name) + ' sucessfully loaded')

                    return interpolator
        
        # if the struct read is not of the right size
        # check for a pickled save of the full class
        # (although this is not recommended)
        if os.path.isfile(path  + '.pth') :
            interpolator = torch.load(path + ".pth")
            interpolator.eval()

            print('Model ' + str(name) + ' sucessfully loaded from a .pth archive')
            
            return interpolator
        
        raise ValueError("Could not find a fully saved regressor model at: " + path)


        
    def forward(self, x):
        """
        Forward evaluation of the model.

        Parameters
        ----------
        x: torch.Tensor
            input features
        """

        return torch.flatten(torch.clamp(self._model(x), max=1))
    
    def loss(self, pred, target):
        return torch.mean(torch.abs(1.0 - torch.div(pred, target)))
    

    def test(self, dataset:DataSet) -> tuple[np.ndarray, np.ndarray]:
        """
        Test the efficiency of the regressor to reconstruct the parameter.

        Parameters
        ----------
        dataset : DataSet
            DataSet containing the training partition and the test partition.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            Prediction for X_e and true values.
        """

        self.set_check_metadata_and_partition(dataset, check_only = True)

        # clean the dataset (# may have some issued for some values)
        # the dataset needs to be checked before hand
        y_test = dataset.extras_array[0, dataset.partition.early_test]
        x_test = dataset.x_array[dataset.partition.early_test]

        mask = ~(np.isnan(y_test) | np.isinf(y_test))
        x_test = torch.tensor(x_test[mask], dtype=torch.float32)
        y_test = torch.tensor(y_test[mask], dtype=torch.float32)
        
        with torch.no_grad():
            y_pred = self.forward(x_test)

        return y_pred.numpy(), y_test.numpy()
    

    

def train_interpolator(model: Interpolator, 
                    dataset: DataSet, 
                    optimizer: torch.optim.Optimizer, 
                    *, 
                    epochs = 50, 
                    learning_rate = 1e-3, 
                    verbose = True, 
                    batch_size = 64, 
                    **kwargs):
    """
    Trains a given regressor.

    Parameters
    ----------
    model : Regressor
        Regressor model to train.
    dataset : DataSet
        Dataset on which to train the regressor.
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

    # set or check the metadata and parition attributes of the model
    model.set_check_metadata_and_partition(dataset)

    # clean the dataset (# may have some issued for some values)
    # the dataset needs to be checked before hand
    y_train = dataset.extras_array[model.id_param, dataset.partition.early_train]
    x_train = dataset.x_array[dataset.partition.early_train]

    y_valid = dataset.extras_array[model.id_param, dataset.partition.early_valid]
    x_valid = dataset.x_array[dataset.partition.early_valid]

    mask = ~(np.isnan(y_train) | np.isinf(y_train))
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    mask = ~(np.isnan(y_valid) | np.isinf(y_valid))
    x_valid = x_valid[mask]
    y_valid = y_valid[mask]
    

    # format the data for the regressor
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
            loss   = model.loss(y_pred, y_batch)

            loss.backward()
            optimizer.step()
            
            train_loss     = np.append(train_loss, loss.item())
            train_accuracy = np.append(train_accuracy, 1-loss.item())


        # evaluation mode
        model.eval()
    
        with torch.no_grad():
            
            for batch in valid_loader:
                x_batch, y_batch = batch
            
                y_pred = model.forward(x_batch)        
                loss   = model.loss(y_pred, y_batch)
                
                valid_loss     = np.append(valid_loss, loss.item())
                valid_accuracy = np.append(valid_accuracy, 1-loss.item())
        
        # get the mean of all batches
        model._train_loss     = np.append(model._train_loss, np.mean(train_loss))
        model._valid_loss     = np.append(model._valid_loss, np.mean(valid_loss))
        model._train_accuracy = np.append(model._train_accuracy, np.mean(train_accuracy))
        model._valid_accuracy = np.append(model._valid_accuracy, np.mean(valid_accuracy))

        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], loss: ({model.train_loss[-1]:.4f}, {model.valid_loss[-1]:.4f}), accuracy = ({model.train_accuracy[-1]:.4f}, {model.valid_accuracy[-1]:.4f})')   
