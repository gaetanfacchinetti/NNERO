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

DATA_PATH = pkg_resources.resource_filename('nnero', 'nn_data/')

class Classifier(NeuralNetwork):
    """
    This is :py:class:`Classifier`, a daughter class of :py:class:`NeuralNetwork` specialised for classifier
    
    Attributes
    ----------
    - name : str
        the name of the model
    - metadata : Metadata
        metadata on which the model is trained
    - partition : DataPartition
        partitioning of the data on which the model is trained
    - train_loss : np.ndarray
        1D array training loss for each training epoch
    - valid_loss : np.ndarray
        1D array validation losses for each training epoch
    - train_accuracy : np.ndarray
        1D array training accuracy for each training epoch
    - valid_accuracy : np.ndarray
        1D array validation accuracy for each training epoch

    Methods
    -------
    - save(path=".", save_partition=True)
        save the neural network
    - load_weights_and_extras(path)
        load the weights and extra available info on the network 
    """

    def __init__(self, 
                *,
                n_input: int = 16, 
                n_hidden_features: int = 32, 
                n_hidden_layers: int = 4, 
                model = None, name: str | None = None):

        # if no name, give a default
        if name is None:
            name = "DefaultClassifier"

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

        # print the number of parameters
        self.print_parameters()


    @classmethod
    def load(cls, path = os.path.join(DATA_PATH, "DefaultClassifier")):
        
        if os.path.isfile(path  + '_struct.npy'):

            with open(path  + '_struct.npy', 'rb') as file:
                struct  = np.load(file)

                if len(struct) == 3:

                    classifier = Classifier(n_input=struct[0], n_hidden_features=struct[1], n_hidden_layers=struct[2])
                    classifier.load_weights_and_extras(path)
                    classifier.eval()

                    return classifier
        
        # if the struct read is not of the right size
        # check for a pickled save of the full class
        # (although this is not recommended)
        if os.path.isfile(path  + '.pth') :
            classifier = torch.load(path + ".pth")
            classifier.eval()
            return classifier
        
        raise ValueError("Could not find a fully saved classifier model at: " + path)


    def forward(self, x):
        return torch.flatten(self._model(x))
    
    @property
    def loss_fn(self):
        return self._loss_fn
    
    def test(self, dataset:DataSet):

        self.set_check_metadata_and_partition(dataset, check_only = True)
        x_test = torch.tensor(dataset.x_array[dataset.partition.total_test],      dtype=torch.float32)
        y_test = torch.tensor(dataset.y_classifier[dataset.partition.total_test], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred  = self.forward(x_test)
            print(f"The accuracy is {100*(y_pred.round() == y_test).float().mean():.4f}%")


    
def train_classifier(model: Classifier, 
                     dataset: DataSet, 
                     optimizer:torch.optim.Optimizer, 
                     *, 
                     epochs = 50, 
                     learning_rate = 1e-3, 
                     verbose = True, 
                     batch_size = 64, 
                     **kwargs):
    
    # set the metadata and parition object of the model
    model.set_check_metadata_and_partition(dataset)

    # format the data for the classifier
    train_dataset = TorchDataset(dataset.x_array[dataset.partition.total_train], dataset.y_classifier[dataset.partition.total_train])
    valid_dataset = TorchDataset(dataset.x_array[dataset.partition.total_valid], dataset.y_classifier[dataset.partition.total_valid])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, **kwargs)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, **kwargs)
    
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

