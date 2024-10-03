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

from .data import TorchDataset, DataSet
from .nn   import NeuralNetwork


class Classifier(NeuralNetwork):

    def __init__(self, 
                *,
                n_input: int = 16, 
                n_hidden_features: int = 32, 
                n_hidden_layers: int = 4, 
                model = None, name: str | None = None):

        if name is None:
            name = "DefaultClassifier"

        if model is None:

                 
            hidden_layers = []
            for _ in range(n_hidden_layers):
                hidden_layers.append(nn.Linear(n_hidden_features, n_hidden_features))
                hidden_layers.append(nn.ReLU())

            model = nn.Sequential(nn.Linear(n_input, n_hidden_features), *hidden_layers, nn.Linear(n_hidden_features, 1), nn.Sigmoid())
        
        super(Classifier, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        self._model = model
        self._loss_fn = nn.BCELoss()

        self.print_parameters()

    @classmethod
    def load(cls, path = "../data/DefaultClassifier.pth"):
        return torch.load(path)
        
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



    
    
    
def train_classifier(model: Classifier, dataset: DataSet, optimizer:torch.optim.Optimizer, *, epochs = 50, learning_rate = 1e-3, verbose = True, batch_size = 64, **kwargs):
    
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

