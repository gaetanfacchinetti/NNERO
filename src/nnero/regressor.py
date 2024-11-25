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

from .data      import TorchDataset, DataSet, uniform_to_true
from .network   import NeuralNetwork
from .cosmology import optical_depth_no_rad

import os
import pkg_resources

DATA_PATH = pkg_resources.resource_filename('nnero', 'nn_data/')




class Regressor(NeuralNetwork):

    def __init__(self, *, 
                 n_input: int = 16, n_output: int = 50, 
                 n_hidden_features: int = 80, n_hidden_layers: int = 5, 
                 alpha_tau : float = 0.5,
                 model = None, name: str | None = None):

        if name is None:
            name = "DefaultRegressor"

        # give a default empty array for the structure
        # stays None if a complex model is passed as input
        struct = np.empty(0)

        if model is None:
            
            hidden_layers = []
            for _ in range(n_hidden_layers):
                hidden_layers.append(nn.Linear(n_hidden_features, n_hidden_features))
                hidden_layers.append(nn.ReLU())

            # create a sequential model
            model = nn.Sequential(nn.Linear(n_input, n_hidden_features), *hidden_layers, nn.Linear(n_hidden_features, n_output))
        
            # save the structure of this sequential model
            struct = np.array([n_input, n_hidden_features, n_hidden_layers])
        
        super(Regressor, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        # structure of the model
        self._struct = struct
        
        # define the model
        self._model     = model

        # define parameters in the loss
        self._alpha_tau = alpha_tau

        # print the number of parameters
        self.print_parameters()
    


    @classmethod
    def load(cls, path = os.path.join(DATA_PATH, "DefaultRegressor")):
        
        if os.path.isfile(path  + '_struct.npy'):

            with open(path  + '_struct.npy', 'rb') as file:
                struct  = np.load(file)

                if len(struct) == 5:

                    regressor = Regressor(n_input=int(struct[0]), 
                                          n_output=int(struct[1]), 
                                          n_hidden_features=int(struct[2]), 
                                          n_hidden_layers=int(struct[3]),
                                          alpha_tau=struct[4])
                    regressor.load_extras(path)
                    regressor.eval()

                    return regressor
        
        # if the struct read is not of the right size
        # check for a pickled save of the full class
        # (although this is not recommended)
        if os.path.isfile(path  + '.pth') :
            regressor = torch.load(path + ".pth")
            regressor.eval()
            return regressor
        
        raise ValueError("Could not find a fully saved regressor model at: " + path)

        
    def forward(self, x):
        return torch.clamp(self._model(x), max=1.0)
    
    def tau_ion(self, x, y):
        z_tensor = torch.tensor(self.metadata.z, dtype=torch.float32)
        omega_b  = uniform_to_true(x[:, self.metadata.pos_omega_b], self.metadata.min_omega_b, self.metadata.max_omega_b)
        omega_c  = uniform_to_true(x[:, self.metadata.pos_omega_c], self.metadata.min_omega_c, self.metadata.max_omega_c)
        hlittle  = uniform_to_true(x[:, self.metadata.pos_hlittle], self.metadata.min_hlittle, self.metadata.max_hlittle)
        return optical_depth_no_rad(z_tensor, y, omega_b, omega_c, hlittle)
    
    def loss_xHII(self, output, target):
        return torch.mean(torch.abs(1.0-torch.div(output, target[:, :-1])))

    def loss_tau(self, tau_pred, target):
        return torch.mean(torch.abs(1.0 - torch.div(tau_pred, target[:, -1])))
    
    def test_tau(self, dataset:DataSet):

        self.set_check_metadata_and_partition(dataset, check_only = True)
        x_test   = torch.tensor(dataset.x_array[dataset.partition.early_test],     dtype=torch.float32)
        tau_test = torch.tensor(dataset.y_regressor[dataset.partition.early_test, -1], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred   = self.forward(x_test)
            tau_pred = self.tau_ion(x_test, y_pred)

        return (1.0-tau_pred/tau_test).numpy()

    def test_xHII(self, dataset:DataSet):

        self.set_check_metadata_and_partition(dataset, check_only = True)
        x_test = torch.tensor(dataset.x_array[dataset.partition.early_test],          dtype=torch.float32)
        y_test = torch.tensor(dataset.y_regressor[dataset.partition.early_test, :-1], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred = self.forward(x_test)

        return y_pred.numpy(), y_test.numpy()
    
    @property
    def alpha_tau(self):
        return self._alpha_tau

    
    
    
def train_regressor(model: Regressor, dataset: DataSet, optimizer:torch.optim.Optimizer, *, epochs = 50, learning_rate = 1e-3, verbose = True, batch_size = 64, **kwargs):
    
    # set the metadata and parition object of the model
    model.set_check_metadata_and_partition(dataset)

    # format the data for the regressor
    train_dataset = TorchDataset(dataset.x_array[dataset.partition.early_train], dataset.y_regressor[dataset.partition.early_train])
    valid_dataset = TorchDataset(dataset.x_array[dataset.partition.early_valid], dataset.y_regressor[dataset.partition.early_valid])
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
            y_pred   = model.forward(x_batch)
            tau_pred = model.tau_ion(x_batch, y_pred)
            
            loss_xHII = model.loss_xHII(y_pred, y_batch)
            loss_tau  = model.loss_tau(tau_pred, y_batch)
            loss     = (1.0-model.alpha_tau) * loss_xHII + model.alpha_tau * loss_tau

            loss.backward()
            optimizer.step()
            
            train_loss     = np.append(train_loss, loss.item())
            train_accuracy = np.append(train_accuracy, 1-loss_tau.item())


        # evaluation mode
        model.eval()
    
        with torch.no_grad():
            
            for batch in valid_loader:
                x_batch, y_batch = batch
            
                y_pred = model(x_batch)              # forward pass
                tau_pred = model.tau_ion(x_batch, y_pred)
                
                loss_xHII = model.loss_xHII(y_pred, y_batch)
                loss_tau  = model.loss_tau(tau_pred, y_batch)
                loss      = (1.0-model.alpha_tau) * loss_xHII + model.alpha_tau * loss_tau
                
                valid_loss     = np.append(valid_loss, loss.item())
                valid_accuracy = np.append(valid_accuracy, 1-loss_tau.item())
        
        # get the mean of all batches
        model._train_loss     = np.append(model._train_loss, np.mean(train_loss))
        model._valid_loss     = np.append(model._valid_loss, np.mean(valid_loss))
        model._train_accuracy = np.append(model._train_accuracy, np.mean(train_accuracy))
        model._valid_accuracy = np.append(model._valid_accuracy, np.mean(valid_accuracy))

        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], loss: ({model.train_loss[-1]:.4f}, {model.valid_loss[-1]:.4f}), accuracy = ({model.train_accuracy[-1]:.4f}, {model.valid_accuracy[-1]:.4f})')   

