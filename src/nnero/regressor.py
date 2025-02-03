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
    """
    Daughter class of :py:class:`NeuralNetwork` specialised for regressors.
    
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
    n_input: int, optional
        Number of output on the neural network 
        (corresponds to the number of redshift bins).
        Default is 50. Overriden if `dataset` is specified.
    n_hidden_features: int, optional
        Number of hidden features per layer. Default is 80.
    n_hidden_layers: int, optional
        Number of layers. Default is 5.
    name: str | None
        Name of the neural network. If None, automatically set to DefaultClassifier.
        Default is None.
    dataset: Dataset | None
        Dataset on which the model will be trained. 
        If provided, gets `n_input` and `n_output` from the data and overrides the user input value.
    use_pca: bool, optional
        If `True`, decompose the interpolated function on the principal component eigenbasis.
        Default is True.
    pca_precision: float, optional
        If `use_pca` is `True` sets how many eigenvalues needs to be considered.
        Only consider eigenvectors with eigenvalues > precision^2 * max(eigenvalues).
        Default is 1e-3.
    alpha_tau: float, optioal
        Weighting of the relative error on X_e and optical depth in the cost function.
        Default is 0.5
        
    Attributes
    ----------
    - name : str
        the name of the model
    """

    def __init__(self, 
                 *, 
                 n_input: int = 16, 
                 n_output: int = 50, 
                 n_hidden_features: int = 80, 
                 n_hidden_layers: int = 5, 
                 alpha_tau : float = 0.5,
                 model = None, 
                 name: str | None = None,
                 use_pca: bool = True,
                 pca_precision: float = 1e-3,
                 dataset: DataSet | None = None):

        if name is None:
            name = "DefaultRegressor"

        # save as attribute wether or not we want to work
        # in the principal component eigenbasis
        self._use_pca = use_pca
        self._pca_precision = pca_precision
        
        # if we provide a dataset with pca_precision we override
        # the number of output to correspond to the number of
        # usefull eigenvectors otherwise it is set to the number 
        # of redshift bins used 
        # if no dataset is given, need to know in advance how
        # many usefull eigenvectors there are and pass is as n_output
        # ----
        # this also initialises the pca vectors in the metadata 
        # attribute of the dataset object
        # ----
        # when loading from a file we do not provide a dataset
        # and therefore do not redo the initialisation of the pca,
        # pca eigenvectors and all are read from the saved metadata
        if dataset is not None:
            
            n_input = len(dataset.metadata.parameters_name)

            if use_pca is True:
                n_output = dataset.init_principal_components(self.pca_precision)
            else:
                n_output = len(dataset.metadata.z)
            
            
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
            model = nn.Sequential(nn.Linear(n_input, n_hidden_features), *hidden_layers, nn.Linear(n_hidden_features, n_output))
        
            # save the structure of this sequential model and more
            struct = np.array([n_input, n_output, n_hidden_features, n_hidden_layers, alpha_tau, use_pca, pca_precision])
        
        # call the (grand)parent constructors
        super(Regressor, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        # structure of the model
        self._struct = struct
        
        # define the model
        self._model = model

        # define parameters in the loss
        self._alpha_tau = alpha_tau

        # if the dataset is already given, set it as the dataset of the network
        if dataset is not None:
            self.set_check_metadata_and_partition(dataset)

    


    @classmethod
    def load(cls, path: str | None = None):
        """
        Loads a regressor.

        Parameters
        ----------
        path: str | None
            Path to the saved files containing the regressor data.
            If None automatically fetch the DefaultRegressor.

        Returns
        -------
        Regressor
        """

        if path is None:
            path = os.path.join(DATA_PATH, "DefaultRegressor")

        name = path.split('/')[-1]
        
        if os.path.isfile(path  + '_struct.npy'):

            with open(path  + '_struct.npy', 'rb') as file:
                struct  = np.load(file)

                if len(struct) == 7:

                    regressor = Regressor(n_input=int(struct[0]), 
                                          n_output=int(struct[1]), 
                                          n_hidden_features=int(struct[2]), 
                                          n_hidden_layers=int(struct[3]),
                                          alpha_tau=struct[4],
                                          use_pca=bool(struct[5]),
                                          pca_precision=struct[6],
                                          name=name)
                    regressor.load_weights_and_extras(path)
                    regressor.eval()

                    print('Model ' + str(name) + ' sucessfully loaded')

                    return regressor
        
        # if the struct read is not of the right size
        # check for a pickled save of the full class
        # (although this is not recommended)
        if os.path.isfile(path  + '.pth') :
            regressor = torch.load(path + ".pth")
            regressor.eval()

            print('Model ' + str(name) + ' sucessfully loaded from a .pth archive')
            
            return regressor
        
        raise ValueError("Could not find a fully saved regressor model at: " + path)

        
    def forward(self, x):
        """
        Forward evaluation of the model.

        Parameters
        ----------
        x: torch.Tensor
            input features
        """

        if self.use_pca is True:
            # reconstruct the value of the function in the principal component eigenbasis
            y = torch.matmul(self._model(x), self.metadata.torch_pca_eigenvectors[0:self.metadata.pca_n_eigenvectors, :])
            y = y + self.metadata.torch_pca_mean_values
            y = 10**y
        else:
            y = self._model(x)
        return torch.clamp(y, max=1.0)
    
    def tau_ion(self, x, y):
        """
        Optical depth to reionization.

        Parameters
        ----------
        x: torch.Tensor
            Input features.
        y: torch.Tensor
            Output of the Regressor. Corresponds to X_e(z).
        """

        z_tensor = torch.tensor(self.metadata.z, dtype=torch.float32)
        omega_b  = uniform_to_true(x[:, self.metadata.pos_omega_b], self.metadata.min_omega_b, self.metadata.max_omega_b)
        omega_c  = uniform_to_true(x[:, self.metadata.pos_omega_dm], self.metadata.min_omega_c, self.metadata.max_omega_c)
        hlittle  = uniform_to_true(x[:, self.metadata.pos_hlittle], self.metadata.min_hlittle, self.metadata.max_hlittle)
        return optical_depth_no_rad(z_tensor, y, omega_b, omega_c, hlittle)
    
    def loss_xHII(self, output, target):
        return torch.mean(torch.abs(1.0-torch.div(output, target[:, :-1])))

    def loss_tau(self, tau_pred, target):
        return torch.mean(torch.abs(1.0 - torch.div(tau_pred, target[:, -1])))
    
    def test_tau(self, dataset:DataSet) -> np.ndarray:
        """
        Test the efficiency of the regressor to reconstruct the optical depth to reionization.

        Parameters
        ----------
        dataset : DataSet
            DataSet containing the training partition and the test partition.

        Returns
        -------
        np.ndarray
            Distribution of relative error between the predicted and true optical depth. 
            Array with the size of the test dataset.
        """

        self.set_check_metadata_and_partition(dataset, check_only = True)
        x_test   = torch.tensor(dataset.x_array[dataset.partition.early_test],     dtype=torch.float32)
        tau_test = torch.tensor(dataset.y_regressor[dataset.partition.early_test, -1], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred   = self.forward(x_test)
            tau_pred = self.tau_ion(x_test, y_pred)

        return (1.0-tau_pred/tau_test).numpy()

    def test_xHII(self, dataset:DataSet) -> tuple[np.ndarray, np.ndarray]:
        """
        Test the efficiency of the regressor to reconstruct the free electron fraction X_e.

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
        x_test = torch.tensor(dataset.x_array[dataset.partition.early_test],          dtype=torch.float32)
        y_test = torch.tensor(dataset.y_regressor[dataset.partition.early_test, :-1], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred = self.forward(x_test)

        return y_pred.numpy(), y_test.numpy()
    
    @property
    def alpha_tau(self):
        return self._alpha_tau

    @property
    def use_pca(self):
        return self._use_pca
    
    @property
    def pca_precision(self):
        return self._pca_precision
    
    @property
    def z(self):
        return self._metadata.z
    

    
def train_regressor(model: Regressor, 
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
    
    # if we use pca but have not yet initialised it in metadata do it at first training step
    # this can happen if no dataset is given to initialise the model while havind use_pca = True
    if (len(model.train_loss) == 0) and (model.use_pca is True) and (len(dataset.metadata.pca_eigenvalues) == 0):
        dataset.init_principal_components(model.pca_precision)

    # set or check the metadata and parition attributes of the model
    model.set_check_metadata_and_partition(dataset)

    # format the data for the regressor
    train_dataset = TorchDataset(dataset.x_array[dataset.partition.early_train], dataset.y_regressor[dataset.partition.early_train])
    valid_dataset = TorchDataset(dataset.x_array[dataset.partition.early_valid], dataset.y_regressor[dataset.partition.early_valid])
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
            
                y_pred = model.forward(x_batch)              # forward pass
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

