import numpy as np
import torch
import torch.nn as nn

from .data      import TorchDataset, DataSet, uniform_to_true
from .nn        import NeuralNetwork
from .cosmology import optical_depth_no_rad_torch


class Regressor(NeuralNetwork):

    def __init__(self, *, n_input = 16, dim = 80, model = None, name = None):

        if name is None:
            name = "DefaultRegressor"

        if model is None:

              model = nn.Sequential(
                    nn.Linear(n_input, dim),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, 50),
                    )
        
        super(Regressor, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        self._model    = model
        

    @classmethod
    def load(cls, path = "./DefaultRegressor.pth"):
        return torch.load(path)
        
    def forward(self, x):
        return torch.clamp(self._model(x), max=1.0)
    
    def tau_ion(self, x, y):
        z_tensor = torch.tensor(self.metadata.z, dtype=torch.float32)
        omega_b  = uniform_to_true(x[:, self.metadata.pos_omega_b], self.metadata.min_omega_b, self.metadata.max_omega_b)
        omega_c  = uniform_to_true(x[:, self.metadata.pos_omega_c], self.metadata.min_omega_c, self.metadata.max_omega_c)
        hlittle  = uniform_to_true(x[:, self.metadata.pos_hlittle], self.metadata.min_hlittle, self.metadata.max_hlittle)
        return optical_depth_no_rad_torch(z_tensor, y, omega_b, omega_c, hlittle)
    
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

    def test_xHII(self, dataset:DataSet, i = None):

        self.set_check_metadata_and_partition(dataset, check_only = True)
        x_test = torch.tensor(dataset.x_array[dataset.partition.early_test],     dtype=torch.float32)
        y_test = torch.tensor(dataset.y_regressor[dataset.partition.early_test, :-1], dtype=torch.float32)
        
        self.eval()
        
        with torch.no_grad():
            y_pred   = self.forward(x_test)

        return y_pred.numpy(), y_test.numpy()

    
    
    
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
            loss     = 0.5 * (loss_xHII + loss_tau)

            loss.backward()
            optimizer.step()
            
            train_loss     = np.append(train_loss, loss.item())
            train_accuracy = np.append(train_accuracy, loss_tau.item())


        # evaluation mode
        model.eval()
    
        with torch.no_grad():
            
            for batch in valid_loader:
                x_batch, y_batch = batch
            
                y_pred = model(x_batch)              # forward pass
                tau_pred = model.tau_ion(x_batch, y_pred)
                
                loss_xHII = model.loss_xHII(y_pred, y_batch)
                loss_tau  = model.loss_tau(tau_pred, y_batch)
                loss     = 0.5 * (loss_xHII + loss_tau)
                
                valid_loss     = np.append(valid_loss, loss.item())
                valid_accuracy = np.append(valid_accuracy, loss_tau.item())
        
        # get the mean of all batches
        model._train_loss     = np.append(model._train_loss, np.mean(train_loss))
        model._valid_loss     = np.append(model._valid_loss, np.mean(valid_loss))
        model._train_accuracy = np.append(model._train_accuracy, np.mean(train_accuracy))
        model._valid_accuracy = np.append(model._valid_accuracy, np.mean(valid_accuracy))

        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], loss: ({model.train_loss[-1]:.4f}, {model.valid_loss[-1]:.4f}), accuracy = ({model.train_accuracy[-1]:.4f}, {model.valid_accuracy[-1]:.4f})')   

