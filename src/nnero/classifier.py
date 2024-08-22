import numpy as np
import torch
import torch.nn as nn

from .data import TorchDataset


class NeuralNetwork(nn.Module):

    def __init__(self, name):
        self._name = name
        self._metadata  = None

    def save(self, path = ".") -> None:
        self.eval()
        torch.save(self,  path + "/" + self._name + ".pth")


class Classifier(NeuralNetwork):

    def __init__(self, n_input = 16, *, model = None, dim = 32, name = "DefaultClassifier", check_name = True):

        
        if check_name and (name == "DefaultClassifier" and (model is not None)):
            raise AttributeError("Plase give your custom model a name different from the default value")

        if model is None:

            model = nn.Sequential(
                nn.Linear(n_input, dim),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
                nn.Linear(dim, 1), nn.Sigmoid()
                )


        self._train_loss     = np.zeros(0)
        self._valid_loss     = np.zeros(0)
        self._train_accuracy = np.zeros(0)
        self._valid_accuracy = np.zeros(0)

        
        super(Classifier, self).__init__(name)
        super(NeuralNetwork, self).__init__()

        self._model = model
        self._loss_fn = nn.BCELoss()

                
    @classmethod
    def load(cls, path = "./DefaultClassifier.pth"):
        return torch.load(path)
        
    def forward(self, x):
        return torch.flatten(self._model(x))
    
    @property
    def loss_fn(self):
        return self._loss_fn

    @property
    def train_loss(self):
        return self._train_loss
    
    @property
    def valid_loss(self):
        return self._valid_loss
    
    @property
    def train_accuracy(self):
        return self._train_accuracy
    
    @property
    def valid_accuracy(self):
        return self._valid_accuracy

    
    
    
def train_classifier(model, dataset, optimizer, *, epochs = 50, learning_rate = 1e-3, verbose = True, batch_size = 64, **kwargs):
    
    # format the data for the classifier
    train_dataset = TorchDataset(dataset.x_array[dataset.indices_tot_train], dataset.y_classifier[dataset.indices_tot_train])
    valid_dataset = TorchDataset(dataset.x_array[dataset.indices_tot_valid], dataset.y_classifier[dataset.indices_tot_valid])
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
            loss = model.loss_fn(y_pred, y_batch)
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

