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

##################
#
# General neural network framework
#
##################

import numpy as np
import torch

import os
from copy import deepcopy
from os.path import join

from .data import DataSet, MetaData, DataPartition


class NeuralNetwork(torch.nn.Module):
    """
    A class wrapping torch.nn.Module for neural network models

    Attributes
    ----------
    - name : str
        the name of the model

    Methods
    -------
    save(path, save_partition)
        
    """


    def __init__(self, name: str) -> None:

        self._name: str = name

        self._metadata:  (MetaData | None)      = None
        self._partition: (DataPartition | None) = None

        self._train_loss     = np.zeros(0)
        self._valid_loss     = np.zeros(0)
        self._train_accuracy = np.zeros(0)
        self._valid_accuracy = np.zeros(0)

        self._struct         = np.empty(0)

        print('Ininitated model ' + str(self.name))


    def save(self, path = ".", save_partition = True) -> None:
        """
        save the neural network model as a .pth file

        if save_partition is false the partitioning of the data into
        train, valid and test is not save (useless for instance once
        we have a fully trained model that we just want to use)
        """

        # when partition is not required only print empty arrays
        if save_partition is False:
            partition       = deepcopy(self._partition)
            self._partition = DataPartition(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

        # putting the model in eval mode
        self.eval()
        
        if len(self.struct) == 0:
            # saving the full class as a pickled object
            torch.save(self, join(path, self._name + ".pth"))
        else:
            # saving the state of the weights (recommended)
            torch.save(self._model.state_dict(), join(path, self._name + "_weights.pth"))

            # add extra information for the structure of the model
            with open(join(path, self._name + "_struct.npy"), 'wb') as file:
                np.save(file, self._struct, allow_pickle=False)

            # add extra information about the metadata used for training
            self.metadata.save(join(path, self._name + "_metadata"))

            # add extra information about the partition used for training
            self.partition.save(join(path, self._name + "_partition"))

            # add extra information about loss and accuracy during training
            with open(join(path, self._name + "_loss.npz"), 'wb') as file:
                np.savez(file, 
                         train_loss = self._train_loss,
                         train_accuracy = self._train_accuracy,
                         valid_loss = self._valid_loss,
                         valid_accuracy = self._valid_accuracy)

        # put the partition back to its original value
        if save_partition is False:
            self._partition = partition


    def load_extras(self, path):
        
        if os.path.isfile(path + '_weights.pth') \
            and os.path.isfile(path + '_partition.npz') \
            and os.path.isfile(path + '_metadata.npz') \
            and os.path.isfile(path + '_loss.npz'):

            # set the weights of the model
            weights = torch.load(path  + '_weights.pth', weights_only=True)
            self._model.load_state_dict(weights)

            # fetch the partition and the metadata used during training
            self._partition = DataPartition.load(path  + '_partition')
            self._metadata  = MetaData.load(path  + '_metadata')
        
            # get the loss and accuracy obtained during training
            with open(path  + '_loss.npz', 'rb') as file:
                data = np.load(file)
                self._train_loss     = data.get('train_loss')
                self._train_accuracy = data.get('train_accuracy')
                self._valid_loss     = data.get('valid_loss')
                self._valid_accuracy = data.get('valid_accuracy')

            return None
        
        raise ValueError("Could not find a fully saved model at: " + path)


    def set_check_metadata_and_partition(self, dataset: DataSet, check_only = False):
        """
        set and check the medatada and partition attributes
        
        raise a ValueError is the dataset is incompatible with the
        current metadata or partition
        """
        
        # set and check the metadata
        if self.metadata is None and not check_only:
            self._metadata = dataset.metadata
        else:
            if self.metadata != dataset.metadata:
                raise ValueError("The metadata is incompatible with the previous round of training.")
        
        # set and check the partition
        if self.partition is None and not check_only:
            self._partition = dataset.partition
        else:
            if self.partition != dataset.partition:
                raise ValueError("The partition is incompatible with the previous round of training.")
            

    def print_parameters(self):
        
        total_params = 0
        print("| Parameters per layers:")
        print("| ----------------------")
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            print('|', name, ':', params)
            total_params += params
        print("| ----------------------")
        print(f"| Total Trainable Params: {total_params}")
        print("  ----------------------")
            

    @property
    def name(self):
        return self._name

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
    
    @property
    def metadata(self):
        return self._metadata
    
    @property
    def partition(self):
        return self._partition
    
    @property
    def struct(self):
        return self._struct