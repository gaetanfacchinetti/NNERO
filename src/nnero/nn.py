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

from copy import deepcopy
from os.path import join

from .data import DataSet, MetaData, DataPartition


class NeuralNetwork(torch.nn.Module):

    def __init__(self, name: str) -> None:

        self._name: str = name

        self._metadata:  (MetaData | None)      = None
        self._partition: (DataPartition | None) = None

        self._train_loss     = np.zeros(0)
        self._valid_loss     = np.zeros(0)
        self._train_accuracy = np.zeros(0)
        self._valid_accuracy = np.zeros(0)

        print('Ininitated model ' + str(self.name))


    def save(self, path = ".", save_partition = True) -> None:
        """
            save the neural network model as a .pth file

        is save_partition is false the partitioning of the data into
        train, valid and test is not save (useless for instance once
        we have a fully trained model that we just want to use)
        """

        if save_partition is False:
            _s_partition    = deepcopy(self._partition)
            self._partition = DataPartition(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))

        self.eval()
        torch.save(self, join(path, self._name + ".pth"))

        if save_partition is False:
            self._partition = _s_partition


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