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

from .data import DataSet, MetaData, DataPartition, MP_KEY_CORRESPONDANCE


class NeuralNetwork(torch.nn.Module):
    """
    A class wrapping py:class:`torch.nn.Module` for neural network models

    Parameters
    ----------
    name: str
        name of the neural network

    Attributes
    ----------
    name: str
        the name of the model
    metadata: Metadata
        metadata on which the model is trained
    partition: DataPartition
        partitioning of the data on which the model is trained
    train_loss: np.ndarray
        1D array training loss for each training epoch
    valid_loss: np.ndarray
        1D array validation losses for each training epoch
    train_accuracy: np.ndarray
        1D array training accuracy for each training epoch
    valid_accuracy: np.ndarray
        1D array validation accuracy for each training epoch
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

        


    def save(self, path: str = ".", save_partition: bool = True) -> None:
        """
        Save the neural network model in a bunch of files.

        Parameters
        ----------
        path: str, optional
            path where to save the neural network
            -- default is the current directory "."
        save_partition: bool, optional
            if save_partition is false the partitioning of the data into
            train, valid and test is not saved (useless for instance once
            we have a fully trained model that we just want to use)
            -- default is True
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


    def load_weights_and_extras(self, path: str) -> None:
        """
        loads the network weights and extra information

        Parameters
        ----------
        path: str
            path to the network to load

        Raises
        ------
        ValueError
            If not all necessary files exists where path points.
        """
        
        # check if the needed files exist
        if os.path.isfile(path + '_weights.pth') and os.path.isfile(path + '_metadata.npz') :

            # set the weights of the model
            weights = torch.load(path  + '_weights.pth', weights_only=True)
            self._model.load_state_dict(weights)

            # fetch the metadata used during training
            self._metadata  = MetaData.load(path  + '_metadata')

            try:
                # fetch the partition used during training
                self._partition = DataPartition.load(path  + '_partition')
                
                # get the loss and accuracy obtained during training
                with open(path  + '_loss.npz', 'rb') as file:
                    data = np.load(file)
                    self._train_loss     = data.get('train_loss')
                    self._train_accuracy = data.get('train_accuracy')
                    self._valid_loss     = data.get('valid_loss')
                    self._valid_accuracy = data.get('valid_accuracy')
            except:
                # partition and saved loss are not necessary for the network
                # to work properly so we can simply pass if it does not work
                pass

            return None
        
        raise ValueError("Could not find a fully saved model at: " + path)


    def set_check_metadata_and_partition(self, dataset: DataSet, check_only: bool = False) -> None:
        """
        set and check the medatada and partition attributes
        
        Parameters
        ----------
        dataset: DataSet
            dataset to compare or to assign to the object
        check_only: bool, optional
            option to only compare the compatibility
            -- default is False
        
        Raises
        ------
        ValueError 
            if the dataset is incompatible with the current metadata or partition
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
            

    def print_structure(self):
        """
        prints the list of parameters in the model
        """
        
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
            

    def info(self):
        
        print("#####################")
        print("Structure of the network")
        self.print_structure()
        print("#####################")
        print("Parameters:")
        inv_corres  = {value:key for key, value in MP_KEY_CORRESPONDANCE.items()}
        param_range = self.parameters_range
        other_name  = [inv_corres[param] for param in self.parameters_name]
        param_info = []
        max_param_len = 0
        for ip, param in enumerate(self.parameters_name):
            param_info.append(f"{param} ({str(other_name[ip])})")
            if len(param_info[-1]) > max_param_len:
                max_param_len = len(param_info[-1])

        for ip, param in enumerate(self.parameters_name):   
            print(f"-> {param_info[ip]:<{max_param_len}} in [{param_range[ip, 0]}, {param_range[ip, 1]}]")
        print("#####################")




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
    
    @property
    def parameters_name(self):
        return self._metadata.parameters_name
    
    @property
    def parameters_range(self):
        return np.vstack((self._metadata.parameters_min_val, self._metadata.parameters_max_val)).T
