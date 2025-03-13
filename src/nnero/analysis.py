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
# Definition of some functions to analyse chains and plot them
#
##################



import glob
import warnings
import os

import numpy as np
from copy import copy

from scipy.ndimage import gaussian_filter, gaussian_filter1d


from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

from .data         import MP_KEY_CORRESPONDANCE
from .predictor    import DEFAULT_VALUES
from .regressor    import Regressor
from .classifier   import Classifier
from .interpolator import Interpolator
from .predictor    import predict_Xe_numpy, predict_tau_numpy, predict_parameter_numpy


EMCEE_IMPORTED = False

try:
    import emcee
    EMCEE_IMPORTED = True
except:
    pass


def neutrino_masses(mnu_1, mnu_2 = 0.0, mnu_3 = 0.0, hierarchy = 'NORMAL'):

    ## DEFINE NEUTRINO MASSES BEFORE ANYTHING ELSE
    delta_m21_2 = 7.5e-5
    delta_m31_NO_2 = 2.55e-3
    delta_m31_IO_2 = 2.45e-3
    

    # degenerate neutrino masses (all equal to the first one)
    if hierarchy == 'DEGENERATE':
        mnu_2 = mnu_1
        mnu_3 = mnu_1

    # normal ordering
    if hierarchy == "NORMAL":
        mnu_2 = np.sqrt(mnu_1**2 + delta_m21_2) 
        mnu_3 = np.sqrt(mnu_1**2 + delta_m31_NO_2)

    # inverse ordering
    if hierarchy == "INVERSE":
        mnu_2 = np.sqrt(mnu_1**2 + delta_m31_IO_2) 
        mnu_3 = np.sqrt(mnu_1**2 + delta_m31_IO_2 + delta_m21_2)

    # create an array of neutrino masses
    m_neutrinos = np.array([mnu_1, mnu_2, mnu_3])
    return m_neutrinos


def to_CLASS_names(array: list[str] | np.ndarray):

    is_array = isinstance(array, np.ndarray)

    labels_correspondance = {value : key for key, value in MP_KEY_CORRESPONDANCE.items()}    
    array = [labels_correspondance[value] if value in labels_correspondance else value for value in array]

    if is_array is True:
        array = np.array(array)

    return array
    

def to_21cmFAST_names(array: list[str] | np.ndarray):

    is_array = isinstance(array, np.ndarray)
    array = [MP_KEY_CORRESPONDANCE[value] if value in MP_KEY_CORRESPONDANCE else value for value in array]

    if is_array is True:
        array = np.array(array)

    return array

#################################################
## CHAIN ANALYSIS TOOLS


class MPChain:
    """
    Class MPChain reading chains from MontePython output files

    Parameters
    ----------
    filename : str
        Path to the the file where the chain is stored.
    """

    def __init__(self, filename: str):
        
        self._filename: str = filename
        self.load()

    
    def load(self) -> None:
        """
        Load the chain and automatically remove the non-markovian points.
        """

        # read the file to get the chain
        with open(self._filename, 'r') as f:
            data = np.loadtxt(f) 

            self.weights = data[:, 0]
            self.lnlkl   = - data[:, 1]
            self._values = data[:, 2:].T

            self._total_n_steps = np.sum(self.weights, dtype=int)
            self._total_length  = self._values.shape[1]


        # reread the file to find the non markovian part of the chain
        with open(self._filename, 'r') as f:

            self._markov_index = 0
            for line in f:
                if (line.strip().startswith('#')) and ('update proposal' in line):
                    self._markov_index = int(line.split(' ')[2])

            # remove the non markovian part by default
            self._values = self._values[:, self._markov_index:]
            self.weights = self.weights[self._markov_index:]
            self.lnlkl = self.lnlkl[self._markov_index:]


        self._max_lnlkl = np.max(self.lnlkl)
        index_max_lnlkl = np.argmax(self.lnlkl)
        self._best_fit = self._values[:, index_max_lnlkl]

        self._n_params = self._values.shape[0]
        self._mean_value: np.ndarray = np.zeros(self._n_params)
        self._var_value:  np.ndarray = np.zeros(self._n_params)


    def remove_burnin(self, global_max_lnlkl: float) -> None:
        """
        Remove the burnin points according to the value of the maximum
        log likelihood over all chains. Only points of the chain that are 
        after its overcrossing of global_max_lnlkl - 3 are kept.

        Parameters
        ----------
        global_max_lnlkl : float
            Global maximum log likelihood over all chains
        """

        if np.all(self.lnlkl < (global_max_lnlkl - 3)):
            self._burnin_index = len(self.lnlkl)
        else:
            burnin_index = np.where(self.lnlkl >= global_max_lnlkl - 3)[0]

            if len(burnin_index) > 0:
                self._burnin_index = burnin_index[0]
            else:
                self._burnin_index = 0

        self._values  = self._values[:, self._burnin_index:]
        self.weights  = self.weights[self._burnin_index:]
        self.lnlkl    = self.lnlkl[self._burnin_index:]

        
    def values(self, discard: int = 0, thin: int = 1) -> np.ndarray:
        """
        Get the values of the chain.

        Parameters
        ----------
        discard : int, optional
            Number of initial points to discard, by default 0.
        thin : int, optional
            Thining factor (taking only one value every value of thin), by default 1.

        Returns
        -------
        np.ndarray with dimension (number of parameters, length of chain)
        """
        
        if discard > self._values.shape[-1]:
            discard = self._values.shape[-1]
            warnings.warn("All points in chain " + self._filename +  " discarded (discard >= chain length)")
        
        return self._values[:, discard::thin]
    
    def compute_stats(self) -> None:
        """
        Compute the mean and standard deviation within the chain.
        Should be called after `remove_burnin()`. 
        """

        n = np.sum(self.weights)

        self._mean_value = np.sum(self._values * self.weights[None, :], axis=-1) / n 
        self._var_value  = (np.sum((self._values**2) * self.weights[None, :], axis=-1) - n * (self._mean_value)**2) / (n-1)


    @property
    def markov_index(self):
        return self._markov_index

    @property
    def max_lnlkl(self):
        return self._max_lnlkl
    
    @property
    def burnin_index(self):
        return self._burnin_index
    
    @property
    def best_fit(self):
        return self._best_fit
    
    @property
    def mean_value(self):
        return self._mean_value
    
    @property
    def var_value(self):
        return self._var_value
    
    @property
    def length(self) -> int:
        """
        Number of accepted steps not counting burnin and non markovian points.

        Returns
        -------
        int
        """
        return self._values.shape[1]
    
    @property
    def total_length(self) -> int:
        """
        Total number of accepted steps

        Returns
        -------
        int
        """
        return self._total_length


    @property
    def total_n_steps(self) -> int:
        """
        Total number of steps.

        Returns
        -------
        int
        """
        return self._total_n_steps
    
    @property
    def n_steps(self) -> int:
        """
        Number of steps not counting burnin and non markovian points.

        Returns
        -------
        int
        """
        return np.sum(self.weights, dtype=int)
    
    @property
    def n_params(self):
        return self._n_params
    
    @property
    def acceptance_rate(self):
        return self.total_length/self.total_n_steps




class Samples(ABC):
    """
    Class containing all chains of a MCMC analysis
    
    Parameters
    ----------
    path : str
        Path to the chains.
    ids : list | np.ndarray | None, optional
        List of chains to take into accoung. If none all possible found chains are added. By default None.

    """

    def __init__(self, path : str, ids: list[int] | np.ndarray | None = None) -> None:
        
        self._path = path
        self._ids  = np.array(ids) if ids is not None else None

    @abstractmethod
    def flat(self, discard: np.ndarray | None = None, thin: None | int = None, **kwargs) -> np.ndarray:
        pass

    @property
    def path(self):
        return self._path
    
    @property
    def ids(self):
        return self._ids
    
    @property
    @abstractmethod
    def param_names(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def scaling_factor(self) -> dict:
        pass


@dataclass
class GaussianInfo:

    mean: np.ndarray | None = None
    cov:  np.ndarray | None = None
    param_names: np.ndarray | list[str] | None = None

    def compatible_with(self, other: Self) -> bool:
        
        all_params = set(list(to_CLASS_names(self.param_names)) + list(to_CLASS_names(other.param_names)) )
       
        for param in all_params:
            if param in to_CLASS_names(self.param_names) and param in to_CLASS_names(other.param_names):
                index_1 = list(to_CLASS_names(self.param_names)).index(param)
                index_2 = list(to_CLASS_names(other.param_names)).index(param)

                if self.mean is not None and other.mean is not None:
                    if self.mean[index_1] != other.mean[index_2]:
                        return False 
                
        return True


class GaussianSamples(Samples):
    """
    Daughter class of Samples for gaussian generated chains.
    """

    def __init__(self, 
                 gaussians: list[GaussianInfo | str] | GaussianInfo | str,
                 add_tau: bool = False,
                 params: list[str] | None = None,
                 *, 
                 n: int = 200_000) -> None:

        super().__init__("", None)

        self._add_tau = add_tau

        # define a list of extra gaussian to add on top
        self._gaussians = gaussians
        if (not isinstance(self._gaussians, list)):
            self._gaussians = [self._gaussians]

        # if given string input read the data
        # and check for compatibility between mean values
        for ig in range(len(self._gaussians)):
            if isinstance(self._gaussians[ig], str):
                self._gaussians[ig] = self.load_data(self._gaussians[ig])
            
            if not self._gaussians[0].compatible_with(self._gaussians[ig]):
                raise ValueError('fiducial parameters should agree')

        # parameters on that we will be contained 
        # in the total covariance matrix
        self._params = params

        if self._params is None and self._gaussians is not None:
            self._params = self._gaussians[0].params

        self._params = to_CLASS_names(self._params)

        # get all the parameters in total
        all_params = []
        for g in self._gaussians:
            for p in g.param_names:
                if p not in all_params:
                    all_params.append(p)

        # build a big inverse covariance matrix
        # that contains all parameters in the input
        # covariance matrices
        all_n = len(all_params)
        all_inv_cov = np.zeros((all_n, all_n))
        all_mean    = np.zeros(all_n)

        for g in self._gaussians:
            this_params = list(to_CLASS_names(g.param_names))
            indices     = [all_params.index(param) for param in this_params]
            all_inv_cov[np.ix_(indices, indices)] += np.linalg.inv(g.cov)
            
            if g.mean is not None:
                all_mean[indices] = g.mean


        # select the part of the covariance matrix for the desired parameters
        all_cov        = np.linalg.inv(all_inv_cov)
        self._indices  = [all_params.index(param) for param in self._params]

        # construct a GaussianInfo object from the total cov and mean defined above
        self._gaussian       = GaussianInfo(all_mean[self._indices], all_cov[np.ix_(self._indices, self._indices)], self._params)
        self._generated_data = np.random.multivariate_normal(all_mean, all_cov, n).T

        self._all_params = all_params

        # add tau_ion as a parameter
        if self._add_tau is True:
            self._params = np.array(['tau_reio'] + list(self._params))
        


    def load_data(self, filename) -> GaussianInfo:

        gaussian = GaussianInfo()

        with open(filename, 'rb') as file:
            data = np.load(file)
            gaussian.cov         = data.get('cov', None)
            gaussian.param_names = list(to_CLASS_names(data['params'])) if 'params' in data else None
            gaussian.mean        = data.get('fiducial', None)

            if gaussian.mean is None:
                gaussian.mean = data.get('mean', None)

        return gaussian

        

    def flat(self, discard: np.ndarray | None = None, thin: None | int = None, **kwargs) -> np.ndarray:
        
        if discard is None:
            discard = 0

        if thin is None:
            thin = 1
        
        flat_chain = self._generated_data[:, discard::thin]

        if self._add_tau is True:

            classifier = kwargs.get('classifier', None)
            regressor  = kwargs.get('regressor', None)

            if regressor is None:
                regressor = Regressor.load()

            mask = np.full_like(flat_chain, fill_value=True, dtype=bool)

            for ip, param in enumerate(to_CLASS_names(regressor.parameters_name)):
                if param in self._all_params: # only take from index 1 as tau is the zeroth
                    index = list(self._all_params).index(param)
                    val_range = regressor.parameters_range
                    mask[index, :] =  flat_chain[index, :] < val_range[ip, 1]
                    mask[index, :] =  (flat_chain[index, :] > val_range[ip, 0]) & mask[index, :]

            valid_columns = np.all(mask, axis=0)
            flat_chain = flat_chain[:, valid_columns]

            tau = compute_tau(flat_chain, self._all_params, classifier, regressor)

            return np.vstack((tau[None, :], flat_chain[self._indices, :]))

        return flat_chain[self._indices, :]

    @property
    def param_names(self) -> np.ndarray:
        return self._params
    
    @property
    def all_param_names(self) -> np.ndarray:
        return self._all_params

    @property
    def scaling_factor(self) -> dict:
        return {name: 1.0 for name in self.param_names}
    
    @property
    def gaussian(self) -> GaussianInfo:
        return self._gaussian


class EMCEESamples(Samples):
    """
    Daughter class of Samples for emcee chains.
    """

    def __init__(self, path : str, add_tau: bool = False) -> None:
        
        if EMCEE_IMPORTED is False:
            print("Cannot import emcee, therefore cannot work with emcee chains")
            return None
        
        super().__init__(path, None)


        self._add_tau = add_tau
        self.load_chains()

        # Check for convergence criterion
        self._has_converged = False
        self._autocorr      = None
        
        try:
            self._autocorr = self._reader.get_autocorr_time()
            self._has_converged = True
        except emcee.autocorr.AutocorrError:
            self._autocorr = self._reader.get_autocorr_time(quiet = True)
            pass
        
        

    def load_chains(self):
        
        self._reader = emcee.backends.HDFBackend(self.path)
        
        filename = os.path.join(*self.path.split('/')[:-1], *self.path.split('/')[-1].split('.')[:-1])

        with open(filename + '.npz', 'rb') as file:

            data = np.load(file)
            self._parameters_theta = data.get('parameters_theta', None)
            self._parameters_xi    = data.get('parameters_xi', None)
            self._xi               = data.get('xi', None)

        if self._add_tau is False:
            self._param_names = np.array(list(self._parameters_theta) + list(self._parameters_xi))
        else:
            self._param_names = np.array(['tau_reio'] + list(self._parameters_theta) + list(self._parameters_xi))
        


    def flat(self, discard: np.ndarray | None = None, thin: None | int = None, **kwargs) -> np.ndarray:
        
        burnin_discard = int(np.max(2*self._autocorr)) if self._autocorr is not None else 0

        if discard is None:
            discard = 0

        if thin is None:
            thin = 1


        flat_chain = self._reader.get_chain(discard=burnin_discard + discard, thin=thin, flat = True)
        flat_chain = np.hstack((flat_chain, np.tile(self._xi, (flat_chain.shape[0], 1)))).T

        if self._add_tau is True:

            classifier = kwargs.get('classifier', None)
            regressor  = kwargs.get('regressor', None)
            tau = compute_tau(flat_chain, self.param_names, classifier, regressor)

            flat_chain = np.vstack((tau[None, :], flat_chain))
            
        return flat_chain

    

    def convergence(self):

        if self._has_converged is False:
            print("Not converged yet")
            print(self._autocorr)

        
    @property
    def reader(self):
        return self._reader
    
    @property
    def autocorr(self) -> np.ndarray:
        return self._autocorr

    @property
    def param_names(self) -> np.ndarray:
        return self._param_names
    
    @property
    def scaling_factor(self) -> dict:
        return {name: 1.0 for name in self.param_names}


def save_sampling_parameters(filename: str, 
                             parameters_theta : list[str], 
                             parameters_xi: list[str], 
                             xi: np.ndarray):
    with open(filename.split('.')[0] + '.npz', 'wb') as file:

        np.savez(file, 
                 parameters_theta=to_CLASS_names(parameters_theta), 
                 parameters_xi=to_CLASS_names(parameters_xi), 
                 xi=xi)


def compute_tau(flat_chain: np.ndarray, 
                param_names: list[str] | np.ndarray,
                classifier: Classifier | None = None,
                regressor: Regressor | None = None) -> np.ndarray:

      

        # get the ordered list of parameters
        if regressor is None:
            regressor  = Regressor.load()
        if classifier is None:
            classifier = Classifier.load()

        r_parameters = to_CLASS_names(regressor.metadata.parameters_name)
        parameters = to_CLASS_names(copy(param_names))

        if isinstance(parameters, list):
            parameters = np.array(parameters)

        if 'tau_reio' in parameters:
            parameters = parameters[parameters != 'tau_reio']

        for param in parameters:
            
            if param == 'sum_mnu':
                param = 'm_nu1'

            # make a list of the given parameters that are needed for the interpolator
            # in the order they are
            data_for_param = []
            for ip, param in enumerate(parameters):
                
                if param == 'sum_mnu':
                    parameters[ip] = 'm_nu1'

                if param in r_parameters:
                    data_for_param.append(param)
            
            # get the data sample 
            data = np.empty((len(r_parameters), flat_chain.shape[-1])) 

            # find the ordering in which data_sample is set in prepare_data_plot
            indices_to_plot = [list(parameters).index(param) for param in data_for_param]

        for ip, param in enumerate(r_parameters): 
            
            # if we ran the MCMC over that parameter
            if param in parameters[indices_to_plot]:
                index = list(parameters[indices_to_plot]).index(param)
                data[ip, :] = flat_chain[index, :]
            else:
                data[ip, :] = DEFAULT_VALUES[to_21cmFAST_names([param])[0]]

        #return data 
        tau = predict_tau_numpy(data.T, classifier, regressor)

        return tau




def compute_parameter(flat_chain: np.ndarray, 
                      param_names: list[str] | np.ndarray,
                      classifier: Classifier | None = None,
                      interpolator: Regressor | None = None,
                      parameter: str | None = None) -> np.ndarray:

        
        # get the ordered list of parameters
        if interpolator is None:
            if interpolator is not None:
                interpolator = Interpolator.load("DefaultInterpolator_" + parameter)
            else:
                raise ValueError("Need to pass the parameter to interpolate to the predictor.")
        
        if classifier is None:
            classifier = Classifier.load()

        # parameters that the interpolator needs
        i_parameters = to_CLASS_names(interpolator.metadata.parameters_name)

        # parameters that are given in the sample
        parameters = to_CLASS_names(copy(param_names))

        if isinstance(parameters, list):
            parameters = np.array(parameters)

        # if tau_reio is given in the parameters we remove it
        if 'tau_reio' in parameters:
            flat_chain = flat_chain[parameters != 'tau_reio'] 
            parameters = parameters[parameters != 'tau_reio']
            

        # make a list of the given parameters that are needed for the interpolator
        # in the order they are
        data_for_param = []
        for ip, param in enumerate(parameters):
            
            if param == 'sum_mnu':
                parameters[ip] = 'm_nu1'

            if param in i_parameters:
                data_for_param.append(param)
        
        # get the data sample 
        data = np.empty((len(i_parameters), flat_chain.shape[-1])) 

        # find the ordering in which data_sample is set in prepare_data_plot
        indices_to_plot = [list(parameters).index(param) for param in data_for_param]

        for ip, param in enumerate(i_parameters): 
            
            # if we ran the MCMC over that parameter
            if param in parameters[indices_to_plot]:
                index = list(parameters[indices_to_plot]).index(param)
                data[ip, :] = flat_chain[index, :]
            else:
                data[ip, :] = DEFAULT_VALUES[param]

        return predict_parameter_numpy(data.T, classifier, interpolator, parameter)




class MPSamples(Samples):
    """
    Daughter class of Samples for MontePython chains.
    """
 

    def __init__(self, 
                 path: str, 
                 ids: list[int] | np.ndarray | None = None):
        
        super().__init__(path, ids)
        self._chains : list[MPChain] = []
        
        self.load_chains()
        self.load_paramnames()

        self._max_lnlkl  = np.max([chain.max_lnlkl for chain in self._chains])
        chain_max_lnlkl  = np.argmax([chain.max_lnlkl for chain in self._chains])
        self._best_fit = self._chains[chain_max_lnlkl].best_fit

        for chain in self._chains:
            chain.remove_burnin(self._max_lnlkl)
        
        #######################
        # print some results

        max_markov_index = len(str(int(np.max([chain.markov_index for chain in self._chains]))))
        max_burnin_index = len(str(int(np.max([chain.burnin_index for chain in self._chains]))))
    
        for ic, chain in enumerate(self._chains):
            print(f'Chain {ic+1:<3} : Removed {chain.markov_index:<{max_markov_index}} non-markovian points, ' \
                  + f'{chain.burnin_index:<{max_burnin_index}} points of burn-in, keep ' + str(chain._values.shape[1]) \
                  + f' steps | (max_lnlkl = {chain.max_lnlkl:.2f}, acceptance_rate = {chain.acceptance_rate:.2f})' )
            
           
        #######################
        # compute some stats

        # define some global quantities (total number of steps and overall mean of the parameters)
        self._total_steps = np.sum(np.array([chain.n_steps for chain in self._chains]), dtype=int)

        self._total_mean = np.zeros(self.n_params)
        for chain in self._chains:
            if chain.length > 0:
                chain.compute_stats()

            self._total_mean  = self._total_mean + chain.n_steps * chain.mean_value
        self._total_mean = self._total_mean / self._total_steps

        self.load_scaling_factor()

           


    def load_chains(self) -> None:
        
        # look for chains in the folder
        chains_path = self.path +  '_*.txt'
        self._chains_name = np.array(glob.glob(chains_path))
        ids = np.array([int(name.split('.')[-2].split('_')[-1]) for name in self._chains_name], dtype=int)
        self._chains_name = self._chains_name[np.argsort(ids)]

        # raise an error if no chain is found
        if len(self._chains_name) == 0:
            raise ValueError("No chain found at " + chains_path)

        # redefine the chain name list from the given ids
        if self.ids is not None:
            self._chains_name = self._chains_name[self.ids]

        # define the number of chains
        self.n_chains = len(self._chains_name)
        
        # prepare an array for the non markovian chain
        self._markov = np.zeros(self.n_chains, dtype=int)

        # read all chains
        self._chains = [MPChain(filename) for filename in self._chains_name]

        self.n_params = self._chains[0].values().shape[0]
        

    def load_paramnames(self):
        
        self._names = np.empty(0)
        self._latex_names = np.empty(0)
       
        with open(self.path + '.paramnames', 'r') as f:
            for line in f:
                ls = line.split('\t')
                self._names = np.append(self._names, ls[0][:-1])
                self._latex_names = np.append(self._latex_names, r'${}$'.format(ls[1][1:-2]))


    def load_scaling_factor(self) -> None:

        self._scaling_factor = {}
        
        with open(os.path.join(*[*(self._path.split('/')[:-1]), 'log.param']), 'r') as file:
            for line in file:
                if (not line.strip().startswith('#')) and ('data.parameters' in line):
                    name  = line.split('[')[1].strip("[] ='")
                    value = float(line.split('[')[2].strip("[] ='\n").split(',')[-2].strip("[] ='"))
                    self._scaling_factor[name] = value  


    def flat(self, discard: np.ndarray | None = None, thin: None | int = None, **kwargs) -> np.ndarray:
        """
        Flatten samples of all chains.

        Parameters
        ----------
        discard : np.ndarray | None, optional
            Number of points to discard at begining of the chain, by default None (0).
        thin : None | int, optional
            Reduce the size of the sample by taking one point every `thin`, by default None.
            If Nont compute the reduced size such that the total length of the sample is 10000.

        Returns
        -------
        np.ndarray
            Sample in a 2 dimensional array of shape (# of parameters, # of points)
        """
        
        if isinstance(discard, int):
            discard = np.full(self.n_chains, discard)

        if discard is None:
            discard = np.zeros(self.n_chains, dtype=int)

        if thin is None:
            m_total_length = 0
            for ichain, chain in enumerate(self.chains):
                m_total_length = m_total_length + chain.values(discard = discard[ichain], thin=1).shape[1]

            if m_total_length > 1e+4:
                thin = int(m_total_length/10000)
            else:
                thin = 1

        res = np.empty((self.n_params, 0))
        for ichain, chain in enumerate(self.chains):
            res = np.concatenate((res, chain.values(discard = discard[ichain], thin=thin)), axis=-1)
        
        return res
    
    
    def convergence(self) -> np.ndarray:
        """
        Gelman-Rubin criterion weighted by the length of the chain as implemented
        in MontePython.

        Returns
        -------
        np.ndarray
            R-1 for all parameters
        """

        within  = np.zeros(self.n_params)
        between = np.zeros(self.n_params)
        
        for chain in self.chains :
            within  = within  + chain.n_steps * chain.var_value 
            between = between + chain.n_steps * (chain.mean_value - self.total_mean)**2

        within  = within / self.total_steps
        between = between / (self.total_steps-1)

        return between/within
    


    def covmat(self, 
               discard: np.ndarray | None = None, 
               params_in_cov : list[str] | None = None) -> np.ndarray:
        """
        Covariance matrix.

        Parameters
        ----------
        discard : np.ndarray | None, optional
            Number of points to discard at begining of the chain, by default None (0).
        data_to_cov : list[str] | None, optional
            List of parameters to put in the covariance matrix (in the order of that list).
            If None consider all parameters available.

        Returns
        -------
        np.ndarray
            Covariance matric (n, n) array
        """
        
        points = prepare_data_plot(self, params_in_cov, discard, thin=1)        
        return np.cov(points)




    def print_best_fit(self, discard: np.ndarray | None = None, **kwargs):
        
        samples_flat = self.flat(discard=discard, thin=1, **kwargs)
        med  = np.median(samples_flat, axis=1)
        mean = np.mean(samples_flat, axis=1)

        nc = np.zeros(len(self.param_names), dtype=int)
        for ip, param in enumerate(self.param_names):
            nc[ip] = len(param)

        max_nc = np.max(nc)

        for ip, param in enumerate(self.param_names):
            fill = " " * (max_nc - nc[ip])
            print(param +  fill + ' : \t' + str(self.best_fit[ip]) + " | " + str(med[ip]) + " | " + str(mean[ip]))

     


    @property
    def chains(self):
        return self._chains
        
    @property
    def max_lnlkl(self):
        return self._max_lnlkl  
    
    @property
    def best_fit(self):
        return self._best_fit
    
    @property
    def param_names(self):
        return self._names
    
    @property
    def param_names_latex(self):
        return self._latex_names
    
    @property
    def scaling_factor(self):
        return self._scaling_factor
    
    @property
    def total_steps(self):
        return self._total_steps
    
    @property
    def total_mean(self):
        return self._total_mean
    




def prepare_data_plot(samples: Samples, data_to_plot, discard = 0, thin = 1, **kwargs):

    data_to_plot = to_CLASS_names(data_to_plot)

    # give the transformation rules between the parameter if there is to be one
    def transform_data(data, name, param_names):

        if name == 'sum_mnu':
            if 'm_nu1' in param_names:
                index_mnu = data_to_plot.index('sum_mnu')
                data[index_mnu] = np.sum(neutrino_masses(data[index_mnu], hierarchy='NORMAL'), axis = 0)

                # change also the param_names
                param_names[index_mnu] = name
            else:
                raise ValueError('Impossible to obtain ' + name + ' from the input data sample')
            
        return param_names


    data = samples.flat(discard=discard, thin=thin, **kwargs)
    param_names = copy(samples.param_names)

    # rescaling the data according to the scaling factor
    for iname, name in enumerate(samples.param_names):
        if (samples.scaling_factor[name] != 1) and (name != '1/m_wdm' and name != 'm_wdm'): 
            # note that we keep the warm dark matter transformed
            # just means that we need to be carefull with the units
            data[iname] = samples.scaling_factor[name] * data[iname]


    # first transform the data if necessary
    if 'sum_mnu' in data_to_plot:
        param_names = transform_data(data, 'sum_mnu', param_names)


    # reduce the data to the indices to plot
    indices_to_plot = [np.where(param_names == param)[0][0] for param in data_to_plot if param in param_names]
    data = data[indices_to_plot]

    # remove outliers for tau_reio
    if 'tau_reio' in data_to_plot:
        index_tau_reio = data_to_plot.index('tau_reio')

        mask = data[index_tau_reio] < 0.1
        n_outliers = np.count_nonzero(~mask)
        
        if n_outliers > 0 : 
            warnings.warn(f'Removing {n_outliers} outlier points for tau_reio: ' + str(data[index_tau_reio, ~mask]))

        data = data[:, mask]
        

    return data




#################################################
## PLOTTING TOOLS

MATPLOTLIB_IMPORTED = False

try:

    import matplotlib.pyplot as plt
    import matplotlib.colors as mpc

    from dataclasses import dataclass

    MATPLOTLIB_IMPORTED = True    

    class AxesGrid:

        def __init__(self, 
                    n: int, 
                    *, 
                    scale: float = 2.0,
                    wspace: float = 0.05, 
                    hspace: float  = 0.05, 
                    labels: list[str] | None = None,
                    names: list[str] | None = None,
                    **kwargs):
            
            # close all pre-existing figures and disable interactive mode
            plt.close('all')
            
            # define a figure object and the grid spec
            self._fig = plt.figure(figsize=(scale*n, scale*n), constrained_layout=False)
            self._spec = self._fig.add_gridspec(ncols=n, nrows=n, wspace=wspace, hspace=hspace)

            # initialise the length
            self._size: int = n

            # initialise an empty list
            self._axs: list[plt.Axes] = [None for i in range(int(n*(n+1)/2))]

            # define the edges of the axes
            self._edges: np.ndarray = np.full((n, 2), fill_value=None)

            # define the labels of the axes
            self._labels: list[str] = [r'${{{}}}_{}$'.format(r'\theta', i) for i in range(self.size)] if labels is None else labels

            # define the name of the parameter attached to each axis
            self._names: list[str] = self._labels if names is None else names

            # define the titles of the axes (showing mean and quantiles if asked)
            self._titles: list[list[str]] = [[] for i in range(self.size)]

            # define the text objects holding the title
            self._titles_text : list[list[plt.Text | None]] = [[] for i in range(self.size)]  

            # define the axes on the grid
            for i in range(n):
                for j in range(i+1):

                    k = self.index_1D(i, j)
                    self._axs[k] = self._fig.add_subplot(self._spec[i, j])

                    if i < n-1:
                        self._axs[k].xaxis.set_ticklabels([])
                    if j > 0: 
                        self._axs[k].yaxis.set_ticklabels([])

                    self._axs[0].yaxis.set_ticklabels([])


            # define default font and rotation of ticks
            self._fontsize: float = self._axs[0].xaxis.label.get_size()
            self._ticks_rotation: float = 50
            self._titles_color: list[str] = [[] for i in range(self.size)]
            self._default_color: str = 'blue'

            self.update_labels(**kwargs)



        def get(self, i: int | str, j: int | str):

            i, j = (self.index_from_name(k) if isinstance(k, str) else k for k in [i, j])
            return self._axs[self.index_1D(i, j)]
        

        def index_1D(self, i: int, j: int):
            
            if j > i:
                raise ValueError("j must be less or equal than i") 

            return int(i*(i+1)/2 + j)
        
        def indices_2D(self, k):
        
            i = np.arange(0, k+2, dtype=int)
            ind_triangle = i*(i+1)/2
            row    = np.searchsorted(ind_triangle, k, side='right')-1
            column = int(k - ind_triangle[row])

            return row, column


        # show the plot
        def show(self):
            self._fig.show()


        # change one label name
        def set_label(self, i:int, name:str):
            self._labels[i] = name

            if i > 0:
                self.get(i, 0).set_ylabel(name)
            
            self.get(self.size-1, i).set_xlabel(name)


        # update all the label properties
        def update_labels(self, **kwargs):
            
            self._labels   = kwargs.get('labels', self._labels)
            self._fontsize = kwargs.get('fontsize', self._fontsize)
            self._ticks_rotation = kwargs.get('ticks_rotation', self._ticks_rotation)

            for i in range(1, self.size):
                k = self.index_1D(i, 0)
                self._axs[k].set_ylabel(self._labels[i], fontsize=self._fontsize)
                self._axs[k].tick_params(axis='y', labelsize=self._fontsize-2)
                for tick in self._axs[k].get_yticklabels():
                        tick.set_rotation(self._ticks_rotation)

            for j in range(self.size):
                k = self.index_1D(self.size-1, j)
                self._axs[k].set_xlabel(self._labels[j], fontsize=self._fontsize)
                self._axs[k].tick_params(axis='x', labelsize=self._fontsize-2)
                for tick in self._axs[k].get_xticklabels():
                    tick.set_rotation(self._ticks_rotation)

        #update the edges
        def update_edges(self, axis: int | str, min: float, max: float):
        
            j = self.index_from_name(axis) if isinstance(axis, str) else axis
            self._edges[j, :] = np.array([min, max]) 

            for i in range(j, self.size):
                self.get(i, j).set_xlim([self._edges[j, 0], self._edges[j, -1]])

            for k in range(0, j):
                self.get(j, k).set_ylim([self._edges[j, 0], self._edges[j, -1]])
        

        # update the titles properties
        def add_title(self, axis: int, new_titles: str, color: str | None = None):
            
            self._titles_color[axis].append(color if color is not None else self._default_color)
            self._titles[axis].append(new_titles)
            self._titles_text[axis].append(None)

        def update_titles(self, height = 1.05, spacing = 1.9, fontsize = None):

            if fontsize is None:
                fontsize = self._fontsize

            for j in range(self.size):
                k = self.index_1D(j, j)
                for it, title in enumerate(self._titles[j]):
                    total_height = height if it == 0 else height+it*spacing*1e-2*fontsize
                    if self._titles_text[j][it] is None: 
                        self._titles_text[j][it] = self._axs[k].text(0.5, total_height, title, fontsize=fontsize, color = self._titles_color[j][it], ha='center', transform=self._axs[k].transAxes)
                    else:
                        self._titles_text[j][it].set_position((0.5, total_height))
                        self._titles_text[j][it].set_text(title)
                        self._titles_text[j][it].set_fontsize(fontsize)

        def index_from_name(self, name: str | list[str]):
            
            if isinstance(name, str):
                return self.names.index(name)
            else:
                return [self.names.index(na) for na in name]

        # List of properties

        @property
        def fig(self):
            return self._fig
        
        @property
        def spec(self):
            return self._spec
        
        @property
        def size(self):
            return self._size
        
        @property
        def edges(self):
            return self._edges
        
        @edges.setter
        def edges(self, value: np.ndarray):
            self._edges = value

        @property
        def labels(self):
            return self._labels
        
        @labels.setter
        def labels(self, value: list[str]):
            self._labels = value
            self.update_labels()

        @property
        def names(self):
            return self._names


    
    @dataclass
    class ProcessedData:

        hists_1D  : np.ndarray | None = None
        hists_2D  : np.ndarray | None = None
        edges     : np.ndarray | None = None
        centers   : np.ndarray | None = None
        levels    : np.ndarray | None = None
        q         : np.ndarray | None = None
        mean      : np.ndarray | None = None
        median    : np.ndarray | None = None
        bestfit   : np.ndarray | None = None
        quantiles : np.ndarray | None = None
        samples   : np.ndarray | None = None
        size      : int | None = None



    def compute_quantiles(sample: np.ndarray, q: float, bins: int | np.ndarray = 30) -> tuple[np.ndarray, np.ndarray]:
        """
        Give the q-th quantile of the input sample.

        Parameters
        ----------
        sample : np.ndarray
            1D array of data points.
        q : float
            Quantile value.
        bins : int | np.ndarray, optional
            Binning of the histogram that is used 
            for a first approximation of the quantile 
            edges, by default 30.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
        min,max bounds
        """

        def split_in_groups(lst):
        
            res = [[int(lst[0])]]
            i = 0

            for il, element in enumerate(lst):
                element = int(element)
                if il > 0:
                    if element - lst[il-1] == 1:
                        res[i].append(element)
                    else:
                        res.append([element])
                        i = i+1

            return res
            
        # define the histogram from the sample
        hist, edges = np.histogram(sample, bins=bins, density=True)
        hist_count, _ = np.histogram(sample, bins=bins)

        # normalise the histogram to 1
        hist = hist/np.max(hist)

        # define the total sum of the histogram
        s_hist = np.sum(hist)

        # order the histogram values
        ordered_hist  = np.sort(hist)[::-1]
        order_of_hist = np.argsort(hist)[::-1]

        # compute the cumulative sum and get all indices that
        # contribute to the quantile
        sum_hist = np.cumsum(ordered_hist)/s_hist 
        indices = np.arange(0, len(hist), dtype=int)[sum_hist < q]

        # the first bin is already high enough we return its boundaries
        if len(indices) == 0:

            e_min = np.empty((1, 2))
            e_max = np.empty((1, 2))

            index_one = order_of_hist[0]

            e_min[0, 0] = edges[:-1][index_one]
            e_max[0, 0] = edges[:-1][index_one]
            e_min[0, 1] = edges[1:][index_one]
            e_max[0, 1] = edges[1:][index_one]

            ilg = 0
            total_count_small = 0

            x_min = np.zeros(1)
            x_max = np.zeros(1)

            m = np.median(sample)

        else:

            # add an extra index to be sure we capture all the good bins
            final_index = int(indices[-1]+1)

            # if we have same size hist values where sum_hist = q then
            # we need to consider all possible bins that can contribute
            # we put all these indices into a table
            multiple_indices = [final_index]

            while final_index < len(ordered_hist)-1 and (ordered_hist[final_index] == ordered_hist[final_index+1]):
                multiple_indices.append(final_index+1)
                final_index = final_index+1

            # chack for each last index what is the one that minimize the number of groups 
            len_groups = []

            for index in multiple_indices:
                bin_indices = np.sort(order_of_hist[np.concatenate((indices, np.array([index])))]) # all bins contributing
                len_groups.append(len(split_in_groups(bin_indices)))

            index_min_groups = np.argmin(len_groups)
            bin_indices    = np.sort(order_of_hist[np.concatenate((indices, np.array([multiple_indices[index_min_groups]])))])
            groups = split_in_groups(bin_indices)


            # here size is different from length,
            # size refers to the 'physical' size
            size_groups   = []
            count_groups  = []
            sample_groups = []

            # get what is inside the groups
            # approximate values of the other groups
            e_min = np.empty((len(groups), 2))
            e_max = np.empty((len(groups), 2))

            x_min = np.empty(len(groups))
            x_max = np.empty(len(groups))

            for ig, group in enumerate(groups):
                index_min, index_max = np.min(group), np.max(group)
                
                e_max[ig, 0], e_max[ig, 1] = edges[:-1][index_max], edges[1:][index_max]
                e_min[ig, 0], e_min[ig, 1] = edges[:-1][index_min], edges[1:][index_min]

                size_groups.append(float(e_max[ig, 1]-e_min[ig, 0]))
                count_groups.append(int(np.sum(hist_count[group])))

                mask = (sample > e_min[ig, 0]) & (sample < e_max[ig, 1])
                sample_groups.append(sample[mask])

                x_min[ig] = e_min[ig, 0]
                x_max[ig] = e_max[ig, 1]

                # adjust the exact value for the largest group
                # and keep only the bins of the smaller groups


            ilg = np.argmax(size_groups)
            total_count_small = np.sum(count_groups) - count_groups[ilg]
            m = np.median(sample_groups[ilg]) if len(groups) > 0 else np.median(sample)


        # define the grid for the finner search
        x_m = np.linspace(e_min[ilg, 0], e_min[ilg, 1], 30)
        x_p = np.linspace(e_max[ilg, 0], e_max[ilg, 1], 30)


        # for each value of x_m find the value of x_p such that quantile is q
        mask  = (sample >= x_m[:, None, None]) & (sample <=  x_p[None, :, None])
        count = (np.count_nonzero(mask, axis=-1) + total_count_small)/len(sample)
        m_x_p = x_p[np.argmin(np.abs(count - q), axis=1)]
        
        if count[0, -2] < q and count[1, -1] < q and count[1, -2] < q:
            x_min[ilg], x_max[ilg] = x_m[0], x_p[-1]
            return x_min, x_max

        # remove the values where x_p is just equal to the max
        # at the very edege of the grid 
        mask_xp = np.argmin(m_x_p != x_p[-1])

        # if we have removed all points then problem
        # we actually do not mask, it is just that we
        # cannot really find the q contour with our
        # resolution. Return the max value
        if mask_xp == 0:
            x_min[ilg] = x_m[-1]
            x_max[ilg] = x_p[-1]
        else:
            xm  = x_m[:mask_xp+1]
            xp  = m_x_p[:mask_xp+1]

            index = np.argmin(np.sqrt((2*m - xm - xp)**2))
            x_min[ilg], x_max[ilg] = xm[index], xp[index]

        return x_min, x_max





    def generate_contours(samples: np.ndarray, 
                        bins: int = 20, 
                        q = [0.68, 0.95],
                        smooth_2D: bool = False,
                        smooth_1D: bool = False,
                        sigma_smooth: float = 1.5,) -> ProcessedData:

        data = ProcessedData()

        n = samples.shape[0]

        hists_1D   = np.empty((n, bins))
        hists_2D   = np.empty((n, n, bins, bins))    # 2D array of 2D array
        levels     = np.empty((n, n, len(q)+1)) # 2D array of 1D array

        quantiles  = np.full((len(q), n, 2), fill_value=np.nan)

        q = np.array(q)
        if np.any(q != sorted(q)):
            raise ValueError('levels should be given in ascending order')
            # check that the input levels are in ascending order

        # edges and centers of the histograms
        edges   = np.vstack([np.linspace(np.min(s), np.max(s), bins+1) for s in samples])
        centers = (edges[:, :-1] + edges[:, 1:]) / 2

        mean   = np.mean(samples, axis=-1)
        median = np.median(samples, axis=-1)


        # loop over all places with 1D histograms
        for i in range(n):
            
            hists_1D[i, :], _ = np.histogram(samples[i, :], bins=edges[i, :], density=True)
            hists_1D[i] = gaussian_filter1d(hists_1D[i], sigma=sigma_smooth) if smooth_1D is True else hists_1D[i]
            
            hists_1D[i] = hists_1D[i] / np.max(hists_1D[i])

            # evaluate the quantiles
            for il, q_val in enumerate(q):
                try:
                    x_min, x_max = compute_quantiles(samples[i, :], q_val, bins=bins)
                except:
                    warnings.warn(f"Impossible to compute quantiles for entry {i}")

                if len(x_min) > 1 or len(x_max) > 1:
                    warnings.warn(f"Quantiles not given for what appears to be a multimodal distribution {i}")
                else:
                    try:
                        quantiles[il, i, 0], quantiles[il, i, 1] = x_min[0], x_max[0]
                    except Exception as e:
                        print(quantiles[il, i, 0], quantiles[il, i, 1], x_min, x_max)
                        raise e

        
        # loop over all places with 2D histograms
        for i in range(1, n):
            for j in range(i):

                hists_2D[i, j, :, :], _, _ = np.histogram2d(samples[j, :], samples[i, :], bins=[edges[j, :], edges[i, :]], density=True)
                hists_2D[i, j] = gaussian_filter(hists_2D[i, j], sigma=sigma_smooth) if smooth_2D is True else hists_2D[i, j]
                
                # Flatten the histogram to sort for cumulative density
                sorted_hist = np.sort(hists_2D[i, j].ravel())[::-1]  # Sort in descending order

                # Compute cumulative density
                cumulative = np.cumsum(sorted_hist) / np.sum(sorted_hist)

                # Find threshold levels for the desired confidence intervals
                levels[i, j, :-1] = sorted_hist[np.searchsorted(cumulative, q[::-1])]
                levels[i, j, -1]  = 1.1*np.max(sorted_hist)

        data.hists_2D  = hists_2D
        data.hists_1D  = hists_1D
        data.samples   = samples
        data.edges     = edges
        data.centers   = centers
        data.levels    = levels
        data.mean      = mean
        data.median    = median
        data.quantiles = quantiles
        data.size      = hists_1D.shape[0]
        data.q         = q
            
        return data



    def plot_data(grid: AxesGrid, 
                data : ProcessedData, 
                show_hist: bool    = False, 
                show_surface: bool = True, 
                show_contour: bool = False,
                show_quantiles: list[bool] = [False, False],
                show_mean: bool = False,
                show_title: bool = True,
                show_points: bool = False,
                redefine_edges: bool = True,
                q_in_title: int = 0.68,
                colors: list[str]  = 'orange',
                axes : list[int] | np.ndarray | None = None,
                exclude_quantiles : int | str | list[int] | list[str] = [],
                exclude_mean : int | str | list[int] | list[str] = [],
                exclude_title : int | str | list[int] | list[str] = [],
                alphas: list[float] = 1.0):
        

        alphas, colors, exclude_quantiles, exclude_mean, exclude_title = ([array] if isinstance(array, float) else array for array in [alphas, colors, exclude_quantiles, exclude_mean, exclude_title])
        exclude_quantiles, exclude_mean, exclude_title = (grid.index_from_name(exclude) if (len(exclude) > 0 and isinstance(exclude[0], str)) else exclude for exclude in [exclude_quantiles, exclude_mean, exclude_title])

        if axes is None:
            axes = np.arange(0, data.size)


        # first define the colors we will need to use
        contour_colors = [mpc.to_rgba(color, alphas[ic]) if isinstance(color, str) else color for ic, color in enumerate(colors)]

        # if we provide one color and we ask for more levels then
        # we define new colors automatically colors
        if len(contour_colors) == 1:
            
            pastelness = np.array([0.7]) if len(data.levels[0, 0]) == 3 else np.linspace(0.5, 0.8, len(data.levels[0, 0])-2)
            pastelness = pastelness[:, None] * np.ones((1, 4))
            pastelness[:, -1] = 0

            # add custom pastel colors to the stack of colors
            contour_colors = np.vstack(((1.0 - pastelness) * np.array(contour_colors) + pastelness, contour_colors))
            
        # plot the contours and points
        for i in range(1, data.size):
            for j in range(i):

                hist = data.hists_2D[i, j]

                if show_points is True:

                    # first thin the samples so that we plot only 5000 points
                    n = data.samples.shape[-1]
                    r = np.max([1, int(n/5000)])

                    grid.get(axes[i], axes[j]).scatter(data.samples[j, ::r], data.samples[i, ::r], marker='o', edgecolors='none', color = contour_colors[-1], s=2, alpha=0.5)


                if show_hist is True:
                    extent = [data.edges[j, 0], data.edges[j, -1], data.edges[i, 0], data.edges[i, -1]]
                    grid.get(axes[i], axes[j]).imshow(hist.T, origin='lower', extent=extent, cmap='Greys', aspect='auto')

                
                if show_surface is True:
                    try:
                        grid.get(axes[i], axes[j]).contourf(*np.meshgrid(data.centers[j], data.centers[i]), hist.T, levels=data.levels[i, j], colors=contour_colors)
                    except ValueError as e:
                        print("Error for axis : ", i, j)
                        raise e

                if show_contour is True:
                    grid.get(axes[i], axes[j]).contour(*np.meshgrid(data.centers[j], data.centers[i]), hist.T, levels=data.levels[i, j], colors=contour_colors)


        
        # fill in the 1D histograms
        for i in range(0, data.size):
            
            grid.get(axes[i], axes[i]).stairs(data.hists_1D[i], edges=data.edges[i, :], color = colors[0])

            if (show_mean is True) and (i not in exclude_mean):
                grid.get(axes[i], axes[i]).axvline(data.mean[i], color=contour_colors[0], linewidth=0.5, linestyle='--')


            for iq, quantile in enumerate(data.quantiles[::-1]):

                if (show_quantiles[iq] is True) and (i not in exclude_quantiles):

                    # fill the histogram in terms of the first quantile
                    mask_edges = (data.edges[i, :] <= quantile[i, 1]) & (data.edges[i, :] >= quantile[i, 0])

                    # corresponding mask for the histogram values
                    mask_hist  = (mask_edges[:-1]*mask_edges[1:] == 1)

                    if len(data.hists_1D[i, mask_hist]) > 0:
                        # plot the quantiles with a shaded histogram
                        grid.get(axes[i], axes[i]).stairs(data.hists_1D[i, mask_hist], edges=data.edges[i, mask_edges], color = contour_colors[iq], fill=True, alpha=0.5)
                        
            title_color = contour_colors[1]
            title_color[-1] = 1.0

            # get the number of quantiles
            if q_in_title in data.q:
                jq = np.where(q_in_title == data.q)[0][0]
            else:
                raise ValueError('quantile in title should be a q value given in generate_contour')
            
            if (show_title is True) and (i not in exclude_title):
                if not np.isnan(data.quantiles[jq, i, 0]) and not np.isnan(data.quantiles[jq, i, 1]) :
                    grid.add_title(axes[i], r'${:.3g}$'.format(data.median[i], color=colors[0]) + '$^{{ +{:.2g} }}_{{ -{:.2g} }}$'.format(data.quantiles[jq, i, 1] - data.median[i], data.median[i] - data.quantiles[jq, i, 0] ), color=title_color)  
                else:
                    grid.add_title(axes[i], r'${:.3g}$'.format(data.median[i], color=colors[0]), color=title_color) 
        
        
        grid.update_titles()


        # (re)define the grid edges
        if redefine_edges is True:
            
            new_boundaries = data.edges[:, [0, -1]]

            old_boundaries = grid.edges[axes]
            old_boundaries[old_boundaries[:, 0] == None] = data.edges[old_boundaries[:, 0] == None, :][:, [0, -1]]

            new_min = np.minimum(new_boundaries[:, 0], old_boundaries[:, 0]) 
            new_max = np.maximum(new_boundaries[:, 1], old_boundaries[:, 1])

            new_edges = np.vstack((new_min, new_max)).T

            grid.edges[axes, :] = new_edges 
            
            for j in range(0, data.size):
            
                for i in range(j, data.size):
                    grid.get(axes[i], axes[j]).set_xlim([grid.edges[axes[j], 0], grid.edges[axes[j], -1]])
                    
                    if i > j:
                        grid.get(axes[i], axes[j]).set_ylim([grid.edges[axes[i], 0], grid.edges[axes[i], -1]])




    def plot_2D_marginal(ax: plt.Axes,
                        data : ProcessedData, 
                        i: int,
                        j: int,
                        show_hist: bool    = False, 
                        show_surface: bool = True, 
                        show_contour: bool = False,
                        show_points: bool  = False,
                        colors: list[str]  = 'orange', 
                        alphas: list[float] = 1.0):
        


        if i < j:
            i, j = j, i

        hist = data.hists_2D[i, j].T
    

        alphas, colors = ([array] if isinstance(array, float) else array for array in [alphas, colors])

        # first define the colors we will need to use
        contour_colors = [mpc.to_rgba(color, alphas[ic]) if isinstance(color, str) else color for ic, color in enumerate(colors)]

        # if we provide one color and we ask for more levels then
        # we define new colors automatically colors
        if len(contour_colors) == 1:
            
            pastelness = np.array([0.7]) if len(data.levels[0, 0]) == 3 else np.linspace(0.5, 0.8, len(data.levels[0, 0])-2)
            pastelness = pastelness[:, None] * np.ones((1, 4))
            pastelness[:, -1] = 0

            # add custom pastel colors to the stack of colors
            contour_colors = np.vstack(((1.0 - pastelness) * np.array(contour_colors) + pastelness, contour_colors))
        
        if show_points is True:

            # first thin the samples so that we plot only 5000 points
            n = data.samples.shape[-1]
            r = np.max([1, int(n/10000)])

            ax.scatter(data.samples[j, ::r], data.samples[i, ::r], marker='o', edgecolors='none', color = contour_colors[-1], s=2, alpha=0.5)

        if show_hist is True:
            extent = [data.edges[j, 0], data.edges[j, -1], data.edges[i, 0], data.edges[i, -1]]
            ax.imshow(hist, origin='lower', extent=extent, cmap='Greys', aspect='auto')

                
        if show_surface is True:
            try:
                ax.contourf(*np.meshgrid(data.centers[j], data.centers[i]), hist, levels=data.levels[i, j], colors=contour_colors)
            except ValueError as e:
                print("Error for axis : ", i, j)
                raise e

        if show_contour is True:
            ax.contour(*np.meshgrid(data.centers[j], data.centers[i]), hist, levels=data.levels[i, j], colors=contour_colors)

        



    def prepare_data_Xe(samples: Samples, 
                    data_to_plot: list[str] | np.ndarray,  
                    discard: int = 0, 
                    thin: int = 100,
                    *,
                    classifier: Classifier | None = None,
                    regressor: Regressor | None = None,):
        
        data_for_Xe = []

        data_to_plot = to_CLASS_names(data_to_plot)

        # get the ordered list of parameters
        if regressor is None:
            regressor  = Regressor.load()

        if classifier is None:
            classifier = Classifier.load()

        parameters = regressor.metadata.parameters_name

        for param in data_to_plot:
            
            if param == 'sum_mnu':
                param = 'm_nu1'

            if (param in MP_KEY_CORRESPONDANCE) and (MP_KEY_CORRESPONDANCE[param] in parameters):
                data_for_Xe.append(param)

        
        labels_correspondance = {value : key for key, value in MP_KEY_CORRESPONDANCE.items()}
        
        # get the data sample 
        data_sample = prepare_data_plot(samples, data_for_Xe, discard=discard, thin=thin, regressor = regressor, classifier = classifier)
        data = np.empty((len(parameters), data_sample.shape[-1])) 

        # find the ordering in which data_sample is set in prepare_data_plot
        indices_to_plot = [np.where(samples.param_names == param)[0][0] for param in data_for_Xe if param in samples.param_names]

        for ip, param in enumerate(parameters): 
            
            # if we ran the MCMC over that parameter
            if labels_correspondance[param] in samples.param_names[indices_to_plot]:
                index = list(samples.param_names[indices_to_plot]).index(labels_correspondance[param])
                data[ip] = data_sample[index, :]
            else:
                data[ip, :] = DEFAULT_VALUES[param]

        return data
        


    def get_Xe_stats(samples: Samples, 
                    data_to_plot: list[str] | np.ndarray,  
                    nbins: int = 100, 
                    discard: int = 0, 
                    thin: int = 100,
                    *,
                    classifier: Classifier | None = None,
                    regressor: Regressor | None = None,
                    smooth: bool = False,
                    sigma_smooth: float = 1.5,
                    **kwargs):

        data = prepare_data_Xe(samples, data_to_plot, discard, thin, classifier=classifier, regressor=regressor).T

        for kw, val in kwargs.items():
            if kw in regressor.parameters_name:
                ikw = list(regressor.parameters_name).index(kw)
                data[:, ikw] = val
            else:
                raise ValueError("Need to pass a kwarg that is in the parameter list of the classifier / regressor")

        Xe = predict_Xe_numpy(theta=data, classifier=classifier, regressor=regressor)

        # here remove some outliers that should not have 
        # passed the likelihood condition
        if np.count_nonzero(Xe[:, -1]==-1)/len(Xe) > 0.01:
            warnings.warn("More than 1 percent of outliers with late reionization")

        Xe = Xe[Xe[:, -1] > 0]
        z = regressor.metadata.z
        
        mean, med = np.mean(Xe, axis=0), np.median(Xe, axis=0)
        min, max  = np.min(Xe), np.max(Xe)

        log_bins = np.linspace(np.log10(min), np.log10(max), nbins)
        log_hist = np.zeros((len(z), len(log_bins)-1))

        lin_bins = np.linspace(min, max, nbins)
        lin_hist = np.zeros((len(z), len(lin_bins)-1))

        # make an histogram for each value of z
        for iz, x in enumerate(Xe.T):

            log_hist[iz], _ = np.histogram(np.log10(x), bins=log_bins, density = True)
            log_hist[iz] = gaussian_filter1d(log_hist[iz], sigma=sigma_smooth) if smooth is True else log_hist[iz]
            log_hist[iz] = log_hist[iz]/np.max(log_hist[iz])
            
            lin_hist[iz], _ = np.histogram(x, bins=lin_bins, density = True)
            lin_hist[iz] = gaussian_filter1d(lin_hist[iz], sigma=sigma_smooth) if smooth is True else lin_hist[iz]
            lin_hist[iz] = lin_hist[iz]/np.max(lin_hist[iz])
        
        log_bins = 10**((log_bins[:-1] + log_bins[1:])/2.0)
        lin_bins = (lin_bins[:-1] + lin_bins[1:])/2.0
        
        return z, mean, med, log_hist, log_bins, lin_hist, lin_bins




    def get_Xe_tanh_stats(samples: Samples, 
                            nbins: int = 100, 
                            discard: int = 0, 
                            thin : int = 100, 
                            x_inf: float = 2e-4, 
                            *,
                            smooth: bool = False,
                            sigma_smooth: float = 1.5,
                            **kwargs):

        def Xe_class(z, z_reio = 8.0):
            return 0.5 * ( 1+np.tanh(  (1+z_reio)/(1.5*0.5)*(1-((1+z)/(1+z_reio))**(1.5))   ) )   

        index = list(samples.param_names).index('z_reio')
        z_reio = samples.flat(discard=discard, thin=thin, **kwargs)[index, :]

        z = np.linspace(0, 35, 100)
        Xe = Xe_class(z, z_reio[:, None]) + x_inf

        mean, med = np.mean(Xe, axis=0), np.median(Xe, axis=0)
        min, max  = np.min(Xe), np.max(Xe)

        log_bins = np.linspace(np.log10(min), np.log10(max), nbins)
        log_hist = np.zeros((len(z), len(log_bins)-1))

        lin_bins = np.linspace(min, max, nbins)
        lin_hist = np.zeros((len(z), len(lin_bins)-1))

        # make an histogram for each value of z
        for iz, x in enumerate(Xe.T):

            log_hist[iz], _ = np.histogram(np.log10(x), bins=log_bins, density = True)
            log_hist[iz] = gaussian_filter1d(log_hist[iz], sigma=sigma_smooth) if smooth is True else log_hist[iz]
            log_hist[iz] = log_hist[iz]/np.max(log_hist[iz])
            
            lin_hist[iz], _ = np.histogram(x, bins=lin_bins, density = True)
            lin_hist[iz] = gaussian_filter1d(lin_hist[iz], sigma=sigma_smooth) if smooth is True else lin_hist[iz]
            lin_hist[iz] = lin_hist[iz]/np.max(lin_hist[iz])
        
        log_bins = 10**((log_bins[:-1] + log_bins[1:])/2.0)
        lin_bins = (lin_bins[:-1] + lin_bins[1:])/2.0
        
        return z, mean, med, log_hist, log_bins, lin_hist, lin_bins
    
except:
    pass