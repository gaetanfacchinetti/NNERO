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

from abc import ABC, abstractmethod

from .data       import MP_KEY_CORRESPONDANCE
from .predictor  import DEFAULT_VALUES
from .regressor  import Regressor
from .classifier import Classifier
from .predictor  import predict_xHII_numpy, predict_tau_numpy

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



class EMCEESamples(Samples):

    def __init__(self, path : str, ids: list[int] | np.ndarray | None = None, add_tau: bool = False) -> None:
        
        if EMCEE_IMPORTED is False:
            print("Cannot import emcee, therefore cannot work with emcee chains")
            return None
        
        super().__init__(path, ids)


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
        
        with open(self.path.split('.')[0] + '.npz', 'rb') as file:

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
            tau = self.compute_tau(flat_chain, classifier, regressor)

            flat_chain = np.vstack((tau[None, :], flat_chain))
            

        return flat_chain



    def compute_tau(self, 
                    flat_chain: np.ndarray, 
                    classifier: Classifier | None = None,
                    regressor: Regressor | None = None) -> np.ndarray:

        data_for_tau = []

        # get the ordered list of parameters
        if regressor is None:
            regressor  = Regressor.load()
        if classifier is None:
            classifier = Classifier.load()

        r_parameters = regressor.metadata.parameters_name
        parameters = copy(self.param_names)

        if 'tau_reio' in parameters:
            parameters = parameters[parameters != 'tau_reio']

        for param in parameters:
            
            if param == 'sum_mnu':
                param = 'm_nu1'

            if ((param in MP_KEY_CORRESPONDANCE) and (MP_KEY_CORRESPONDANCE[param] in r_parameters)) or param in r_parameters:
                data_for_tau.append(param)
        
        labels_correspondance = {value : key for key, value in MP_KEY_CORRESPONDANCE.items()}
        
        # get the data sample 
        data = np.empty((len(r_parameters), flat_chain.shape[-1])) 

        # find the ordering in which data_sample is set in prepare_data_plot
        indices_to_plot = [list(parameters).index(param) for param in data_for_tau if param in parameters]

        for ip, param in enumerate(r_parameters): 
            
            # if we ran the MCMC over that parameter
            if labels_correspondance[param] in parameters[indices_to_plot]:
                index = list(parameters[indices_to_plot]).index(labels_correspondance[param])
                data[ip] = flat_chain[index, :]
            elif param in parameters[indices_to_plot]:
                index = list(parameters[indices_to_plot]).index(param)
                data[ip] = flat_chain[index, :]
            else:
                data[ip, :] = DEFAULT_VALUES[param]


        #return data 
        tau = predict_tau_numpy(data.T, classifier, regressor)

        return tau
    

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



class MPSamples(Samples):
    """
    Class containing all chains of a MCMC analysis
    
    Parameters
    ----------
    path : str
        Path to the chains.
    ids : list | np.ndarray | None, optional
        List of chains to take into accoung. If none all possible found chains are added. By default None.

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


    def flat(self, discard: np.ndarray | None = None, thin: None | int = None, **kwargs):
        
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




    def print_best_fit(self, discard: np.ndarray | None = None ):
        
        samples_flat = self.flat(discard=discard, thin=1)
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
    



#################################################
## PLOTTING TOOLS


import matplotlib.pyplot as plt
import matplotlib.colors as mpc

from dataclasses import dataclass

LATEX_LABELS = {'omega_b' :  r'$\omega_{\rm b}$', 'omega_dm' : r'$\omega_{\rm dm}$', 'h' : r'$h$', 'ln10^{10}A_s' : r'$\ln 10^{10} A_{\rm s}$',
                'n_s' : r'$n_{\rm s}$', 'm_nu1' : r'$m_{\nu 1}~{\rm [eV]}$', 'sum_mnu' : r'$\sum {m_\nu}~{\rm [eV]}$', 'log10_f_star10' : r'$\log_{10}f_{\star, 10}$',
                  'alpha_star' : r'$\alpha_\star$', 'log10_f_esc10' : r'$\log_{10} f_{\rm esc, 10}$', 'alpha_esc' : r'$\alpha_{\rm esc}$',
                  't_star' : r'$t_\star$', 'log10_m_turn' : r'$\log_{10} M_{\rm turn}$', 'log10_lum_X' : r'$\log_{10} L_X$', 'nu_X_thresh' : r'$E_0$',
                  '1/m_wdm' : r'$\mu_{\rm WDM}$', 'mu_wdm' : r'$\mu_{\rm WDM}$', 'f_wdm' : r'$f_{\rm WDM}$', 'tau_reio' : r'$\tau$'}

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



def compute_quantiles(hist, edges, q):

    # normalising the histogram
    hist = hist/np.max(hist)

    y_arr = np.linspace(0.0, 1.0, 1000)
    f_arr = np.empty(len(y_arr))
    e_arr = np.empty((len(y_arr), 2))

    # precompute the sum of the total sum of the histogram
    s_hist  = np.sum(hist, axis=-1)

    for iy, y in enumerate(y_arr):

        f_arr[iy] = np.sum(hist[hist > y], axis=-1)/s_hist
        edges_bound_min  = edges[:-1][hist > y]
        edges_bound_max  = edges[1:][hist > y]
        e_arr[iy, 0] = np.min(edges_bound_min) if len(edges_bound_min) > 0 else np.nan
        e_arr[iy, 1] = np.max(edges_bound_max) if len(edges_bound_max) > 0 else np.nan

    iq_arr = np.where(f_arr > q)[0]

    # we could do much efficient by just computing f until it reaches
    # the desired value of confidence level
    # however, then we cannot do this check for multimodel distributions
    if not np.all(np.diff(iq_arr) == 1):
        print("Impossible to find a confidence interval for multimodal distribution")
        return 0, 0

    iq = iq_arr[-1]

    return e_arr[iq, 0], e_arr[iq, 1]


def generate_contours(samples: np.ndarray, bins: int = 20, q = [0.68, 0.95]) -> ProcessedData:

    data = ProcessedData()

    n = samples.shape[0]

    hists_1D   = np.empty((n, bins))
    hists_2D   = np.empty((n, n, bins, bins))    # 2D array of 2D array
    levels     = np.empty((n, n, len(q)+1)) # 2D array of 1D array

    quantiles  = np.empty((len(q), n, 2))

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
        hists_1D[i] = hists_1D[i] / np.max(hists_1D[i])

        # evaluate the quantiles
        for il, q_val in enumerate(q):
            try:
                quantiles[il, i, 0], quantiles[il, i, 1] = compute_quantiles(hists_1D[i], edges[i], q_val)
            except:
                print("impossible to compute quantiles for entry", i)

    
    # loop over all places with 2D histograms
    for i in range(1, n):
        for j in range(i):

            hists_2D[i, j, :, :], _, _ = np.histogram2d(samples[j, :], samples[i, :], bins=[edges[j, :], edges[i, :]], density=True)
            
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

            if show_points is True:

                # first thin the samples so that we plot only 5000 points
                n = data.samples.shape[-1]
                r = np.max([1, int(n/5000)])

                grid.get(axes[i], axes[j]).scatter(data.samples[j, ::r], data.samples[i, ::r], marker='o', edgecolors='none', color = contour_colors[-1], s=2, alpha=0.5)


            if show_hist is True:
                extent = [data.edges[j, 0], data.edges[j, -1], data.edges[i, 0], data.edges[i, -1]]
                grid.get(axes[i], axes[j]).imshow(data.hists_2D[i, j].T, origin='lower', extent=extent, cmap='Greys', aspect='auto')

            
            if show_surface is True:
                try:
                    grid.get(axes[i], axes[j]).contourf(*np.meshgrid(data.centers[j], data.centers[i]), data.hists_2D[i, j].T, levels=data.levels[i, j], colors=contour_colors)
                except ValueError as e:
                    print("Error for axis : ", i, j)
                    raise e

            if show_contour is True:
                grid.get(axes[i], axes[j]).contour(*np.meshgrid(data.centers[j], data.centers[i]), data.hists_2D[i, j].T, levels=data.levels[i, j], colors=contour_colors)


         

    
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
            grid.add_title(axes[i], r'${:.3g}$'.format(data.mean[i], color=colors[0]) + '$^{{ +{:.2g} }}_{{ -{:.2g} }}$'.format(data.quantiles[jq, i, 1] - data.mean[i], data.mean[i] - data.quantiles[jq, i, 0] ), color=title_color)  
    
    grid.update_titles()

    
    # (re)define the grid edges
    for j in range(0, data.size):
    
        new_boundaries = data.edges[:, [0, -1]]

        old_boundaries = grid.edges[axes]
        old_boundaries[old_boundaries[:, 0] == None] = data.edges[old_boundaries[:, 0] == None, :][:, [0, -1]]
    
        new_min = np.minimum(new_boundaries[:, 0], old_boundaries[:, 0]) 
        new_max = np.maximum(new_boundaries[:, 1], old_boundaries[:, 1])

        new_edges = np.vstack((new_min, new_max)).T

        grid.edges[axes, :] = new_edges 

        for i in range(j, data.size):
            grid.get(axes[i], axes[j]).set_xlim([grid.edges[j, 0], grid.edges[j, -1]])
            
            if i > j:
                grid.get(axes[i], axes[j]).set_ylim([grid.edges[i, 0], grid.edges[i, -1]])




    


def prepare_data_plot(samples: Samples, data_to_plot, discard = 0, thin = 1):

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


    data = samples.flat(discard=discard, thin=thin)
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

    


def get_xHII_stats(samples: Samples, 
                   data_to_plot: list[str] | np.ndarray, 
                   q: list[float] = [0.68, 0.95], 
                   bins: int = 30, 
                   discard: int = 0, 
                   thin: int = 1000,
                   *,
                   classifier: Classifier | None = None,
                   regressor: Regressor | None = None):

    data_for_xHII = []

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
            data_for_xHII.append(param)

    
    labels_correspondance = {value : key for key, value in MP_KEY_CORRESPONDANCE.items()}
    
    # get the data sample 
    data_sample = prepare_data_plot(samples, data_for_xHII, discard=discard, thin=thin)
    data = np.empty((len(parameters), data_sample.shape[-1])) 

    # find the ordering in which data_sample is set in prepare_data_plot
    indices_to_plot = [np.where(samples.param_names == param)[0][0] for param in data_for_xHII if param in samples.param_names]

    for ip, param in enumerate(parameters): 
        
        # if we ran the MCMC over that parameter
        if labels_correspondance[param] in samples.param_names[indices_to_plot]:
            index = list(samples.param_names[indices_to_plot]).index(labels_correspondance[param])
            data[ip] = data_sample[index, :]
        else:
            data[ip, :] = DEFAULT_VALUES[param]

    xHII = predict_xHII_numpy(theta=data.T, classifier=classifier, regressor=regressor)
    
    # here remove some outliers that should not have 
    # passed the likelihood condition
    if np.count_nonzero(xHII[:, -1]==-1)/len(xHII) > 0.01:
        warnings.warn("More than 1 percent of outliers with late reionization")

    xHII = xHII[xHII[:, -1] > 0]

    mean = np.mean(xHII, axis=0)
    med  = np.median(xHII, axis=0)

    z = regressor.metadata.z
    quantiles = np.empty((len(q), len(z), 2))

    # make an histogram for each value of z
    for iz, x in enumerate(np.log10(xHII.T)):
        hist, edges = np.histogram(x, bins = bins)
        for iq, q_val in enumerate(q):
            quantiles[iq, iz, 0], quantiles[iq, iz, 1] = compute_quantiles(hist, edges, q=q_val)
            quantiles[iq, iz, :] = 10**(quantiles[iq, iz, :])
    
    return z, mean, med, quantiles



#def add_tau_reio_to_sample(sample, pa)


def get_xHII_tanh_stats(samples: Samples, q: list[float] = [0.68, 0.95], bins: int = 30, 
                        discard: int = 0, thin : int = 100, x_inf: float = 2e-4):

    def xHII_class(z, z_reio = 8.0):
        return 0.5 * ( 1+np.tanh(  (1+z_reio)/(1.5*0.5)*(1-((1+z)/(1+z_reio))**(1.5))   ) )   

    index = list(samples.param_names).index('z_reio')
    z_reio = samples.flat(discard=discard, thin=thin)[index, :]

    z = np.linspace(0, 35, 200)
    xHII = xHII_class(z, z_reio[:, None]) + x_inf

    mean = np.mean(xHII, axis=0)
    med  = np.median(xHII, axis=0)

    quantiles = np.empty((len(q), len(z), 2))

    # make an histogram for each value of z
    for iz, x in enumerate(np.log10(xHII.T)):
        hist, edges = np.histogram(x, bins = bins)
        
        if np.all(np.diff(x) ==  0):
            for iq, q_val in enumerate(q):
                quantiles[iq, iz, 0], quantiles[iq, iz, 1] = x[0], x[1]
        else:
            for iq, q_val in enumerate(q):
                quantiles[iq, iz, 0], quantiles[iq, iz, 1] = compute_quantiles(hist, edges, q=q_val)
        
        for iq, q_val in enumerate(q):
            quantiles[iq, iz, :] = 10**(quantiles[iq, iz, :])
        
    return z, mean, med, quantiles