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


import random
import numpy as np
import torch
from scipy import interpolate
from os.path import abspath, exists

from typing import Self

from .cosmology import optical_depth_no_rad


LATEX_LABELS = {'hlittle' : r'$h$', 'Ln_1010_As' : r'$\ln (10^{10}A_{\rm s})$', 'F_STAR10' : r'$\log_{10} f_{\star, 10}$',
                  'ALPHA_STAR' : r'$\alpha_\star$', 't_STAR' : r'$t_\star$', 'F_ESC10' : r'$\log_{10} f_{\rm esc, 10} $', 
                  'ALPHA_ESC' : r'$\alpha_{\rm esc}$', 'M_TURN' : r'$\log_{10} M_{\rm turn}$', 'Omdmh2' : r'$\omega_{\rm dm}$', 
                  'Ombh2' : r'$\omega_{\rm b}$', 'POWER_INDEX' : r'$n_{\rm s}$', 'M_WDM' : r'$m_{\rm WDM}~{\rm [keV]}$', 
                  'INVERSE_M_WDM' : r'$\mu_{\rm WDM}$', 'FRAC_WDM' : r'$f_{\rm WDM}$', 'NEUTRINO_MASS_1' : r'$m_{\nu_1}$',
                  'LOG10_PMF_SB' : r'$\log_{10} s_{\rm B}$', 'PMF_NB' : r'$n_{\rm B}$', 'tau_reio' : r'$\tau$', 
                  'L_X' : r'$\log_{10} L_X$', 'NU_X_THRESH' : r'$E_0$', 'sum_mnu' : r'$\sum {m_\nu}~{\rm [eV]}$', 
                  'LOG10_PMF_SIGMA_B0' : r'$\log_{10}{\sigma_{\rm B, 0}}$'}


MP_KEY_CORRESPONDANCE = {'log10_f_star10' : 'F_STAR10', 'alpha_star' : 'ALPHA_STAR', 't_star' : 't_STAR', 'log10_f_esc10' : 'F_ESC10', 
                         'alpha_esc' : 'ALPHA_ESC', 'Omch2' : 'Omdmh2', 'omega_dm' : 'Omdmh2', 'omega_b' : 'Ombh2', 'h': 'hlittle', 'ln10^{10}A_s' : 'Ln_1010_As',
                         'n_s' : 'POWER_INDEX', 'm_nu1' : 'NEUTRINO_MASS_1', 'f_wdm' : 'FRAC_WDM', 'm_wdm' : 'M_WDM', 'nu_X_thresh' : 'NU_X_THRESH',
                         'log10_pmf_sb' : 'LOG10_PMF_SB', 'pmf_nb' : 'PMF_NB', 'log10_m_turn' : 'M_TURN', 'log10_lum_X' : 'L_X', '1/m_wdm' : 'INVERSE_M_WDM',
                         'log10_pmf_sigma_b0' : 'LOG10_PMF_SIGMA_B0'}

#LATEX_LABELS_CLASS = {'omega_b' :  r'$\omega_{\rm b}$', 'omega_dm' : r'$\omega_{\rm dm}$', 'h' : r'$h$', 'ln10^{10}A_s' : r'$\ln 10^{10} A_{\rm s}$',
#                'n_s' : r'$n_{\rm s}$', 'm_nu1' : r'$m_{\nu 1}~{\rm [eV]}$', 'sum_mnu' : r'$\sum {m_\nu}~{\rm [eV]}$', 'log10_f_star10' : r'$\log_{10}f_{\star, 10}$',
#                  'alpha_star' : r'$\alpha_\star$', 'log10_f_esc10' : r'$\log_{10} f_{\rm esc, 10}$', 'alpha_esc' : r'$\alpha_{\rm esc}$',
#                  't_star' : r'$t_\star$', 'log10_m_turn' : r'$\log_{10} M_{\rm turn}$', 'log10_lum_X' : r'$\log_{10} L_X$', 'nu_X_thresh' : r'$E_0$',
#                  '1/m_wdm' : r'$\mu_{\rm WDM}$', 'mu_wdm' : r'$\mu_{\rm WDM}$', 'f_wdm' : r'$f_{\rm WDM}$', 'tau_reio' : r'$\tau$'}


def latex_labels(labels: list[str]) -> list[str]:
    
    res = [None] * len(labels)

    for il, label in enumerate(labels):

        if label in LATEX_LABELS.keys(): 
            res[il] = LATEX_LABELS[label]
        elif label in MP_KEY_CORRESPONDANCE.keys():  
            if MP_KEY_CORRESPONDANCE[label] in LATEX_LABELS.keys():
                res[il] = LATEX_LABELS[MP_KEY_CORRESPONDANCE[label]]
            else:
                res[il] = label
        else:
            res[il] = label
        
    return res


def preprocess_raw_data(file_path: str, *, random_seed: int = 1994, frac_test: float = 0.1, frac_valid: float = 0.1, extras: list[str] | None = None) -> None:
    """
    Preprocess a raw .npz file. 
    Creates another numpy archive that can be directly used to create a :py:class:`DataSet` object.

    Parameters
    ----------
    file_path : str
        Path to the raw data file. The raw data must be a .npz file with the following information. 
        - z (or z_glob): redshift array
        - features_run:  Sequence of drawn input parameters for which there is a value for the ionization fraction
        - features_fail: Sequence of drawn input parameters for which the simulator failed because reionization was too late
        - ...
    random_seed : int, optional
        Random seed for splitting data into a training/validation/testing subset, by default 1994.
    frac_test : float, optional
       Fraction of the total data points in the test subset, by default 0.1.
    frac_valid : float, optional
        Fraction of the total data points in the validation subset, by default 0.1
    """

    # start by setting the random seed
    random.seed(random_seed)

    with open(file_path, 'rb') as file:

        # data must have been stored in a numpy archive with the correct format
        data = np.load(file, allow_pickle=True)
        z_glob               = data.get('z_glob', None)
        features_run         = data.get('features_run', None)
        features_super_late  = data.get('features_late', None)
        cosmology_run        = data.get('cosmology_run', None)
        cosmology_super_late = data.get('cosmology_late', None)
        parameters_min_val   = data.get('parameters_min_val', None)
        parameters_max_val   = data.get('parameters_max_val', None)
        xHIIdb               = data.get('xHIIdb', None)
        parameters_name      = data.get('parameters_name', None)
        ps                   = data.get('ps', None)
        
        extras_array = None
        if extras is not None and features_run is not None:
            extras_array = np.zeros((len(extras), features_run.shape[0]))
            for iex, ex in enumerate(extras):
                extras_array[iex] = data.get(ex, np.zeros(features_run.shape[0]))

        # Check for different notations
        z_glob = z_glob if (z_glob is not None) else data.get('z', None)
        features_super_late = features_super_late if (features_super_late is not None) else data.get('features_fail', None)

    # --------------------------

    new_redshifts = [5.9]

    # add some values we may need for the data classification
    new_z = np.sort(new_redshifts)
    for _z in new_z: 
        if _z not in z_glob:
            pos = np.searchsorted(z_glob, _z)
            value = interpolate.interp1d(z_glob, xHIIdb, kind='slinear')(_z)
            z_glob = np.sort(np.append(z_glob, _z))
            xHIIdb = np.insert(xHIIdb, pos, value, axis=1)


    features  = np.vstack((features_run, features_super_late))
    cosmology = np.concatenate((cosmology_run, cosmology_super_late)) 

    # total number of features
    n_r  = features_run.shape[0]
    n_sl = features_super_late.shape[0]
    n_tot = n_r + n_sl

    # parameters for xHIIdb
    xHIIdb = np.vstack((xHIIdb, np.zeros((n_sl, xHIIdb.shape[1]))))

    if ps is not None: ps = np.concatenate((ps, np.zeros((n_sl, ps.shape[1], ps.shape[2]))))

    if extras is not None:
        extras_array = np.hstack((extras_array, np.zeros((extras_array.shape[0], n_sl))))

    # shuffling all the data between late and run
    r = random.sample(range(n_tot), n_tot)

    features  = features[r]
    cosmology = cosmology[r]
    xHIIdb    = xHIIdb[r]
    ps        = ps[r]

    if extras is not None:
        for iex, _ in enumerate(extras_array):
            extras_array[iex] = extras_array[iex, r]

    # data selection, only considering the "early time" reionizations
    pos = np.searchsorted(z_glob, 5.9) 

    # define the early time reionization quantities
    indices_early = np.where(xHIIdb[:, pos] > 0.69)[0]

    # divide the early data into train, test and validation datasets
    # to that end shuffles indices early and grab slices of the shuffled dataset
    n_early = len(indices_early)
    r_early = random.sample(range(n_early), n_early)
    r_indices_early = indices_early[r_early]

    n_early_test  = int(frac_test*n_early)
    n_early_valid = int(frac_valid*n_early)

    indices_early_test  = np.sort(r_indices_early[:n_early_test])
    indices_early_valid = np.sort(r_indices_early[n_early_test:(n_early_test+n_early_valid)])
    indices_early_train = np.sort(r_indices_early[(n_early_test+n_early_valid):])

    # devide now the entire data into train, test and validation datasets
    r_indices_tot = random.sample(range(n_tot), n_tot)

    n_tot_test  = int(frac_test*n_tot)
    n_tot_valid = int(frac_valid*n_tot)

    indices_total_test  = np.sort(r_indices_tot[:n_tot_test])
    indices_total_valid = np.sort(r_indices_tot[n_tot_test:(n_tot_test+n_tot_valid)])
    indices_total_train = np.sort(r_indices_tot[(n_tot_test+n_tot_valid):])

    # save in file with _pp extension standing for "preprocessed"
    with open(file_path[:-4] + "_pp.npz", 'wb') as file:

        # data must have been stored in a numpy archive with the correct format
        np.savez(file, 
                 redshifts = z_glob, 
                 features = features, 
                 cosmology = cosmology, 
                 xHIIdb = xHIIdb, 
                 parameters_min_val = parameters_min_val, 
                 parameters_max_val = parameters_max_val, 
                 parameters_name = parameters_name,
                 indices_early_test  = indices_early_test,
                 indices_early_valid = indices_early_valid,
                 indices_early_train = indices_early_train,
                 indices_total_test  = indices_total_test,
                 indices_total_valid = indices_total_valid,
                 indices_total_train = indices_total_train,
                 random_seed = random_seed,
                 frac_test = frac_test,
                 frac_valid = frac_valid,
                 extras_array = extras_array,
                 extras_name  = extras,
                 ps = ps)


def true_to_uniform(x: float | np.ndarray,
                    min: float | np.ndarray,
                    max: float | np.ndarray) -> float | np.ndarray:
    """
    Transforms features uniformely distributed along [a, b] into features
    uniformely distributed between [0, 1] as fed to the neural networks.

    Parameters
    ----------
    x : float | np.ndarray
        input featurs distributed uniformely on [a, b]
    min : float | np.ndarray
        minimum value `a`
    max : float | np.ndarray
        maximum value `b`

    Returns
    -------
    float | np.ndarray

    Raises
    ------
    ValueError
        min should be less than max
    """

    if not np.all(min <= max):
        raise ValueError("The minimum value is bigger than the maximum one")
    
    return (x - min) / (max - min)

def uniform_to_true(x: float | np.ndarray,
                    min: float | np.ndarray,
                    max: float | np.ndarray) -> float | np.ndarray:
    """
    Inverse transformation of `true_to_uniform`.

    Parameters
    ----------
    x : float | np.ndarray
        input featurs distributed uniformely on [0, 1]
    min : float | np.ndarray
        minimum value `a`
    max : float | np.ndarray
        maximum value `b`

    Returns
    -------
    float | np.ndarray

    Raises
    ------
    ValueError
        min should be less than max
    """
    
    if not np.all(min <= max):
        raise ValueError("The minimum value is bigger than the maximum one")
    
    return (max - min) * x + min



class DataPartition:
    """
    DataPartition class. 

    Partitioning of the data into a training set, a testing set and a validation set.

    Parameters
    ----------
    early_train : np.ndarray
        indices of the data array with an early enough reionization used for training
    early_valid : np.ndarray
        indices of the data array with an early enough reionization used for validation
    early_test : np.ndarray
        indices of the data array with an early enough reionization used for testing
    total_train : np.ndarray
        all indices of the data array used for training
    total_valid : np.ndarray
        all indices of the data array used for validation
    total_test : np.ndarray
        all indices of the data array used for testing
    """

    def __init__(self, 
                 early_train: np.ndarray,
                 early_valid: np.ndarray, 
                 early_test: np.ndarray, 
                 total_train: np.ndarray, 
                 total_valid: np.ndarray, 
                 total_test: np.ndarray) -> None:
        
        self._early_dict = {'train' : early_train, 'valid': early_valid, 'test' : early_test}
        self._total_dict = {'train' : total_train, 'valid': total_valid, 'test' : total_test}
        
        self._early = np.sort(np.concatenate((self.early_test, self.early_valid, self.early_train)))

    def __call__(self):
        _new_early = {('early_' + k): val for k, val in self._early_dict.items()}
        _new_total = {('total_' + k): val for k, val in self._total_dict.items()}
        return (_new_early | _new_total)
    
    def __eq__(self, other):

        other_dict = other()

        for key, val in self().items():
            if (val is not None) and (other_dict[key] is not None):
                if len(val) != len(other_dict[key]): 
                    return False
                if np.any(val != other_dict[key]):
                    return False
            if (val is None) and (other_dict[key] is not None):
                return False
            if  (val is not None) and (other_dict[key] is None):
                return False
            
        return True
    
    def save(self, name:str) -> None:
        """
        Save the data partition.

        Parameters
        ----------
        name : str
            name of the data partition file
        """

        with open(name + '.npz', 'wb') as file:
            np.savez(file = file, 
                     early_train = self.early_train, 
                     early_valid = self.early_valid,
                     early_test  = self.early_test,
                     total_train = self.total_train,
                     total_valid = self.total_valid,
                     total_test  = self.total_test)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load a previously saved data partition.

        Parameters
        ----------
        path : str
            path to the data partition saved file.

        Returns
        -------
        DataPartition
        """
        
        with open(path + '.npz', 'rb') as file:
            data = np.load(file, allow_pickle=False)    
            return DataPartition(data.get('early_train'), 
                                 data.get('early_valid'), 
                                 data.get('early_test'), 
                                 data.get('total_train'), 
                                 data.get('total_valid'), 
                                 data.get('total_test'))
    
    @property
    def early_train(self):
        return self._early_dict['train']
    
    @property
    def early_valid(self):
        return self._early_dict['valid']
    
    @property
    def early_test(self):
        return self._early_dict['test']
    
    @property
    def total_train(self):
        return self._total_dict['train']
    
    @property
    def total_valid(self):
        return self._total_dict['valid']
    
    @property
    def total_test(self):
        return self._total_dict['test']
    
    @property
    def early(self):
        return self._early
    


class MetaData:
    """
    MetaData class
    
    Metadata that is saved with the neural network for predictions.

    Parameters
    ----------
    z : np.ndarray
        array of redshifts
    parameters_name : list | np.ndarray
        name of the parameters (input features)
    parameters_min_val : np.ndarray
        minimum value of the parameters (input features)
    parameters_max_val : np.ndarray
        maximum value of the parameters (input features)
    """

    def __init__(self,
                 z: np.ndarray,
                 parameters_name: list | np.ndarray,
                 parameters_min_val: np.ndarray,
                 parameters_max_val: np.ndarray) -> None:
       
        self._z                  = z
        self._parameters_name    = np.array(parameters_name)
        self._parameters_min_val = parameters_min_val
        self._parameters_max_val = parameters_max_val

        # derived quantities
        self._pos_omega_b  = np.where(self.parameters_name == 'Ombh2')[0][0]
        self._pos_omega_dm = np.where(self.parameters_name == 'Omdmh2')[0][0]
        self._pos_hlittle  = np.where(self.parameters_name == 'hlittle')[0][0]

        self._min_omega_b = self._parameters_min_val[self._pos_omega_b]
        self._min_omega_c = self._parameters_min_val[self._pos_omega_dm]
        self._min_hlittle = self._parameters_min_val[self._pos_hlittle]

        self._max_omega_b = self._parameters_max_val[self._pos_omega_b]
        self._max_omega_c = self._parameters_max_val[self._pos_omega_dm]
        self._max_hlittle = self._parameters_max_val[self._pos_hlittle]

        # principal component analysis quantities

        self._pca_mean_values  = np.empty(0)
        self._pca_eigenvalues  = np.empty(0)
        self._pca_eigenvectors = np.empty((0, 0))
        self._pca_n_eigenvectors = 0

        self._torch_pca_mean_values  = torch.tensor(self._pca_mean_values,  dtype=torch.float32)
        self._torch_pca_eigenvectors = torch.tensor(self._pca_eigenvectors, dtype=torch.float32)
        
    

    def __call__(self):
        return {'z' : self._z, 
                'parameters_name'    : self._parameters_name, 
                'parameters_min_val' : self._parameters_min_val,
                'parameters_max_val' : self._parameters_max_val,
                'pca_eigenvalues'    : self._pca_eigenvalues,
                'pca_eigenvectors'   : self._pca_eigenvectors,
                'pca_mean_values'    : self._pca_mean_values}

    def __eq__(self, other):
        
        other_dict = other()

        for key, val in self().items():
            if (val is not None) and (other_dict[key] is not None):
                if len(val) != len(other_dict[key]): 
                    return False
                if np.any(val != other_dict[key]):
                    return False
            if (val is None) and (other_dict[key] is not None):
                return False
            if  (val is not None) and (other_dict[key] is None):
                return False
        
        if self.pca_n_eigenvectors != other.pca_n_eigenvectors:
            return False
            
        return True
    
    def save(self, name: str) -> None:
        """
        Save the metadata.

        Parameters
        ----------
        name : str
            name of the metadata file
        """

        with open(name + '.npz', 'wb') as file:
            np.savez(file = file, 
                     z = self.z, 
                     parameters_name = self.parameters_name,
                     parameters_min_val = self.parameters_min_val,
                     parameters_max_val = self.parameters_max_val,
                     pca_mean_values    = self.pca_mean_values,
                     pca_eigenvalues    = self.pca_eigenvalues,
                     pca_eigenvectors   = self.pca_eigenvectors,
                     pca_n_eigenvectors = self.pca_n_eigenvectors)

    @classmethod
    def load(cls, path: str) -> Self:
        """
        Load a previously saved metadata file.

        Parameters
        ----------
        path : str
            path to the metadata saved file.

        Returns
        -------
        MetaData
        """
        
        with open(path + '.npz', 'rb') as file:
            data = np.load(file, allow_pickle=True)    
            metadata = MetaData(data.get('z'), 
                            data.get('parameters_name'), 
                            data.get('parameters_min_val'), 
                            data.get('parameters_max_val'))
            
            # get the pca decomposition
            metadata._pca_eigenvalues    = data.get('pca_eigenvalues',    np.empty(0))
            metadata._pca_eigenvectors   = data.get('pca_eigenvectors',   np.empty(0))
            metadata._pca_mean_values    = data.get('pca_mean_values',    np.empty(0))
            metadata._pca_n_eigenvectors = data.get('pca_n_eigenvectors', 0)

            metadata._torch_pca_mean_values  = torch.tensor(metadata.pca_mean_values,  dtype=torch.float32)
            metadata._torch_pca_eigenvectors = torch.tensor(metadata.pca_eigenvectors, dtype=torch.float32)

            return metadata

    @property
    def z(self):
        return self._z
    
    @property
    def parameters_name(self):
        return self._parameters_name
    
    @property
    def parameters_min_val(self):
        return self._parameters_min_val
    
    @property
    def parameters_max_val(self):
        return self._parameters_max_val
    
    @property
    def pos_omega_b(self):
        return self._pos_omega_b
    
    @property
    def pos_omega_dm(self):
        return self._pos_omega_dm
    
    @property
    def pos_hlittle(self):
        return self._pos_hlittle
    
    @property
    def min_omega_b(self):
        return self._min_omega_b
    
    @property
    def min_omega_c(self):
        return self._min_omega_c
    
    @property
    def min_hlittle(self):
        return self._min_hlittle
    
    @property
    def max_omega_b(self):
        return self._max_omega_b
    
    @property
    def max_omega_c(self):
        return self._max_omega_c
    
    @property
    def max_hlittle(self):
        return self._max_hlittle
    
    @property
    def pca_mean_values(self):
        return self._pca_mean_values
    
    @property
    def pca_eigenvalues(self):
        return self._pca_eigenvalues
    
    @property
    def pca_eigenvectors(self):
        return self._pca_eigenvectors
    
    @property
    def pca_n_eigenvectors(self):
        return self._pca_n_eigenvectors
    
    @property
    def torch_pca_eigenvectors(self):
        return self._torch_pca_eigenvectors
    
    @property
    def torch_pca_mean_values(self):
        return self._torch_pca_mean_values
    




class DataSet:
    """
    DataSet class

    Compile the data necessary for training.

    Parameters
    ----------
    file_path: str
        path to the file that contains the raw data
    z: np.ndarray
        array of the redshits of interpolation of the nn
    use_PCA: bool, optional
        prepare the data to perform the regression in the principal component basis, default is True
    precision_PCA: float, optional
        if use_PCA is `True`, select the number of useful eigenvectors from this coefficient 
        -- only the eigenvectors with eigenvalues larger than precision_PCA * the largest eigenvalue
        are considered as useful
    frac_test: float, optional 
        fraction of test data out of the total sample, default is 0.1
    frac_valid: float, optional
        fraction of validation data out of the total sample, default is 0.1
    seed_split: int, optional
        random seed for data partitioning, default is 1994
    """

    def __init__(self, 
                 file_path : str, 
                 z : np.ndarray | None = None, 
                 *, 
                 frac_test: float  = 0.1, 
                 frac_valid: float = 0.1,
                 seed_split: int   = 1994,
                 extras: list[str] | None = None) -> None:

        # --------------------------------
        # initialisation from input values 

        # directory was the data is stored
        self._file_path = abspath(file_path)

        # define a default redshift array on which to make the predictions
        # define the labels of the regressor
        if z is None:
            _z = np.array([4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 
                                        5.9, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 
                                        8, 8.25, 8.5, 8.75, 9, 9.5, 10, 10.5, 11, 
                                        11.5, 12, 12.5, 13, 13.5, 14, 14.5, 15, 
                                        15.5, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                                        25, 26, 27, 29, 31, 33, 35])
        else:
            _z = z


        # -----------------------------
        # prepare and read the datafile

        # if raw data has not yet been preprocessed
        if not exists(file_path[:-4]+ "_pp.npz"):
            preprocess_raw_data(file_path, random_seed=seed_split, frac_test = frac_test, frac_valid = frac_valid, extras = extras)
        else:
            with open(file_path[:-4]+ "_pp.npz", 'rb') as file:
                data = np.load(file, allow_pickle=True)
                
                # if we do not have the same seed or fraction of valid and test samples we preprocess the data again
                if frac_test != data.get('frac_test', None) or frac_valid != data.get('frac_valid', None) or seed_split != data.get('random_seed', None):
                    preprocess_raw_data(file_path, random_seed=seed_split, frac_test = frac_test, frac_valid = frac_valid, extras = extras)


        self._extras_array = None
        self._extras_name  = None

        with open(file_path[:-4]+ "_pp.npz", 'rb') as file:
            data = np.load(file, allow_pickle=True)
            
            self._redshifts = data.get('redshifts', None)
            self._features  = data.get('features',  None)
            self._cosmology = data.get('cosmology', None)
            self._xHIIdb    = data.get('xHIIdb',    None)
            self._ps        = data.get('ps', None)

            # possibility to add extra values for each run
            if isinstance(data.get('extras_array', None), np.ndarray):
                self._extras_array = data.get('extras_array')
                self._extras_name  = data.get('extras_name', None)

            # define a metadata object
            self._metadata = MetaData(_z, 
                                      data.get('parameters_name',    None),
                                      data.get('parameters_min_val', None),
                                      data.get('parameters_max_val', None))

            # define a partition object
            self._partition = DataPartition(data.get('indices_early_train', None),
                                            data.get('indices_early_valid', None),
                                            data.get('indices_early_test',  None),
                                            data.get('indices_total_train', None),
                                            data.get('indices_total_valid', None),
                                            data.get('indices_total_test',  None))        

        n_tot = len(self._features)

        # evaluate the optical depth to reionization for all runs
        # this is done with an optimised function for the evaluation of tau with numpy arrays
        # assume a late time universe with no radiation (very good approximation)
        self._tau = optical_depth_no_rad(self._redshifts, self._xHIIdb, 
                                                     self._features[:, self.metadata.pos_omega_b], 
                                                     self._features[:, self.metadata.pos_omega_dm],
                                                     self._features[:, self.metadata.pos_hlittle])
        
        self._x_array = true_to_uniform(self._features, self.metadata.parameters_min_val, self.metadata.parameters_max_val)
        
        # set 0 to the late reionization and 1 to that are early enough
        self._y_classifier = np.zeros(len(self._features))
        self._y_classifier[self.partition.early] = 1.0

           
        self._y_regressor = np.zeros((n_tot, len(self.metadata.z) + 1))
        for i in range(n_tot):
            self._y_regressor[i, -1] = self._tau[i]
            if i in self.partition.early:
                self._y_regressor[i, :-1] = interpolate.interp1d(self._redshifts, self._xHIIdb[i, :])(self.metadata.z)
            
        # convert to float32 objects
        self._x_array      = np.array(self._x_array, np.float32)
        self._y_classifier = np.array(self._y_classifier, np.float32)
        self._y_regressor  = np.array(self._y_regressor, np.float32)

        if self._extras_array is not None:
            self._extras_array = np.array(self._extras_array, np.float32)
        # --------------------------


    def init_principal_components(self, pca_precision:float = 1e-3) -> int:
        """
        Initialise the principal component analysis decomposition

        Parameters
        ----------
        pca_precision : float, optional
            precision for the principal analysis reconstruction, by default 1e-3

        Returns
        -------
        int
            number of necessary eigenvectors to reach the desired precision
        """

        # array on which we perform the principal component analysis
        arr = interpolate.interp1d(self._redshifts, np.log10(self._xHIIdb[self.partition.early_train, :]))(self.metadata.z)
        
        # mean of the functions
        pca_mean_values = np.mean(arr, axis=0)

        # shift the function to a centered distribution
        arr_centered = arr - pca_mean_values

        # self-covariance matrix
        cov = np.cov(arr_centered, rowvar=False)

        # eigenvalues of the covariance operator
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # reorganise the eigenvector from decreasing eigenvalues
        # and transpose the eigenvector matrix
        # now eigenvector[i, :] is the i-th eigenvector
        idx = np.argsort(eigenvalues)[::-1]
        pca_eigenvalues  = eigenvalues[idx]
        pca_eigenvectors = eigenvectors[:, idx].T

        # define the number of vectors from the precision on the eigenvalues
        pca_n_eigenvectors = np.where(np.sqrt(pca_eigenvalues/pca_eigenvalues[0]) < pca_precision)[0][0]

        # update the metadata accordingly
        self.metadata._pca_mean_values    = pca_mean_values
        self.metadata._pca_eigenvalues    = pca_eigenvalues
        self.metadata._pca_eigenvectors   = pca_eigenvectors
        self.metadata._pca_n_eigenvectors = pca_n_eigenvectors

        # define de torch version of the eigenvectors and mean values
        self.metadata._torch_pca_mean_values  = torch.tensor(self.metadata.pca_mean_values,  dtype=torch.float32)
        self.metadata._torch_pca_eigenvectors = torch.tensor(self.metadata.pca_eigenvectors, dtype=torch.float32)

        return pca_n_eigenvectors
    

    @property
    def z(self):
        return self._z
    
    @property
    def x_array(self):
        return self._x_array
    
    @property
    def y_classifier(self):
        return self._y_classifier
    
    @property
    def y_regressor(self):
        return self._y_regressor
    
    @property
    def metadata(self):
        return self._metadata
    
    @property
    def partition(self):
        return self._partition

    @property
    def tau(self):
        return self._tau
    
    @property
    def extras_array(self):
        return self._extras_array
    
    @property
    def extras_name(self):
        return self._extras_name
    
    @property
    def ps(self):
        return self._ps
    



class TorchDataset(torch.utils.data.Dataset):
    """
    Wrapper of torch Dataset.

    Parameters
    ----------
    x_data : np.ndarray
        input features
    y_data : np.ndarray
        output labels
    """
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = x_data
        self.y_data = y_data
    
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx: int) -> tuple:
        x = self.x_data[idx]
        y = self.y_data[idx]
        return x, y




