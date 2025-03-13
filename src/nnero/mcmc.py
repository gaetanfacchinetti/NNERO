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
import warnings

import os, pickle
import pkg_resources

from abc import ABC, abstractmethod
from typing import Callable

from scipy       import special, interpolate
from .data       import uniform_to_true
from .predictor  import predict_tau_from_xHII_numpy, predict_xHII_numpy, DEFAULT_VALUES
from .classifier import Classifier
from .regressor  import Regressor

from .astrophysics import m_halo, phi_uv, dn_dm
from .cosmology import h_factor_no_rad, ShortPowerSpectrumRange
from .constants import CONVERSIONS, CST_MSOL_MPC

CLASSY_IMPORTED = False

try:
    import classy
    CLASSY_IMPORTED = True
except:
    pass


LIKELIHOOD_DATA_PATH = pkg_resources.resource_filename('nnero', 'lkl_data/')


######################
## Define likelihood at fixed cosmology

class Likelihood(ABC):

    def __init__(self, parameters: list[str]) -> None:

        self._parameters = parameters
        self._index = {key: i for i, key in enumerate(parameters)} 


    def get_x(self,  theta: np.ndarray, xi: np.ndarray) -> np.ndarray:
        
        if len(theta.shape) == 1:
            theta = theta[None, :]

        if len(xi.shape) == 1:
            xi = np.tile(xi, (theta.shape[0], 1))

        return np.hstack((theta, xi))
    

    def get_x_dict(self, x: np.ndarray) -> dict:
        return {param:x[:,self._index[param]] for param in self._parameters}

    def get_x_dicts(self, x:np.ndarray) -> list[dict]:
        return [ {param:x[i,self._index[param]] for param in self._parameters} for i in range(x.shape[0])]
            
   
    def loglkl(self, theta: np.ndarray, xi: np.ndarray, **kwargs) -> np.ndarray:

        x = self.get_x(theta, xi)

        if x.shape[-1] != len(self.parameters):
            raise ValueError('theta and xi should make an array that is the size of the parameter vector' )
        
        return self._loglkl(x, **kwargs)


    @abstractmethod
    def _loglkl(self, x, **kwargs):
        pass

    @property
    def parameters(self):
        return self._parameters
    
    @property
    def index(self):
        return self._index
    
    @property
    def vectorizable(self):
        return self._vectorizable


class UVLFLikelihood(Likelihood):
    """
    Likelihood for the UV luminosity functions.

    Parameters
    ----------
    k : np.ndarray
        Array of modes on which the matter power spectrum is given (in 1/Mpc).
    pk : np.ndarray
        Matter power spectrum.
    """

    def __init__(self, 
                 parameters: list[str],  
                 *,
                 parameters_xi: np.ndarray | None = None,
                 xi: np.ndarray | None = None,
                 k: np.ndarray | None = None, 
                 pk: np.ndarray | None = None,
                 precompute: bool = False) -> None:
    
    
        self._k  = k
        self._pk = pk

        if precompute:

            if (self._k is None) and (self._pk is None):

                if CLASSY_IMPORTED is True:

                    h = 0.7
                    omega_m = 0.3*(h**2)
                    cosmo = classy.Class()
                    
                    params_cosmo = {'output' : 'mPk', 'P_k_max_h/Mpc' : 650.0, 'h' : 0.7, 'omega_m' : omega_m}

                    if parameters_xi is not None and 'Ln1010As' in parameters_xi:
                        params_cosmo['ln10^{10}A_s'] = xi[parameters_xi.index('Ln1010As')]
                    
                    if parameters_xi is not None and 'Ombh2' in parameters_xi:
                        params_cosmo['omega_b'] = xi[parameters_xi.index('Ombh2')]

                    if parameters_xi is not None and 'POWER_INDEX' in parameters_xi:
                        params_cosmo['n_s'] = xi[parameters_xi.index('POWER_INDEX')]

                    if parameters_xi is None:
                        print("Attention: matter power spectrum evaluated for a default cosmology.")

                    cosmo.set(params_cosmo)
                    cosmo.compute()

                    self._k     = np.logspace(-2.5, np.log10(cosmo.pars['P_k_max_h/Mpc'] * h), 50000)
                    self._pk    = np.array([cosmo.pk_lin(_k, 0) for _k in self._k]) 

                else: 
                    raise ValueError("Need to import CLASS to pecompute the matter power spectrum if not given as input.")
        
        self.sheth_a = 0.322
        self.sheth_q = 1.0
        self.sheth_p = 0.3 
        self.c       = 2.5
        self.window  = 'sharpk'

        elements_to_check = ['ALPHA_STAR', 't_STAR', 'F_STAR10', 'M_TURN']
        if not np.all([element in parameters for element in elements_to_check]):
            raise ValueError('In order to use UVLFLikelihood, need to pass alpha_star, t_star, log10_f_star10 and log10_m_turn as inpur parameters')

        with open(os.path.join(LIKELIHOOD_DATA_PATH, "UVData.pkl"), 'rb') as file:
        
            data_uv = pickle.load(file)

            self.z_uv_exp = data_uv['z_uv']
            self.m_uv_exp = data_uv['m_uv']

            self.phi_uv_exp = data_uv['p_uv']

            self.sigma_phi_uv_down = data_uv['sigma_p_uv_down']
            self.sigma_phi_uv_up   = data_uv['sigma_p_uv_up']

        self._z_array = []
        self._z_index_from_table = [[] for j in [0, 1, 2, 3]]

        self._precompute = precompute
        self._masses = np.empty(0)
        self._dndmh  = np.empty(0)

        # if we can precompute the halo mass function 
        # because the matter power spectrum does not vary
        # then we do it here
        if self._precompute:

            for j in [0, 1, 2, 3]:

                # loop on the redshift bins
                for z in self.z_uv_exp[j]:

                    if z not in self._z_array:
                        self._z_array.append(z)
                    
                    self._z_index_from_table[j].append(self._z_array.index(z))


            self._z_array = np.array(self._z_array)

            rhom0  = (0.3*(0.7**2)) * CST_MSOL_MPC.rho_c_over_h2        
            m_min  = 1.1 * 4.0*np.pi/3.0 * rhom0 / (self.k[-1] / self.c)**3
            m_max  = 4.0*np.pi/3.0 * rhom0 / (self.k[0]/ self.c)**3  / 1e+3

            self._masses = np.logspace(np.log10(m_min), np.log10(m_max), 5000)

            #precompute the halo mass function
            self._dndmh = dn_dm(self._z_array, self._masses, self.k, self.pk, omega_m = (0.3*(0.7**2)), h=0.7, sheth_a=self.sheth_a, sheth_q=self.sheth_q, sheth_p=self.sheth_p, window=self.window, c=self.c)

        super().__init__(parameters)


    @property
    def k(self):
        return self._k
    
    @property
    def pk(self):
        return self._pk



    def _loglkl(self, x, **kwargs):
     
        omega_b = x[:, self.index['Ombh2']]

        h       = np.full_like(omega_b, fill_value=0.7)
        omega_m = np.full_like(omega_b, fill_value=0.3 * (0.7**2))


        alpha_star = x[:, self.index['ALPHA_STAR']]
        t_star     = x[:, self.index['t_STAR']]
        f_star10   = 10**x[:, self.index['F_STAR10']]
        m_turn     = 10**x[:, self.index['M_TURN']]
     
        log_lkl = np.zeros(x.shape[0])

        # if we pass pk in argument then we cannot use 
        # the precoputed values for the halo mass function
        # (that is self._dndmh)
        precompute = self._precompute
        if kwargs.get('pk', None) is not None:
            precompute = False

        # allow k and pk to be passed as extra arguments of the function
        # in the case they depend on some parameters in theta
        k  = kwargs.get('k', self.k)
        pk = kwargs.get('pk', self.pk)

        if len(k.shape) == 1:  k  = np.tile(k, (omega_b.shape[0], 1))
        if len(pk.shape) == 1: pk = np.tile(pk, (omega_b.shape[0], 1))

        # loop on the datasets
        # we do not include Bouwens et al 2015 (10.1088/0004-637X/803/1/34)
        # stored at index 0, therefore we start the loop at 1
        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(self.z_uv_exp[j]):

                hz = 100 * h_factor_no_rad(z, omega_b, omega_m - omega_b, h) * CONVERSIONS.km_to_mpc
                mh, mask_mh = m_halo(hz, self.m_uv_exp[j][iz], alpha_star, t_star, f_star10, omega_b, omega_m)

                try:

                    if precompute is False:
                        # predict the UV luminosity function on the range of magnitude m_uv at that redshift bin
                        # in the future, could add sheth_a, sheth_q, sheth_p and c as nuisance parameters
                        phi_uv_pred_z = phi_uv(z, hz, self.m_uv_exp[j][iz], k, pk, alpha_star, t_star, f_star10, m_turn, omega_b, omega_m, h, 
                                                    self.sheth_a, self.sheth_q, self.sheth_p, window = self.window, c = self.c, mh = mh, mask = mask_mh)
                        

                    else:

                        dndmh = interpolate.interp1d(self._masses, self._dndmh[0, self._z_index_from_table[j][iz], :])(mh)
                        phi_uv_pred_z = phi_uv(z, hz, self.m_uv_exp[j][iz], k, pk, alpha_star, t_star, f_star10, m_turn, omega_b, omega_m, h, 
                                                    self.sheth_a, self.sheth_q, self.sheth_p, window = self.window, c = self.c, mh = mh, 
                                                    mask = mask_mh, dndmh = dndmh)    
                    
                    phi_uv_pred_z = np.squeeze(phi_uv_pred_z, axis=1)

                    #if np.any(np.isnan(phi_uv_pred_z.flatten())):
                        #print(z, hz, self.m_uv_exp[j][iz], k, pk, alpha_star, t_star, f_star10, m_turn, omega_b, omega_m, h, mh, mask_mh)
                        #raise ValueError('Arghhh')
                    
                except ShortPowerSpectrumRange:
                    # kill the log likelihood in that case by setting it to -infinity
                    raise ValueError('Power spectrum not evaluated on a large enough range')               

                # get a sigma that is either the down or the up one depending 
                # if prediction is lower / higher than the observed value        
                mask         = (phi_uv_pred_z > self.phi_uv_exp[j][iz][None, :])
                sigma        = np.tile(self.sigma_phi_uv_down[j][iz], (x.shape[0], 1))
                sigma[mask]  = np.tile(self.sigma_phi_uv_up[j][iz],  (x.shape[0], 1))[mask]
        
                # update the log likelihood
                contrib = np.sum(np.log(np.sqrt(2.0/np.pi)/(self.sigma_phi_uv_up[j][iz] + self.sigma_phi_uv_down[j][iz])) - (phi_uv_pred_z - self.phi_uv_exp[j][iz])**2/(2*(sigma**2)), axis=-1)
                log_lkl = log_lkl + contrib

        return log_lkl
    


    def get_k_max(self, x) -> (None | float):
        
    
        omega_b = x[:, self.index['Ombh2']]

        h       = np.full_like(omega_b, fill_value=0.7)
        omega_m = np.full_like(omega_b, fill_value=0.3 * (0.7**2))

        alpha_star = x[:, self.index['ALPHA_STAR']]
        t_star     = x[:, self.index['t_STAR']]
        f_star10   = 10**x[:, self.index['F_STAR10']]

        min_mh = np.full(x.shape[0], fill_value=np.inf)

        for j in [1, 2, 3]:

            # loop on the redshift bins
            for iz, z, in enumerate(self.z_uv_exp[j]):

                hz = 100 * h_factor_no_rad(z, omega_b, omega_m - omega_b, h) * CONVERSIONS.km_to_mpc # approximation of the hubble factor
                mh, _ = m_halo(hz, self.m_uv_exp[j][iz], alpha_star, t_star, f_star10, omega_b, omega_m)

                mh = np.squeeze(mh, axis=1)   
                min_mh = np.minimum(min_mh, np.min(mh, axis=-1))
                    
        rhom0  = omega_m * CST_MSOL_MPC.rho_c_over_h2        
        k_max = 1.3 * self.c * (3*min_mh/(4*np.pi)/rhom0)**(-1/3)

        # one should (almost) never need self.kmax if large enough
        # set here as a security to do not make CLASS take to much
        # time and crash
        # print("Value of k_max: ", np.minimum(k_max / h, 10000))
        return np.minimum(k_max / h, 10000)





class OpticalDepthLikelihood(Likelihood):

    def __init__(self, 
                 parameters: list[str],  
                 *,
                 classifier: Classifier,
                 regressor:  Regressor,
                 median_tau:   float  = 0.0544,
                 sigma_tau:  float = 0.0073) -> None:
        
        self._classifier = classifier
        self._regressor  = regressor 
        self._median_tau   = median_tau
        self._sigma_tau  = sigma_tau   

        if not isinstance(self._sigma_tau, np.ndarray):
            self._sigma_tau = np.array([self._sigma_tau])

        super().__init__(parameters)

        # define an 'order list' to reorganise the parameters
        # in the order they are passed to the classifier and regressor
        ordered_params = regressor.metadata.parameters_name
        self._order    = [self.parameters.index(param) for param in ordered_params if param in self.parameters]

        
    @property
    def classifier(self):
        return self._classifier
    
    @property
    def regressor(self):
        return self._regressor
    
    @property
    def median_tau(self):
        return self._median_tau
    
    @property
    def sigma_tau(self):
        return self._sigma_tau


    def _loglkl(self, x, **kwargs):

        # get the number of parallel evaluations
        n = x.shape[0]

        xx = np.array([DEFAULT_VALUES[param] for param in self.regressor.metadata.parameters_name])
        xx = np.tile(xx, (n, 1))

        indices = np.array([list(self.regressor.metadata.parameters_name).index(param) for param in self.regressor.metadata.parameters_name if param in self.parameters])
        
        xx[:, indices] = x[:, self._order]

        # predict the ionization fraction from the NN
        xHII = predict_xHII_numpy(xx, self.classifier, self.regressor)

        # setting the result to -inf when the classifier returns it as a wrong value
        #loglkl = np.zeros(n)
        #loglkl[xHII[:, 0] == -1] = -np.inf
        # get the values in input (if given) or initialise to Planck 2018 results
        #tau     = self.mean_tau
        #var_tau = self.sigma_tau**2
        #loglkl = loglkl - 0.5 * ((tau- tau_pred)**2/var_tau + np.log( 2*np.pi * var_tau))

        # setting the result to -inf when the classifier returns it as a wrong value
        # initialise the result
        res = np.zeros(n)
        res[xHII[:, 0] == -1] = -np.inf

        # compute the optical depth to reionization
        tau_pred = predict_tau_from_xHII_numpy(xHII, xx, self.regressor.metadata)
        
        s_tau_m = np.full(n, fill_value=self.sigma_tau[0])
        s_tau_p = np.full(n, fill_value=self.sigma_tau[1])
        m_tau   = np.full(n, fill_value=self.median_tau) - s_tau_p * np.sqrt(2.0) * special.erfcinv(0.5*(1+s_tau_m/s_tau_p))

        # compute the truncated gaussian for the reionization data
        mask  = tau_pred > m_tau
        s_tau = s_tau_m
        s_tau[mask] = s_tau_p[mask]

        res = res + np.log(np.sqrt(2.0/np.pi)/(s_tau_m + s_tau_p)) - (tau_pred - m_tau)**2/(2*s_tau**2)

        return res
    
    


class ReionizationLikelihood(Likelihood):

    def __init__(self, 
                 parameters: list[str],
                 *,
                classifier: Classifier,
                regressor:  Regressor):
        
                
        self._classifier = classifier
        self._regressor  = regressor 

        # defining McGreer data
        # https://doi.org/10.1093/mnras/stu2449
        self.z_reio     = np.array([5.6, 5.9])
        self.x_reio     = np.array([0.96, 0.94])
        self.std_x_reio = np.array([0.05, 0.05])

        super().__init__(parameters)

        # define an 'order list' to reorganise the parameters
        # in the order they are passed to the classifier and regressor
        ordered_params = regressor.metadata.parameters_name
        self._order    = [self.parameters.index(param) for param in ordered_params if param in self.parameters]
        
       

    @property
    def classifier(self):
        return self._classifier
    
    @property
    def regressor(self):
        return self._regressor
    



    def _loglkl(self, x, **kwargs):

        # get the number of parallel evaluations
        n = x.shape[0]

        xx = np.array([DEFAULT_VALUES[param] for param in self.regressor.metadata.parameters_name])
        xx = np.tile(xx, (n, 1))

        indices = np.array([list(self.regressor.metadata.parameters_name).index(param) for param in self.regressor.metadata.parameters_name if param in self.parameters])
        
        xx[:, indices] = x[:, self._order]

        # predict the ionization fraction from the NN
        xHII_full = predict_xHII_numpy(xx, self.classifier, self.regressor)
        xHII = interpolate.interp1d(self.regressor.z, xHII_full)(self.z_reio)

        # setting the result to -inf when the classifier returns it as a wrong value
        # initialise the result
        res = np.zeros((n, *self.z_reio.shape))
        res[xHII[:, 0] == -1] = -np.inf
        
        std = np.tile(self.std_x_reio, (n, 1))
        x_reio = np.tile(self.x_reio, (n, 1))

        # compute the truncated gaussian for the reionization data
        norm_reio = -np.log(1.0 - x_reio + np.sqrt(np.pi/2.0)*std*special.erf(x_reio/(np.sqrt(2))/std))
        res = res + norm_reio

        mask = xHII < x_reio
        

        res[mask] = res[mask] - 0.5 * (xHII[mask] - x_reio[mask])**2/(std[mask]**2)

        return np.sum(res, axis=-1)
    

    

##############
# MCMC related functions

def log_prior(theta: np.ndarray, 
              theta_min: np.ndarray, 
              theta_max: np.ndarray, 
              **kwargs) -> np.ndarray:
    
    """
    Natural logarithm of the prior

    assume flat prior except for the parameters for which
    a covariance matrix and average value are given

    Parameters:
    -----------
    - theta: (n, d) ndarray
        parameters
        d is the dimension of the vector parameter 
        n is the number of vector parameter treated at once
    - theta_min: (d) ndarray
        minimum value of the parameters allowed
    - theta_max:
        maximum value of the parameters allowed

    kwargs:
    -------
    - mask: optional, (d) ndarray
        where the covariance matrix applies
        the mask should have p Trues and d-p False
        with p the dimension of the covariance matrix
        if cov and my given with dim d then mask still optional
    - mu: optional, (p) ndarray
        average value of the gaussian distribution
    - cov: optional, (p, p) ndarray
        covariance matrix
    """

    if len(theta.shape) == 1:
        theta = theta[None, :]

    res = np.zeros(theta.shape[0])

    # setting unvalid values to -infinity
    res[np.any(theta < theta_min, axis=-1)] = -np.inf
    res[np.any(theta > theta_max, axis=-1)] = -np.inf

    cov:  np.ndarray = kwargs.get('cov',  None)
    mu:   np.ndarray = kwargs.get('mu',   None)
    mask: np.ndarray = kwargs.get('mask', None)
    
    # add a gaussian distribution from a covariance matrix
    if (cov is not None) and (mu is not None):

        p = mu.shape[0]
        d = theta.shape[-1]

        # first makes some sanity checks
        assert cov.shape[0] == p, "Incompatible dimensions of the average vector and covariance matrix"
        if (mask is None) and (p == d):
            mask = np.full(d, fill_value=True, dtype=bool)

        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)

        # we perform the multiplicate of (p, p) with (n, p) on axes (1, 1) and then of (p, n) with (p, n) on axes (0, 0) and get back the diagonal value of the resulting (n, n)
        numerator   = - 0.5 * np.diagonal(np.tensordot((theta[:, mask] - mu).T,  np.tensordot(inv_cov, (theta[:, mask] - mu), axes=(1, 1)), axes=(0, 0)))
        denominator = + 0.5 * ( p * np.log(2*np.pi) + np.log(det_cov))
    
        res = res + numerator - denominator

    return res  


def log_likelihood(theta: np.ndarray, 
                   xi: np.ndarray,
                   likelihoods: list[Likelihood],
                   **kwargs) -> np.ndarray:
    """
    Compute the log Likelihood values.

    Parameters
    ----------
    theta : np.ndarray
        Varying parameters.
    xi : np.ndarray
        Extra fixed parameters.
    likelihoods : list[Likelihood]
        The likelihoods to evaluate for the fit.

    Returns
    -------
    np.ndarray
        Values of the log Likelihood for each chain.
    """
    
    res = np.zeros(theta.shape[0])

    need_pk: bool   = False
    uvlf_lkl: UVLFLikelihood | None = None

    if 'matter_power_spectrum_computer' in kwargs:
        for likelihood in likelihoods:
            if isinstance(likelihood, UVLFLikelihood):
                need_pk  = True
                uvlf_lkl = likelihood

    if need_pk is True:

        # first, get the k_max needed
        x: np.ndarray      = uvlf_lkl.get_x(theta, xi)
        k_max: float       = uvlf_lkl.get_k_max(x)

        # second, compute the matter power spectrum from a given function
        k_arr: np.ndarray  = np.logspace(np.log10(uvlf_lkl.k[0]), np.log10(k_max), 50000).T
        func: Callable     = kwargs.get('matter_power_spectrum_computer')
        x_dicts: dict       = uvlf_lkl.get_x_dicts(x)

        pk_arr = np.zeros((x.shape[0], k_arr.shape[1]))

        for i in range(x.shape[0]):
            pk_arr[i] = func(k_arr[i], x_dicts[i])

        kwargs['k']  = k_arr
        kwargs['pk'] = pk_arr
        
    # makes the sum of the log prior and log likelihood 
    for likelihood in likelihoods:
        res = res  + likelihood.loglkl(theta, xi, **kwargs)

    return res


def log_probability(theta:     np.ndarray, 
                    xi:        np.ndarray,
                    theta_min: np.ndarray, 
                    theta_max: np.ndarray,
                    likelihoods: list[Likelihood],
                    **kwargs) -> np.ndarray:
    
    if len(theta.shape) == 1:
        theta = theta[None, :]

    # compute the log prior
    res = log_prior(theta, theta_min, theta_max, **kwargs)
    
    # mask the infinities as we cannot compute the log_likelihood there
    mask = np.isfinite(res)
    res[~mask] = -np.inf

    res[mask] = log_likelihood(theta[mask, :], xi, likelihoods, **kwargs)

    return res 


def initialise_walkers(theta_min: np.ndarray, 
                       theta_max: np.ndarray, 
                       xi       : np.ndarray,
                       likelihoods: list[Likelihood],
                       n_walkers: int = 64, 
                       **kwargs):
    
    i = 0
    n_params = theta_min.shape[0]
    pos      = np.zeros((0, n_params))
    
    while pos.shape[0] < n_walkers and i < 1000 * n_walkers:

        # draw a value for the initial position
        prop = uniform_to_true(np.random.rand(1, n_params), theta_min, theta_max)

        try:
            
            log_lkl = log_likelihood(prop, xi, likelihoods, **kwargs)
        
            # check that the likelihood is finite at that position
            if np.all(np.isfinite(log_lkl)):

                # if finite add it to the list of positions
                pos = np.vstack((pos, prop))

        except:
            pass

        
        i = i+1

    if i >= 1000 * n_walkers:
        warnings.warn("The initialisation hit the safety limit of 1000 * n_walkers to initialise, pos may not be of size n_walkers.\n\
                      Consider reducing the parameter range to one where the likelihood is proportionnaly more defined")
        
    return pos



