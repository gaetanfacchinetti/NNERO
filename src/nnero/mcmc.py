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

from scipy       import special
from .data       import uniform_to_true
from .predictor  import predict_tau_from_xHII_numpy, predict_xHII_numpy
from .classifier import Classifier
from .regressor  import Regressor

def log_prior(theta: np.ndarray, 
              theta_min: np.ndarray, 
              theta_max: np.ndarray, 
              **kwargs) -> np.ndarray:
    
    """
        natural logarithm of the prior

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



def log_likelihood(theta: np.ndarray, **kwargs) -> np.ndarray:
    
    if len(theta.shape) == 1:
        theta = theta[None, :]

    classifier: Classifier = kwargs.get('classifier', None)
    regressor:  Regressor  = kwargs.get('regressor',  None)
    use_tau:    bool       = kwargs.get('use_tau', True)
    use_reio:   bool       = kwargs.get('use_reio', True)

    # predict the ionization fraction from the NN
    xHII = predict_xHII_numpy(theta, classifier, regressor)

    # setting the result to -inf when the classifier returns it as a wrong value
    res = np.zeros(theta.shape[0])
    res[xHII[:, 0] == -1] = -np.inf

    # get the value of the ionization fraction at redshift 5.9
    iz = np.argmin(np.abs(regressor.metadata.z - 5.9))
    xHII_59 = xHII[:, iz] 

    if use_tau:

        # get the values in input (if given) or initialise to Planck 2018 results
        tau = kwargs.get('tau', 0.0561)
        var_tau  = kwargs.get('var_tau', 0.0071**2)

        # compute the optical depth to reionization
        tau_pred = predict_tau_from_xHII_numpy(xHII, theta, regressor.metadata)
        res = res - 0.5 * ((tau- tau_pred)**2/var_tau + np.log( 2*np.pi * var_tau))
    
    if use_reio:

         # get the values in input (if given) or initialise to McGreer results
        x_reio      = 0.94
        var_x_reio  = 0.05**2

        # compute the truncated gaussian for the reionization data
        norm_reio = -np.log(1.0 - x_reio + np.sqrt(np.pi/2)*special.erf(x_reio/(np.sqrt(2*var_x_reio))))
        res = res + norm_reio
        mask = xHII_59 < x_reio
        res[mask] = res[mask] - 0.5 * (xHII_59[mask] - x_reio)**2/var_x_reio

    return res 


def log_probability(theta: np.ndarray, 
                    theta_min: np.ndarray, 
                    theta_max: np.ndarray,
                    **kwargs) -> np.ndarray:
    
    # compute the log prior
    res = log_prior(theta, theta_min, theta_max, **kwargs)
    
    # mask the infinities as we cannot compute the log_likelihood there
    mask = np.isfinite(res)
    res[~mask] = -np.inf

    # makes the sum of the log prior and log likelihood 
    res[mask] = res[mask] + log_likelihood(theta[mask, :], **kwargs)

    return res 


def initialise_walkers(theta_min: np.ndarray, 
                       theta_max:np.ndarray, 
                       n_walkers: int = 64, 
                       **kwargs):
    
    i = 0
    n_params = theta_min.shape[0]
    pos      = np.zeros((0, n_params))
    
    while pos.shape[0] < n_walkers and i < 1000 * n_walkers:

        # draw a value for the initial position
        prop = uniform_to_true(np.random.rand(1, n_params), theta_min, theta_max)

        # check that the likelihood is finite at that position
        if np.all(np.isfinite(log_likelihood(prop, **kwargs))):

            # if finite add it to the list of positions
            pos = np.vstack((pos, prop))
        
        i = i+1

    if i >= 1000 * n_walkers:
        warnings.warn("The initialisation hit the safety limit of 1000 * n_walkers to initialise, pos may not be of size n_walkers.\n\
                      Consider reducing the parameter range to one where the likelihood is proportionnaly more defined")
        
    return pos