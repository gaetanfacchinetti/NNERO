
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
# Definition of cosmological quantities
# Fast computation of the optical depth to reionization
# (heavily use of numpy or torch vectorisation)
# The cosmology is assumed to be flat (omega_k = 0)
#
##################

import numpy as np
import torch

from .constants import CST_EV_M_S_K, CST_NO_DIM, CST_MSOL_MPC, CONVERSIONS


def convert_array(arr: float, to_torch: bool = False) -> (np.ndarray | torch.Tensor):
    if isinstance(arr, np.ndarray | torch.Tensor):
        if (isinstance(arr, np.ndarray) and to_torch is False) or (isinstance(arr, torch.Tensor) and to_torch is True):
            return arr
        if isinstance(arr, np.ndarray) and to_torch is True:
            return torch.tensor(arr)
        if isinstance(arr, torch.Tensor) and to_torch is False:
            return arr.numpy()
    return np.array([arr]) if not to_torch else torch.tensor([arr])

 

#################################
## Simple functions that can be evaluated fast with numpy arrays

# ----------------------------------------------------------
# Densities and abundances

def rho_baryons(omega_b : float | np.ndarray | torch.Tensor) -> (float | np.ndarray | torch.Tensor):
    """
    Baryon energy density (in eV / m^3)

    Parameters
    ----------
    omega_b: float | np.ndarray | torch.Tensor
        reduced abundance of baryons (i.e. times h^2)

    Returns
    -------
    float or np.ndarray or torch.tensor
    """

    return  omega_b * CST_EV_M_S_K.rho_c_over_h2 # in eV / m^3


def n_baryons(omega_b : float | np.ndarray | torch.Tensor) -> (float | np.ndarray | torch.Tensor):
    """
    Baryon number density (in 1 / m^3)

    Parameters
    ----------
    omega_b: float | np.ndarray | torch.Tensor
        reduced abundance of baryons (i.e. times h^2)

    Returns
    -------
        float or np.ndarray or torch.tensor
    """

    return rho_baryons(omega_b) * ( (1.0- CST_NO_DIM.YHe) / CST_EV_M_S_K.mass_hydrogen +  CST_NO_DIM.YHe / CST_EV_M_S_K.mass_helium  ) # in 1/m^3


def n_hydrogen(omega_b : float | np.ndarray | torch.Tensor) -> (float | np.ndarray | torch.Tensor):
    """
    Hydrogen number density (in 1 / m^3)

    Parameters
    ----------
    omega_b: float | np.ndarray | torch.Tensor
        reduced abundance of baryons (i.e. times h^2)

    Returns
    -------
        float or np.ndarray or torch.tensor
    """

    return rho_baryons(omega_b) * (1.0 - CST_NO_DIM.YHe) / CST_EV_M_S_K.mass_hydrogen


def n_ur(m_nus: np.ndarray | torch.Tensor) -> (np.ndarray | torch.Tensor):
    """
    Number of ultra-relativistic degrees of freedom
    
    Parameters
    ----------
    m_nus: np.ndarray | torch.Tensor
        shape (q1, q2, ..., qn, 3), mass of the three neutrinos
        in a given model
    
    Returns
    -------
    np.ndarray or torch.Tensor with shape (q1, q2, ..., qn)
    """

    return CST_NO_DIM.Neff - np.count_nonzero(m_nus, axis=-1)


def omega_r(m_nus: np.ndarray | torch.Tensor) -> (np.ndarray | torch.Tensor):
    """
    Reduced abundance of radiation today
    
    Parameters
    -----------
    m_nus: np.ndarray | torch.Tensor
        shape (q1, q2, ..., qn, 3), mass of the three neutrinos
        in a given model

    Returns
    -------
    np.ndarray or torch.Tensor with shape (q1, q2, ..., qn)
    """
    return  4.48162687719e-7 * CST_EV_M_S_K.T0**4 * (1.0 + 0.227107317660239 * n_ur(m_nus))


def _interp_omega_nu(y):
    """ interpolation function """
    # This interpolation formula is taken from the HYREC-2 code: https://github.com/nanoomlee/HYREC-2
    return (1.+0.317322*0.0457584*y**(3.47446+1.) + 2.05298*0.0457584*y**(3.47446-1.))/(1.+0.0457584*y**(3.47446))*(3.45e-8*(CST_EV_M_S_K.Tnu0**4))*5.6822*2


def omega_nu(z:     float | np.ndarray | torch.Tensor, 
             m_nus: np.ndarray | torch.Tensor
             )-> np.ndarray | torch.Tensor :
    """
    Efficient implementation of reduced neutrino abundance for numpy arrays. 
    Parameters must be floats, arrays or tensor with compatible dimensions. 
    
    Parameters
    ----------
    m_nus: np.ndarray | torch.Tensor
        shape (q1, q2, ..., qn, 3), mass of the three neutrinos
        in a given model
    z: float | numpy.ndarray | torch.Tensor
        shape (p, ) if array, redshift range

    Returns
    -------
    numpy.ndarray or torch.Tensor with shape (q1, ..., qn, p)
    """

    # convert z to an array if it is a float
    z = convert_array(z)

    # a is of shape (1, ..., p, 1)
    p = len(z.flatten())
    a = 1.0/(1.0+z.reshape((*tuple(1 for _ in m_nus.shape[:-1]), p, 1)))

    m_nus = m_nus.reshape((*m_nus.shape[:-1], 1, 3))
    
    # y is of shape (q1, q2, ..., qn, p, 3) 
    y = m_nus/CST_EV_M_S_K.k_Boltz/(CST_EV_M_S_K.Tnu0/a)
    
    # res is of shape (q1, q2, ..., qn, p, 3)
    res = _interp_omega_nu(y)

    # y[..., i] is of shape (q1, ..., qn, p)
    # ensure that we do not sum over 0 entries
    for i in range(3):
        res[y[..., i] == 0, i] = 0

    return np.sum(res, axis=-1)




# ----------------------------------------------------------
# Hubble factor and optical depth


def h_factor_numpy(z:       float | np.ndarray | torch.Tensor,
                   omega_b: float | np.ndarray | torch.Tensor,
                   omega_c: float | np.ndarray | torch.Tensor, 
                   h:       float | np.ndarray | torch.Tensor,
                   m_nus:   np.ndarray | torch.Tensor
                   ) -> (np.ndarray | torch.Tensor):
    """
    E(z) = H(z) / H0 with radiation

    Efficient evalutation of hubble rate parameters for numpy arrays 
    or torch tensors (also works with simple float ). In the first
    case the shape of these numpy arrays must be compatible (see below).
    
    Parameters
    ----------
    z: float | numpy.ndarray | torch.Tensor
        shape (p, ) if array, redshift range
    omega_b: float | numpy.ndarray | torch.Tensor
        shape (q1,..., qn) if array, reduced abundance of baryons (i.e. times h^2)
    omega_c: float | numpy.ndarray | torch.Tensor
        shape (q1,..., qn) if array, reduced abundance of dark matter (i.e. times h^2)
    h: float | numpy.ndarray | torch.Tensor
        shape (q1,..., qn) if array,, Hubble factor
    mnus: numpy.ndarray | torch.Tensor
        shape (q1,..., qn, 3) if array or at least shape (, 3), mass of the neutrinos
       
    Returns
    -------
    numpy.ndarray or torch.Tensor with shape (q1,..., qn, p)
    """

    # convert the input if they are just floats
    omega_b = convert_array(omega_b)
    omega_c = convert_array(omega_c)
    h       = convert_array(h)
    z       = convert_array(z)

    if len(m_nus.shape) == 1:
        m_nus = m_nus[None, :]
                   
    # a is of shape (1, ... p)
    p = len(z.flatten())
    a = 1.0/(1.0+z.reshape((*tuple(1 for _ in h.shape), p)))

    # h is of shape (q1, ..., qn, 1)
    _h = h[..., None]

    # omega values is of shape (q1, ..., qn, 1)
    m_omega_r  = omega_r(m_nus)[..., None]
    m_omega_nu = omega_nu(0, m_nus)
    m_omega_m  = (omega_b + omega_c)[..., None]

    m_omega_l  = (_h**2) - m_omega_m - m_omega_r - m_omega_nu

    # result is of shape (q1, ..., qn, p)
    return np.sqrt(m_omega_m / (a**3) + (m_omega_r + omega_nu(z, m_nus)) / (a**4) + m_omega_l) / _h



def h_factor_no_rad(z:       float | np.ndarray | torch.Tensor, 
                    omega_b: float | np.ndarray | torch.Tensor, 
                    omega_c: float | np.ndarray | torch.Tensor, 
                    h:       float | np.ndarray | torch.Tensor
                    ) -> (np.ndarray | torch.Tensor):
    """
    E(z) = H(z) / H0 without radiation

    Efficient evalutation of hubble rate parameters for numpy arrays 
    or torch tensors (also works with simple float ). In the first
    case the shape of these numpy arrays must be compatible (see below).
    
    Parameters
    ----------
    z: float | numpy.ndarray | torch.Tensor
        shape (p, ) if array, redshift range
    omega_b: float | numpy.ndarray | torch.Tensor
        shape (q1,..., qn) if array, reduced abundance of baryons (i.e. times h^2)
    omega_c: float | numpy.ndarray | torch.Tensor
        shape (q1,..., qn) if array, reduced abundance of dark matter (i.e. times h^2)
    h: float | numpy.ndarray | torch.Tensor
        shape (q1,..., qn) if array, Hubble factor

    Returns
    -------
    numpy.ndarray or torch.Tensor with shape (q1,..., qn, p)
    """

    # convert the input if they are just floats
    omega_b = convert_array(omega_b)
    omega_c = convert_array(omega_c)
    h       = convert_array(h)
    z       = convert_array(z)
                   
    # a is of shape (1, ..., p)
    p = len(z.flatten())
    a = 1.0/(1.0+z.reshape((*tuple(1 for _ in h.shape), p)))

    # h is of shape (q1, ..., qn, 1)
    _h = h[..., None]

    # omega values is of shape (q1, ..., qn, 1)
    m_omega_m  = (omega_b + omega_c)[..., None]
    m_omega_l  = (_h**2) - m_omega_m 

    # result is of shape (q1, ..., qn, p)
    # np.sqrt also works on torch tensors

    xp = np
    if isinstance(z, torch.Tensor):
        xp = torch

    return xp.sqrt(m_omega_m / (a**3)  + m_omega_l) / _h




def optical_depth_no_rad(z:       float | np.ndarray | torch.Tensor, 
                         xHII:    np.ndarray | torch.Tensor,
                         omega_b: float | np.ndarray | torch.Tensor, 
                         omega_c: float | np.ndarray | torch.Tensor, 
                         h:       float | np.ndarray | torch.Tensor,
                         *, 
                         low_value: float = 1.0,
                         with_helium: bool = True,
                         cut_integral_min: bool = True):
    """
    Optical depth to reionization without radiation.

    Efficient evaluation of the opetical depth to reionization (dimensionless)
    uses fast numpy / torch operations with trapezoid rule
    (assume that radiation is neglibible on the range of z). Also neglegts the
    influence of double reionization of helium at small redshifts


    Parameters
    ----------
    z: float | numpy.ndarray | torch.Tensor
        shape (p,) if array, redshift range
    xHII: numpy.ndarray | torch.Tensor
        shape (n, p), ionization fraction vs the redshift for the n models
    omega_b: float | numpy.ndarray | torch.Tensor
        shape (n,) if array, reduced abundance of baryons (i.e. times h^2)
    omega_c: float | numpy.ndarray | torch.Tensor
        shape (n,) if array, reduced abundance of dark matter (i.e. times h^2)
    h: float | numpy.ndarray | torch.Tensor
        shape (n,) if array, Hubble factor
    low_value: float
        value of xHII at redshift smaller than min(z)


    Returns
    -------
    numpy.ndarray or torch.Tensor with shape(n, p)
    """
    
    # define if we work with torch or numpy
    xp = np
    to_torch = False
    if isinstance(xHII, torch.Tensor):
        xp = torch
        to_torch = True

    # convert the floats to arrays or tensors
    z       = convert_array(z, to_torch)
    h       = convert_array(h, to_torch)
    omega_b = convert_array(omega_b, to_torch)
    omega_c = convert_array(omega_c, to_torch)

    if len(z.shape) == 1:
        if isinstance(z, np.ndarray):
            z = z[None, :]
        if isinstance(z, torch.Tensor):
            z = z.unsqueeze(0)

    if len(xHII.shape) == 1:
        if isinstance(xHII, np.ndarray):
            xHII = xHII[None, :]
        if isinstance(xHII, torch.Tensor):
            xHII = xHII.unsqueeze(0)

    # prepare data for redshifts < min(z)
    # we assume that at small z value xHII = 1
    z_small    = xp.linspace(0,  z.min(), 20)[None, :]    
    xHII_small = xp.full((xHII.shape[0], z_small.shape[-1]), fill_value=low_value)

    # concatenate the two parts (small and large z)
    rs = xp.concatenate((z_small, z), axis=-1)
    Xe = xp.concatenate((xHII_small, xHII), axis=-1)

    # add the Helium contribution
    if with_helium is True:
        fHe = CST_NO_DIM.YHe / (1.0 -  CST_NO_DIM.YHe) * CST_EV_M_S_K.mass_hydrogen / CST_EV_M_S_K.mass_helium  
        Xe[:, rs[0, :] <= 3] = Xe[:, rs[0, :] <= 3] + fHe/(1+fHe)
        
    if cut_integral_min is True:
        # find the minimum and cut the integral as it is done in CLASS
        index = xp.argmin(Xe, axis=-1)
        mask = xp.arange(Xe.shape[-1]) >= index[:, None]
        Xe[mask] = 0.0
    
    # fast trapezoid integration scheme
    # h_factor_numpy is of shape (n, p), z of shape (1, p) and xHII of shape (n, p)
    # integrand is of shape (n, p), trapz of shape (n, p-1)
    # res is of shape (n,)
    integrand = Xe * (1+rs)**2 / h_factor_no_rad(rs, omega_b, omega_c, h)
    trapz     = (integrand[..., 1:] + integrand[..., :-1])/2.0
    dz        = xp.diff(rs, axis=-1)
    res       = xp.sum(trapz * dz, axis=-1)

    """
    # fast trapezoid integration scheme (on small z values)
    # h_factor_numpy is of shape (p, n), z_small of shape (1, p) and xHII_small of shape (1, p)
    # integrand_small is of shape (n, p), trapz_small of shape (n, p-1)
    # res is of shape (n,)
    integrand_small = xHII_small * (1+z_small)**2 / h_factor_no_rad(z_small, omega_b, omega_c, h)
    trapz_small     = (integrand_small[..., 1:] + integrand_small[..., :-1])/2.0
    dz_small        = xp.diff(z_small, axis=-1)
    res             = xp.sum(trapz_small * dz_small, axis=-1)

    # fast trapezoid integration scheme (on large z values)
    # h_factor_numpy is of shape (n, p), z of shape (1, p) and xHII of shape (n, p)
    # integrand is of shape (n, p), trapz of shape (n, p-1)
    # res is of shape (n,)
    integrand = xHII * (1+z)**2 / h_factor_no_rad(z, omega_b, omega_c, h)
    trapz     = (integrand[..., 1:] + integrand[..., :-1])/2.0
    dz        = xp.diff(z, axis=-1)
    res       = res + xp.sum(trapz * dz, axis=-1)
    """

    # adding the correct prefactor in front
    # pref is of shape (n,)
    pref = CST_EV_M_S_K.c_light * CST_EV_M_S_K.sigma_Thomson * n_baryons(omega_b) / (100 * h * CONVERSIONS.km_to_mpc)

    return (pref * res).flatten()



#############################################

class ShortPowerSpectrumRange(Exception):
    """
    Exception raised for too short ranges of modes to accurately compute integrals

    Attributes
    ----------
    message: str
        Explanation of the error.

    Parameters
    ----------
    scales: np.ndarray
        Range of modes we consider.
    message: str
        Explanation of the error.
    """

    def __init__(self, scales: np.ndarray, message: str ="Matter power spectrum range is too short") -> None:
        self.message = message
        self.scales  = scales
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message + "\n for scales: " + str(self.scales) + " Mpc"


## Structure formation

def check_ik_R(ik_radius: int, p: int, radius: np.ndarray) -> None:
    """
    Check that the array of mode is long enough for the radius considered.

    Parameters
    ----------
    ik_radius : int
        Index of the array of mode where equal (or closest) to the desired radius.
    p : int
        Length of the array of modes.
    radius : np.ndarray
        Array of radiuses corresponding to the modes.

    Raises
    ------
    ShortPowerSpectrumRange
        If array not long enough.
    """

    # as no way to control the precision of the integral, 
    # require that at least it is computed on enough points


    if not np.all(ik_radius > 0.1*p):
        vals_problem = radius[..., 0][ik_radius <= 0.1*p]
        if np.any(vals_problem == vals_problem): # ignore the nan
            raise ShortPowerSpectrumRange(vals_problem[vals_problem == vals_problem], "The result may be laking precision, consider running CLASS up to lower values of k (p = {})".format(p) ) 
    
    if not np.all(ik_radius < p-1):
        vals_problem = radius[..., 0][ik_radius >= p-1]
        if np.any(vals_problem == vals_problem):
            raise ShortPowerSpectrumRange(vals_problem[vals_problem == vals_problem], "The result may be laking precision, consider running CLASS up to larger values of k (p = {})".format(p)) 


def sigma_r(radius: float | np.ndarray, 
           k: np.ndarray, 
           pk: np.ndarray, 
           *, 
           window: str = 'sharpk', 
           ik_radius : np.ndarray | None = None) -> np.ndarray:
    
    """
    Standard deviation of the matter power spectrum inside `radius`.
    Note that all physical dimensions must be self consistent.
    
    Parameters
    ----------
    radius: float | numpy.ndarray 
        shape (s,) or (q1, ..., qn, r1, ..., rm, s), smoothing scale r in Mpc
    k: numpy.ndarray
        shape (q1, ..., qn, p), modes in Mpc^{-1}
    pk: numpy.ndarray
        shape (q1, ..., qn, p), power spectrum in Mpc^3
    window: str, optional
        smoothing function
    ik_radius: numpy.ndarray, optional
        shape of radius, indices of the k array corresponding to k = 1/radius
        
    Returns
    -------
    numpy.ndarray with shape (q1, ..., qn, r1, ..., rm, s) or (q1, ..., qn, s)
    """

    # make an array out of the input radius if it was not one
    radius = convert_array(radius)

    # dimension of parameters q
    n = len(k.shape) - 1               

    # if only a simple dim 1 array is passed as radius
    # add some other dummy dimensions at least of size n
    # this means however that m = 0
    if len(radius.shape) == 1 :
        radius = radius.reshape((*tuple(1 for _ in range(n)), len(radius)))

    # dimension of parameters r
    m = len(radius.shape) - len(k.shape) 
    
    # put at least shape (1, p) to k and pk
    if len(k.shape) == 1:
        k  = k[None, ...]
        pk = pk[None, ...]

    # change shape of radius to (q1, ..., qn, r1, ..., rm, s, 1)
    radius = radius[..., None]

    # get the last dimension
    p = k.shape[-1]

    # reshape k to the form (q1, ..., qn, 1, ...1, p)
    k = k.reshape(( *(k.shape[:n]), *tuple(1 for _ in range(m)), 1, p) )
    pk = pk.reshape(( *(pk.shape[:n]), *tuple(1 for _ in range(m)), 1, p) )

    # maximum bound of integration
    if window == 'sharpk':
        
        if ik_radius is None:

            ik_radius = np.argmin( (k - 1.0/radius)**2 , axis=-1)
            check_ik_R(ik_radius, p, radius)
 
    else:
        raise ValueError("no other window function that sharpk implemented yet")
    
    mask  = np.where(np.arange(p) < ik_radius[..., None], np.ones((*radius.shape[:-1], p)),  np.zeros((*radius.shape[:-1], p))) # shape (n, r, q, p)
    dlnk  = np.diff(np.log(k), axis=-1)
    
    integ = mask * (k**3) / (2*(np.pi**2)) * pk
    trapz = (integ[..., :-1] + integ[..., 1:])/2.0
    
    return np.sqrt(np.sum(trapz * dlnk, axis = -1))
        



def dsigma_r_dr(radius: float | np.ndarray, 
                k: np.ndarray, 
                pk: np.ndarray, 
                *, 
                window: str = 'sharpk', 
                sigma_radius : np.ndarray | None = None) -> np.ndarray:
    """
    Derivative of the standard deviation of the matter power spectrum 
    inside `radius`. Note that all physical dimensions must be self consistent.
    
    Parameters
    ----------
    radius: float | numpy.ndarray 
        shape (q1, ..., qn, r1, ..., rm, s), smoothing scale r in Mpc
    k: numpy.ndarray
        shape (q1, ..., qn, p), modes in Mpc^{-1}
    pk: numpy.ndarray
        shape (q1, ..., qn, p), power spectrum in Mpc^3
    window: str, optional
        smoothing function
    ik_radius: numpy.ndarray, optional
        shape of radius, indices of the k array corresponding to k = 1/radius
        
    Returns
    -------
    numpy.ndarray with shape (q1, ..., qn, r1, ..., rm, s)
    """

    m = len(radius.shape) - len(k.shape) # dimension of parameters r
    n = len(k.shape) - 1                 # dimension of parameters q
    
    # put at least shape (1, p) to k and pk
    if len(k.shape) == 1:
        k = k[None, ...]
        pk = pk[None, ...]
    
    p = k.shape[-1]

    # make an array out of the input radius if it was not one
    # change shape to (q1, ..., qn, r1, ..., rm, s, 1)
    radius = convert_array(radius)
    _radius = radius[..., None]

    # reshape k to the form (q1, ..., qn, 1, ...1, p) with m+1 1
    _k = k.reshape(( *(k.shape[:n]), *tuple(1 for _ in range(m)), 1, p) )
    
    # reshape pk to the form (q1, ..., qn, 1, ... 1, p) with m 1
    _pk = pk.reshape(( *(pk.shape[:n]), *tuple(1 for _ in range(m)), p) )

    if window == 'sharpk':
        
        # find the index of k corresponding to 1/radius
        ik_radius = np.argmin( (_k - 1.0/_radius)**2 , axis=-1)
        check_ik_R(ik_radius, p, _radius)

        if sigma_radius is None:
            sigma_radius = sigma_r(radius, k, pk, window=window, ik_radius = ik_radius)

        return  - np.take_along_axis(_pk, ik_radius, axis=-1) / radius**4 / sigma_radius / ((2*np.pi)**2)
   
    else:
        raise ValueError("no other window function that sharpk implemented yet")




def sigma_m(mass:float | np.ndarray, 
            k:np.ndarray, 
            pk:np.ndarray, 
            omega_m: float | np.ndarray, 
            *, 
            window: str = 'sharpk', 
            ik_radius: np.ndarray | None = None, 
            c: float = 2.5):
    
    """
    Standard deviation of the matter power spectrum on mass scale.
    Note that all physical dimensions must be self consistent.
    
    Parameters
    ----------
    mass: float | numpy.ndarray
        shape (q1, ..., qn, r1, ..., rm, s) mass scale in Msol
    k: numpy.ndarray
        shape (q1, ..., qn, p), modes in Mpc^{-1}
    pk: numpy.ndarray
        shape (q1, ..., qn, p), power spectrum in Mpc^3
    omega_m: float, np.ndarray 
        shape (q1, ..., qn)
    window: str, optional
        smoothing function
    ik_radius: numpy.ndarray, optional
        shape of radius, indices of the k array corresponding to k = 1/radius
    c: float, optional
        conversion factor for the mass in the sharpk window function

    Returns
    -------
    numpy.ndarray with shape (q1, ..., qn, r1, ..., rm, s)
    """

    mass    = convert_array(mass)
    omega_m = convert_array(omega_m)

    n = len(omega_m.shape)
    m = len(mass.shape) - n -1

    # put at least shape (1, p) to k and pk
    if len(k.shape) == 1:
        k = k[None, ...]
        pk = pk[None, ...]

    assert np.all(omega_m.shape == k.shape[:-1]) and  np.all(omega_m.shape == pk.shape[:-1]), "Dimension of parameters omega_m and k (or pk) should agree"

    # reshape omega_m to the of the output shape (q1, ..., qn, r1, ..., rm, s)
    omega_m = omega_m.reshape( (*(omega_m.shape), *tuple(1 for _ in range(m)), 1) )
    
    # define the averaged (background) matter density today
    rhom0 = omega_m * CST_MSOL_MPC.rho_c_over_h2

    if window == 'sharpk':
        radius = (3*mass/(4*np.pi)/rhom0)**(1/3)/c
    else:
        raise ValueError("no other window function that sharpk implemented yet")
    
    return sigma_r(radius, k, pk, window = window, ik_radius = ik_radius) 


def dsigma_m_dm(mass:float | np.ndarray, 
                k: np.ndarray, 
                pk: np.ndarray, 
                omega_m: float | np.ndarray, 
                *, 
                window: str = 'sharpk', 
                sigma_mass: np.ndarray | None = None, 
                c: float = 2.5):
    """
    Derivative of the standard deviation of the matter power spectrum 
    on mass scale. Note that all physical dimensions must be self consistent.
    
    Parameters
    ----------
    mass: float | numpy.ndarray
        shape (q1, ..., qn, r1, ..., rm, s) mass scale in Msol
    k: numpy.ndarray
        shape (q1, ..., qn, p), modes in Mpc^{-1}
    pk: numpy.ndarray
        shape (q1, ..., qn, p), power spectrum in Mpc^3
    omega_m: float, np.ndarray 
        shape (q1, ..., qn)
    window: str, optional
        smoothing function
    sigma_mass: numpy.ndarray, optional
        shape of mass, value of sigma_m at input mass
    c: float, optional
        conversion factor for the mass in the sharpk window function

    Returns
    -------
    numpy.ndarray with shape (q1, ..., qn, r1, ..., rm, s)
    """
    mass    = convert_array(mass)
    omega_m = convert_array(omega_m)

    n = len(omega_m.shape)
    m = len(mass.shape) - n -1

    # put at least shape (1, p) to k and pk
    if len(k.shape) == 1:
        k = k[None, ...]
        pk = pk[None, ...]

    assert np.all(omega_m.shape == k.shape[:-1]) and  np.all(omega_m.shape == pk.shape[:-1]), "Dimension of parameters omega_m and k (or pk) should agree"

    # reshape omega_m to the of the output shape (q1, ..., qn, r1, ..., rm, s)
    omega_m = omega_m.reshape( (*(omega_m.shape), *tuple(1 for _ in range(m)), 1) )
    
    # define the averaged (background) matter density today
    rhom0 = omega_m * CST_MSOL_MPC.rho_c_over_h2

    if window == 'sharpk':
        radius = (3*mass/(4*np.pi)/rhom0)**(1/3)/c
        drdm   = radius/(3*mass) 
    else:
        raise ValueError("no other window function that sharpk implemented yet")
    
    return dsigma_r_dr(radius, k, pk, window=window, sigma_radius=sigma_mass) * drdm


def growth_function(z: float | np.ndarray, 
                    omega_m: float | np.ndarray, 
                    h: float | np.ndarray):
    """
    Growth function of the linear density contrast

    Analatical fit from Caroll

    Parameters
    ----------
    z: float | numpy.ndarray
        shape (r,) if array, redshift range
    omega_m: float, numpy.ndarray 
        shape (q1, ..., qn), reduced matter abundance
    h: float, numpy.ndarray
        shape (q1, ..., qn), reduced hubble constant

    Returns
    -------
    numpy.ndarray with shape (q1, ..., qn, r)
    """

    z       = convert_array(z)
    h       = convert_array(h)
    omega_m = convert_array(omega_m)
    
    z = z.reshape( (*tuple(1 for _ in omega_m.shape), len(z.flatten())))

    Omega_m   = (omega_m / h**2)[..., None] 
    Omega_l   = 1.0 - Omega_m

    h_factor = (Omega_m * (1+z)**3 + Omega_l)

    Omega_m_z = Omega_m*(1+z)**3 / h_factor
    Omega_l_z = Omega_l / h_factor
    
    return 2.5*Omega_m_z/(Omega_m_z**(4.0/7.0) - Omega_l_z + (1.0 + 0.5*Omega_m_z) * (1.0 + 1.0/70.0*Omega_l_z))/(1+z)
    


def dn_dm(z: float | np.ndarray, 
          mass: float | np.ndarray, 
          k: np.ndarray, 
          pk: np.ndarray, 
          omega_m: float | np.ndarray, 
          h: float | np.ndarray, 
          sheth_a: float = 0.322,
          sheth_q: float = 1.0,
          sheth_p: float = 0.3,
          *, 
          window: str = 'sharpk', 
          c: float = 2.5):
    
    """
    Halo mass function in Msol^{-1} Mpc^{-3}

    Parameters
    ----------
    z: float | numpy.ndarray 
        shape (r,) redshift values 
    mass: float | numpy.ndarray
        shape (s,) or (q, r, s,) mass scale in Msol
    k : numpy.ndarray
        shape (q, p), modes in Mpc^{-1}
    pk : numpy.ndarray
        shape (q, p), power spectrum in Mpc^3
    omega_m: float, np.ndarray 
        shape (q,), reduced matter abundance
    h: float, numpy.ndarray
        shape (q1, ..., qn), reduced hubble constant
    window : str, optional
        smoothing function
    c : float, optional
        conversion factor for the mass in the sharpk window function

    Returns
    -------
    numpy.ndarrray with shape (q, r, s) -- in Msol / Mpc^3
    """


    mass    = convert_array(mass)    # shape (s,) or (q, r, s)
    omega_m = convert_array(omega_m) # shape (q,)
    z       = convert_array(z)       # shape (r,)

    r = len(z)
    s = len(mass)

    if len(mass.shape) == 1:
        
        q = len(omega_m)
        
        _mass = np.empty((q, r, s))
        _mass[:] = mass # shape (q, r, s)

    else:
        
        q = mass.shape[-1]
        _mass = mass


    # define the averaged (background) matter density today
    rhom0 = omega_m[:, None, None] * CST_MSOL_MPC.rho_c_over_h2 # shape (q, 1, 1)
    
    sigma    = sigma_m(_mass, k, pk, omega_m, window=window, c = c) # shape (q, r, s)
    dsigmadm = dsigma_m_dm(_mass, k, pk, omega_m, window=window, sigma_mass=sigma, c = c) # shape (q, r, s)
    growth_z = growth_function(z, omega_m, h)[:, :, None] # shape (q, r, 1)
    growth_0 = growth_function(0, omega_m, h)[:, None] # shape (q, 1, 1)
        
    one_over_sigma = np.zeros_like(sigma)
    mask = ((sigma != 0) & (sigma == sigma))
    one_over_sigma[mask]  = 1.0/sigma[mask]
    one_over_sigma[~mask] = np.nan

    nuhat = np.sqrt(sheth_q) * 1.686 * one_over_sigma * growth_0 / growth_z # shape (q, r, s)

    return -(rhom0/mass) * dsigmadm * one_over_sigma * np.sqrt(2./np.pi)* sheth_a * (1+ nuhat**(-2*sheth_p)) * nuhat * np.exp(-nuhat*nuhat/2.0)
