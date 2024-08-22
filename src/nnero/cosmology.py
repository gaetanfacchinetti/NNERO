
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
# Definition of simple background cosmological quantities
# the cosmology is assumed to be flat (omega_k = 0)
#
##################

import numpy as np
import torch


# define usefull units and conversion rates
_MASS_HYDROGEN_ = 0.93878299831e+9 # in eV
_MASS_PROTON_   = 0.938272e+9 # in eV
_MASS_HELIUM_   = 3.728400e+9 # in eV
_SIGMA_THOMSON_ = 6.6524616e-29 # in m^2
_C_LIGHT_       = 299792458 # in m / s
_K_BOLTZ_       = 8.617343e-5 # Boltzmann constant in eV/K 
_Y_HE_          = 0.245
_T_0_           = 2.7255
_T_NU_0_        = 1.9454
_N_EFF_         = 3.044
_RHO_C_OVER_H2  = 1.0598e+10 # eV / m^3
 

#################################
## Simple functions that can be evaluated fast with numpy arrays

# ----------------------------------------------------------
# Densities and abundances

def rho_baryons(omega_b):
    """
        baryon energy density (in eV / m^3)

    Parameters:
    -----------
    - omega_b
        reduced abundance of baryons today
    """

    return  omega_b * _RHO_C_OVER_H2 # in eV / m^3

def n_baryons(omega_b):
    """
        baryon number density (in 1/m^3)

    Parameters:
    -----------
    - omega_b
        reduced abundance of baryons today
    """
    return rho_baryons(omega_b) / _MASS_PROTON_ / (1 + _Y_HE_ / 4 * (_MASS_HELIUM_/_MASS_HYDROGEN_ -1)) # in 1/m^3

def n_ur(m_nus):
    """
        number of ultra-relativistic degrees of freedom
    
    Parameters:
    -----------
    - m_nus : np.ndarray of shape (n, 3)
        mass of the three neutrinos
    """
    return _N_EFF_ - np.count_nonzero(m_nus, axis=-1)

def omega_r(m_nus):
    """
        reduced abundance of radiation today
    
    Parameters:
    -----------
    - m_nus : np.ndarray of shape (n, 3)
        mass of the three neutrinos
    """
    return  4.48162687719e-7 * _T_0_**4 * (1.0 + 0.227107317660239 * n_ur(m_nus))

def _interp_omega_nu(y):
    # This interpolation formula is taken from the HYREC-2 code: https://github.com/nanoomlee/HYREC-2
    return (1.+0.317322*0.0457584*y**(3.47446+1.) + 2.05298*0.0457584*y**(3.47446-1.))/(1.+0.0457584*y**(3.47446))*(3.45e-8*(_T_NU_0_**4))*5.6822*2

def omega_nu_numpy(z, m_nus):
    """
        efficient implementation of reduced neutrino abundance for numpy arrays
    
    all input parameters must be numpy arrays
    
    Parameters:
    ----------
    - m_nus: shape (n, 3)
        neutrino masses
    - z must be of shape (p, )
        redshift range

    Returns:
    -------
    - reduced neutrino abundance with shape (n, p)
    """

    # a is of shape (p, 1, 1)
    p = len(z.flatten())
    a = 1.0/(1.0+z.reshape(p, 1, 1))
    
    # y is of shape (p, n, 3) 
    y = m_nus/_K_BOLTZ_/(_T_NU_0_/a)
    
    # res is of shape (p, n, 3)
    res = _interp_omega_nu(y)

    # y[:, :, i] is of shape (p, n)
    # ensure that we do not sum over 0 entries
    for i in range(3):
        res[y[:, :, i] == 0, i] = 0

    return np.sum(res, axis=-1).T

# ----------------------------------------------------------
# Hubble factor and optical depth

def h_factor_numpy(z : np.ndarray, omega_b : np.ndarray, omega_c : np.ndarray, h : np.ndarray, m_nus : np.ndarray):
    """
        efficient evalutation of hubble rate parameters for numpy arrays

    all input parameters must be numpy arrays
    
    Parameters:
    ----------
    - omega_b, omega_c: shape (n, )
        abundance of baryons and dark matter
    - m_nus: shape (n, 3)
        neutrino masses
    - z must be of shape (p, )
        redshift range

    Returns:
    -------
    - hubble factor with shape (n, p)
    """
                   
    # a is of shape (1, p)
    p = len(z.flatten())
    a = 1.0/(1.0+z.reshape(1, p))

    # h is of shape (n, 1)
    _h = h[:, None]

    # omega values is of shape (n, 1)
    m_omega_r  = omega_r(m_nus)[:, None]
    m_omega_nu = omega_nu_numpy(np.array([0]), m_nus)
    m_omega_m  = (omega_b + omega_c)[:, None]
    m_omega_l  = (_h**2) - m_omega_m - m_omega_r - m_omega_nu


    # result is of shape (p, n)
    return np.sqrt(m_omega_m / (a**3) + (m_omega_r + omega_nu_numpy(z, m_nus)) / (a**4) + m_omega_l) / _h



def h_factor_no_rad_numpy(z : np.ndarray, omega_b : np.ndarray, omega_c : np.ndarray, h : np.ndarray):
    """
        efficient evalutation of hubble rate parameters for numpy arrays

    all input parameters must be numpy arrays
    
    Parameters:
    ----------
    - omega_b, omega_c: shape (n, )
        abundance of baryons and dark matter
    - m_nus: shape (n, 3)
        neutrino masses
    - z must be of shape (p, )
        redshift range

    Returns:
    -------
    - hubble factor with shape (n, p)
    """
                   
    # a is of shape (1, p)
    p = len(z.flatten())
    a = 1.0/(1.0+z.reshape(1, p))

    # h is of shape (n, 1)
    _h = h[:, None]

    # omega values is of shape (n, 1)
    m_omega_m  = (omega_b + omega_c)[:, None]
    m_omega_l  = (_h**2) - m_omega_m 

    # result is of shape (p, n)
    return np.sqrt(m_omega_m / (a**3)  + m_omega_l) / _h



def optical_depth_numpy(z, xHII, omega_b, omega_c, h, m_nus : np.ndarray, low_value = 1.0):
    """
        efficient evaluation of the opetical depth to reionization
        optical depth to reionization (dimensionless)
        uses fast numpy operations with trapezoid rule

    Parameters:
    -----------
    - z: shape (p, )
        redshift range
    - xHII: shape (n, p)


    Returns:
    --------
    - optical depth for each value of z shape (n,)
    """

    _z = z[None, :]
    
    # prepare data for redshifts < min(z)
    # we assume that at small z value xHII = 1
    z_small    = np.linspace(0,  np.min(z), 20)[None, :]    
    xHII_small = np.full((1, len(z_small)), fill_value=low_value) 
    
    # fast trapezoid integration scheme (on small z values)
    # h_factor_numpy is of shape (p, n), z_small of shape (1, p) and xHII_small of shape (1, p)
    # integrand_small is of shape (n, p), trapz_small of shape (n, p-1)
    # res is of shape (n,)
    integrand_small = xHII_small * (1+z_small)**2 / h_factor_numpy(z_small, omega_b, omega_c, h, m_nus)
    trapz_small     = (integrand_small[..., 1:] + integrand_small[..., :-1])/2.0
    dz_small        = np.diff(z_small, axis=-1)
    res             = np.sum(trapz_small * dz_small, axis=-1)

    # fast trapezoid integration scheme (on large z values)
    # h_factor_numpy is of shape (p, n), z of shape (1, p) and xHII of shape (n, p)
    # integrand is of shape (n, p), trapz of shape (n, p-1)
    # res is of shape (n,)
    integrand = xHII * (1+_z)**2 / h_factor_numpy(_z, omega_b, omega_c, h, m_nus)
    trapz     = (integrand[..., 1:] + integrand[..., :-1])/2.0
    dz        = np.diff(_z, axis=-1)
    res       = res + np.sum(trapz * dz, axis=-1)
    
    # adding the correct prefactor in front
    pref = _C_LIGHT_ * _SIGMA_THOMSON_ * n_baryons(omega_b) / (h * 3.2407792896393e-18)
    
    return pref * res




def optical_depth_no_rad_numpy(z, xHII, omega_b, omega_c, h, low_value = 1.0):
    """
        efficient evaluation of the opetical depth to reionization
        optical depth to reionization (dimensionless)
        uses fast numpy operations with trapezoid rule
        (assume that radiation is neglibible on the range of z)

    Returns:
    --------
    optical depth for each value of z
    """

    _z = z[None, :]

    # prepare data for redshifts < min(z)
    # we assume that at small z value xHII = 1
    z_small    = np.linspace(0,  np.min(z), 20)[None, :]    
    xHII_small = np.full((1, len(z_small)), fill_value=low_value)[None, :] 

    # fast trapezoid integration scheme (on small z values)
    # h_factor_numpy is of shape (p, n), z_small of shape (1, p) and xHII_small of shape (1, p)
    # integrand_small is of shape (n, p), trapz_small of shape (n, p-1)
    # res is of shape (n,)
    integrand_small = xHII_small * (1+z_small)**2 / h_factor_no_rad_numpy(z_small, omega_b, omega_c, h)
    trapz_small     = (integrand_small[..., 1:] + integrand_small[..., :-1])/2.0
    dz_small        = np.diff(z_small, axis=-1)
    res             = np.sum(trapz_small * dz_small, axis=-1)

    # fast trapezoid integration scheme (on large z values)
    # h_factor_numpy is of shape (p, n), z of shape (1, p) and xHII of shape (n, p)
    # integrand is of shape (n, p), trapz of shape (n, p-1)
    # res is of shape (n,)
    integrand = xHII * (1+_z)**2 / h_factor_no_rad_numpy(_z, omega_b, omega_c, h)
    trapz     = (integrand[..., 1:] + integrand[..., :-1])/2.0
    dz        = np.diff(_z, axis=-1)
    res       = res + np.sum(trapz * dz, axis=-1)

    # adding the correct prefactor in front
    # pref is of shape (n,)
    pref = _C_LIGHT_ * _SIGMA_THOMSON_ * n_baryons(omega_b) / (h * 3.2407792896393e-18)

    return (pref * res).flatten()




def optical_depth_no_rad_torch(z, xHII, omega_b, omega_c, h, low_value = 1.0):
    """
        efficient evaluation of the opetical depth to reionization
        optical depth to reionization (dimensionless)
        uses fast numpy operations with trapezoid rule
        (assume that radiation is neglibible on the range of z)

    Returns:
    --------
    optical depth for each value of z
    """

    _z = z[None, :]

    # prepare data for redshifts < min(z)
    # we assume that at small z value xHII = 1
    z_small    = torch.linspace(0,  torch.min(z), 20)[None, :]    
    xHII_small = torch.full((1, len(z_small)), fill_value=low_value)

    # fast trapezoid integration scheme (on small z values)
    # h_factor_numpy is of shape (p, n), z_small of shape (1, p) and xHII_small of shape (1, p)
    # integrand_small is of shape (n, p), trapz_small of shape (n, p-1)
    # res is of shape (n,)
    integrand_small = xHII_small * (1+z_small)**2 / h_factor_no_rad_numpy(z_small, omega_b, omega_c, h)
    trapz_small     = (integrand_small[..., 1:] + integrand_small[..., :-1])/2.0
    dz_small        = torch.diff(z_small, axis=-1)
    res             = torch.sum(trapz_small * dz_small, axis=-1)

    # fast trapezoid integration scheme (on large z values)
    # h_factor_numpy is of shape (p, n), z of shape (1, p) and xHII of shape (n, p)
    # integrand is of shape (n, p), trapz of shape (n, p-1)
    # res is of shape (n,)
    integrand = xHII * (1+_z)**2 / h_factor_no_rad_numpy(_z, omega_b, omega_c, h)
    trapz     = (integrand[..., 1:] + integrand[..., :-1])/2.0
    dz        = torch.diff(_z, axis=-1)
    res       = res + torch.sum(trapz * dz, axis=-1)

    # adding the correct prefactor in front
    # pref is of shape (n,)
    pref = _C_LIGHT_ * _SIGMA_THOMSON_ * n_baryons(omega_b) / (h * 3.2407792896393e-18)

    return pref * res

#############################################



#############################################
# User friendy Cosmology class

class ParamsDefault:

    def __init__(self, new_params:dict = None, **kwargs) -> None:

        # either pass the input as a dictionnary or through kwargs
        if new_params is not None:
            new_params = (new_params | kwargs)
        else:
            new_params = kwargs

        self.create_params(new_params)
        
        for key, value in self._params.items():
            self.__dict__[key] = value


    # define the params dictionary
    def create_params(self, new_params: dict) -> None:
        self._params = self._defaults.copy()

        if new_params is None:
            return None

        for key, value in new_params.items():
            # can only set parameters already present in the default dictionnary
            if key in self._defaults:
                self._params[key] = value
            else:
                raise ValueError("Trying to initialise parameter " + key + " that is not in the default list")

    # modify the __setattr__ to prevent any unwanted modifications
    def __setattr__(self, name, value):

        self.__dict__[name] = value

        if name in self._defaults:
            raise Exception("Attributes are read only! Use update() to modify them.")

    # reinitialise to the default values
    def set_defaults(self):

         for key, value in self._defaults.items():
            if key in self._defaults:
                self._params[key] = value
                self.__dict__[key] = value


    def update(self, **new_params):
        """ update parameters """

        for key, value in new_params.items():
            if key in self._defaults:
                self._params[key] = value
                self.__dict__[key] = value
            else:
                raise ValueError("Trying to modify a parameter not in default")
    
    def __str__(self):
        return "NNERO object with parameters: " + str(self._params)
    
    def __call__(self):
        return self._params
    

## Define a simple cosmology class
class Cosmology(ParamsDefault):
    
    def __init__(self, new_params : dict = None, **kwargs) -> None:
        
        self._defaults = {
        "h"       : 0.6735837, 
        "omega_b" : 0.02242,
        "omega_c" : 0.11933,
        "mnu1"    : 0.06,
        "mnu2"    : 0.0,
        "mnu3"    : 0.0}

        super().__init__(new_params, **kwargs)


    @property
    def m_nus(self):
        return np.array([self.mnu1, self.mnu2, self.mnu3])
    
    @property
    def n_ur(self):
        return self.Neff - np.count_nonzero(self.m_nus) 

    @property
    def omega_r(self):
        return omega_r(cosmo.m_nus)
    
    def omega_nu(self, z):
        
        res = 0
        a = 1.0/(1.0+z)
        
        for mnu in self.m_nus:  
            if mnu > 0:
                y = mnu/_K_BOLTZ_/(self.Tnu0/a)
                # This interpolation formula was taken from  the HYREC-2 code: https://github.com/nanoomlee/HYREC-2
                res = res + (1.+0.317322*0.0457584*y**(3.47446+1.) + 2.05298*0.0457584*y**(3.47446-1.))/(1.+0.0457584*y**(3.47446))*(3.45e-8*(self.Tnu0**4))*5.6822*2 

        return res
    
    @property
    def omega_l(self):
        return self.h^2 - self.omega_c - self.omega_b - self.omega_r - self.omega_nu(0)

    @property
    def rho_b(self):
        return rho_baryons(self.omega_b)

    @property
    def n_b(self):
        return self.rho_b / _MASS_PROTON_ / (1 + _Y_HE_ / 4 * (_MASS_HELIUM_/_MASS_HYDROGEN_ -1)) # in 1/m^3




def hubble_factor(z, cosmo=Cosmology()):
    """
        hubble factor H(z)/H_0 (dimensionless)

    Parameters:
    -----------
    - z: np.ndarray, float
        redshift
    - cosmo: Cosmology, optional
        cosmology

    Returns:
    --------
    hubble factor H(z)/H0
    """

    a = 1.0/(1.0+z)
    return np.sqrt((cosmo.omega_b + cosmo.omega_c) / (a**3) + (cosmo.omega_r + cosmo.omega_nu(z)) / (a**4) + cosmo.omega_l) / cosmo.h



def hubble_rate(z, cosmo = Cosmology()):
    """
        hubble rate in s^{-1} directly computed from the C-code

    Parameters:
    -----------
    - z: np.ndarray, float
        redshift
    - cosmo: Cosmology, optional
        cosmology

    Returns:
    --------
    hubble rate in s^{-1}
    """
 
    return hubble_factor(z, cosmo) * 3.2407792896393e-18 * cosmo.h




def optical_depth(z, xHII, cosmo = Cosmology(), low_value = 1.0):
    """
        efficient evaluation of the opetical depth to reionization
        optical depth to reionization (dimensionless)
        uses fast numpy operations with trapezoid rule

    Parameters:
    -----------
    - z: np.ndarray
        redshift
    - xHII: np.ndarray
        free electron fraction
    - cosmo: Cosmology, optional
        cosmology
    - low_value: float
        value of xHII for redshift below the minimum redshift
        in the array z

    Returns:
    --------
    optical depth for each value of z
    """
    
    # prepare data for redshifts < min(z)
    z_small    = np.linspace(0,  np.min(z), 20)              # small z value
    xHII_small = np.full(len(z_small), fill_value=low_value) # small z xHII values (we assume that at small value it's 1)
    
    # fast trapezoid integration scheme (on small z values)
    integrand_small = xHII_small * (1+z_small)**2 / hubble_factor(z_small, cosmo)
    trapz_small     = (integrand_small[..., 1:] + integrand_small[..., :-1])/2.0
    dz_small        = np.diff(z_small)
    res             = np.sum(trapz_small * dz_small)

    # fast trapezoid integration scheme (on large z values)
    integrand = xHII * (1+z)**2 / hubble_factor(z, cosmo)
    trapz     = (integrand[..., 1:] + integrand[..., :-1])/2.0
    dz        = np.diff(z)
    res       = res + np.sum(trapz * dz, axis=-1)
    
    pref = _C_LIGHT_ * _SIGMA_THOMSON_ * n_baryons(cosmo) / (cosmo.h * 3.2407792896393e-18)
    return pref * res


#############################################
# Additional useful functions
  
def z_centre_reio(z, xHII):
    return z[np.argmin((xHII - 0.5)**2, axis=-1)]
