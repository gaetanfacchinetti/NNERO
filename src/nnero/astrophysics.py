
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
# Definition of simple astrophysical quantities
# for the computation of UV-luminosity functions
#
##################

import numpy as np
import warnings

_YR_TO_S_  = 3600 * 24 * 365.25
_KM_TO_MPC_ = 3.2407792896393e-18

def convert_numpy(arr: float):
    if isinstance(arr, np.ndarray):
        return arr
    return np.array([arr])


def check_ik_R(ik_R, p):

    # as no way to control the precision of the integral, 
    # require that at least it is computed on enough points
    # ask for at least 20 points
    
    if not np.all(ik_R > 0.1*p):
        warnings.warn("The result may be laking precision, consider running CLASS up to lower values of k") 
    
    if not np.all(ik_R < 0.9*p):
        warnings.warn("The result may be laking precision, consider running CLASS up to larger values of k") 


def sigmaR(radius: float | np.ndarray, k: np.ndarray, pk: np.ndarray, *, 
           window: str = 'sharpk', ik_R : np.ndarray | None = None):
    
    """
    standard deviation of the matter power spectrum inside `radius`
    note that all physical dimensions must be self consistent
    
    Parameters
    ----------
    - radius: float, np.ndarray with shape (q,) or (n, q) or (n, r, q)
        scale on which we compute the

    Returns
    -------
    - result of shape (n, r, q)
    """

    # make an array out of the input R if it was not one
    radius = convert_numpy(radius)

    if len(radius.shape) == 1: # then we gave an array of length q
        radius = radius[None, None, :]

    if len(radius.shape) == 2: # then we gave an array of shape (n, q)
        radius = radius[:, None, :]
    
    # dimension of the arrays
    n = radius.shape[0]
    r = radius.shape[1]
    q = radius.shape[2]
    p = len(k)

    # maximum bound of integration
    if window == 'sharpk':
        
        if ik_R is None:

            ik_R = np.argmin( (k[None, None, None, :] - 1.0/radius[..., None])**2 , axis=-1) # shape of ik_max (n, r, q,)
            check_ik_R(ik_R, p)
 
    else:
        raise ValueError("no other window function that sharpk implemented yet")
    
    mask  = np.where(np.arange(p) < ik_R[:, :, :, None], np.ones((n, r, q, p)),  np.zeros((n, r, q, p))) # shape (n, r, q, p)
    dlnk  = np.diff(np.log(k), axis=-1)
    
    integ = mask * (k[None, None, None, :]**3) / (2*(np.pi**2)) * pk[None, None, None, :]
    trapz = (integ[..., :-1] + integ[..., 1:])/2.0
    
    return np.sqrt(np.sum(trapz * dlnk, axis = -1))
        

def dsigmaRdR(radius: float | np.ndarray, k: np.ndarray, pk: np.ndarray, 
              *, window: str = 'sharpk', sigma_R : np.ndarray | None = None):
    
    # make an array out of the input R if it was not one
    radius = convert_numpy(radius)

    if len(radius.shape) == 1:
        radius = radius[None, None, :]

    if len(radius.shape) == 2:
        radius = radius[:, None, :]

    if window == 'sharpk':
        
        # find the index of k corresponding to 1/R
        ik_R = np.argmin( (k[None, None, None, :] - 1.0/radius[:, :, :, None])**2 , axis=-1)
        check_ik_R(ik_R, len(k))

        if sigma_R is None:
            sigma_R = sigmaR(radius, k, pk, window=window, ik_R = ik_R)
        
        return  - pk[ik_R] / radius**4 / sigma_R / ((2*np.pi)**2)
   
    else:
        raise ValueError("no other window function that sharpk implemented yet")


_RHO_C_H2_MSUN_MPC3_ = 2.7754e+11 # in units of h^2 M_odot Mpc^{-3}

def sigmaM(M:float | np.ndarray, k:np.ndarray, pk:np.ndarray, omega_m: float | np.ndarray, *, window: str = 'sharpk', ik_R: np.ndarray | None = None, c: float = 2.5):
    
    """
    standard deviation of the matter power spectrum on mass scale M
    note that all physical dimensions must be self consistent
    
    Parameters
    ----------
    - M: float, np.ndarray (q,) or (n, r, q)
        scale on which we compute the
    - omega_m: float, np.ndarray (n,)

    Returns
    -------
    result of shape (n, r, q)
    """

    M       = convert_numpy(M)
    omega_m = convert_numpy(omega_m)[:, None, None]
 
    if len(M.shape) == 1:
        M = M[None, None, :]
    
    rhom0 = omega_m * _RHO_C_H2_MSUN_MPC3_

    if window == 'sharpk':
        radius = (3*M/(4*np.pi)/rhom0)**(1/3)/c
    else:
        raise ValueError("no other window function that sharpk implemented yet")
    
    return sigmaR(radius, k, pk, window = window, ik_R = ik_R) 


def dsigmaMdM(M:float | np.ndarray, k:np.ndarray, pk:np.ndarray, omega_m: float | np.ndarray, *, window: str = 'sharpk', sigma_M: np.ndarray | None = None, c: float = 2.5):

    M       = convert_numpy(M)
    omega_m = convert_numpy(omega_m)[:, None, None]

    rhom0 = omega_m * _RHO_C_H2_MSUN_MPC3_

    if len(M.shape) == 1:
        M = M[None, None, :]

    if window == 'sharpk':
        radius = (3*M/(4*np.pi)/rhom0)**(1/3)/c
        dRdM   = radius/(3*M) 
    else:
        raise ValueError("no other window function that sharpk implemented yet")
    
    return dsigmaRdR(radius, k, pk, window=window, sigma_R=sigma_M) * dRdM


def growth_function(z: float | np.ndarray, omega_m: float | np.ndarray, h: float | np.ndarray):

    z       = convert_numpy(z)
    h       = convert_numpy(h)
    omega_m = convert_numpy(omega_m)
    
    
    Omega_m   = (omega_m / h**2)[:, None] 
    Omega_l   = 1.0 - Omega_m

    h_factor = (Omega_m * (1+z)**3 + Omega_l)

    Omega_m_z = Omega_m*(1+z)**3 / h_factor
    Omega_l_z = Omega_l / h_factor
    
    z = z[None, :]
    
    return 2.5*Omega_m_z/(Omega_m_z**(4.0/7.0) - Omega_l_z + (1.0 + 0.5*Omega_m_z) * (1.0 + 1.0/70.0*Omega_l_z))/(1+z)
    


def dndM(z: float | np.ndarray, M: float | np.ndarray, k: np.ndarray, pk: np.ndarray, omega_m: float | np.ndarray, h: float | np.ndarray, *, window: str = 'sharpk', c: float = 2.5, SHETH_A = 0.322, SHETH_q = 1.0, SHETH_p = 0.3):
    
    """
        halo mass function

    Parameters
    ----------
    - z: float, np.ndarray (r, )
    - M: float, np.ndarray (q, ) or (n, r, q)
    - omega_m : float, np.ndarray (n, )

    Return
    ------
    - res of shape (n, r, q) in Msol / Mpc^3
    """

    M       = convert_numpy(M)
    omega_m = convert_numpy(omega_m)
    
    rhom0 = (omega_m * _RHO_C_H2_MSUN_MPC3_)[:, None, None] # shape (n, 1, 1)

    sigma    = sigmaM(M, k, pk, omega_m, window=window, c = c) # shape (n, r, q)
    dsigmadm = dsigmaMdM(M, k, pk, omega_m, window=window, sigma_M=sigma, c = c) # shape (n, r, q)
    growth_z = growth_function(z, omega_m, h)[:, :, None] # shape (n, r, 1)
    growth_0 = growth_function(0, omega_m, h)[:, None] # shape (n, 1, 1)
    nuhat    =  np.sqrt(SHETH_q) * 1.686 / sigma * growth_0 / growth_z
    
    return -(rhom0/M) * (dsigmadm/sigma) * np.sqrt(2./np.pi)* SHETH_A * (1+ nuhat**(-2*SHETH_p)) * nuhat * np.exp(-nuhat*nuhat/2.0)


def m_halo(z: float | np.ndarray,
           m_uv: float | np.ndarray, 
           alpha_star: float | np.ndarray, 
           t_star: float | np.ndarray, 
           f_star10: float | np.ndarray,
           omega_b: float | np.ndarray,
           omega_c: float | np.ndarray,
           h: float | np.ndarray):
    """
        halo mass in term of the UV magnitude for a given astrophysical model

    Parameters:
    ----------
    z: float, np.ndarray (r,)
    m_uv: float, np.ndarry (q,)
    omega_b: float, np.ndaray (n,)

    Returns:
    -------
    result of shape (n, r, q)
    """

    z          = convert_numpy(z)
    m_uv       = convert_numpy(m_uv)
    alpha_star = convert_numpy(alpha_star)
    t_star     = convert_numpy(t_star)
    f_star10   = convert_numpy(f_star10)
    omega_b    = convert_numpy(omega_b)
    omega_c    = convert_numpy(omega_c)
    h          = convert_numpy(h)

    gamma_UV = 1.15e-28 * 10**(0.4*51.63) / _YR_TO_S_
    hz       = (nnero.cosmology.h_factor_no_rad_numpy(z, omega_b, omega_c, h) * _KM_TO_MPC_)[:, :, None] # shape (n, r)
    fb       = omega_b/(omega_b+omega_c)

    return 1e+10 * ( gamma_UV/(hz*1e+10) * t_star[:, None, None] / f_star10[:, None, None] / fb[:, None, None] *  10**(-0.4*m_uv[None, None, :]) )**(1.0/(alpha_star[:, None, None]+1.0))

def dmhalo_dmuv(z: float | np.ndarray,
             m_uv: float | np.ndarray, 
             alpha_star: float | np.ndarray, 
             t_star: float | np.ndarray, 
             f_star10: float | np.ndarray,
             omega_b: float | np.ndarray,
             omega_c: float | np.ndarray,
             h: float | np.ndarray,
             *,
             mh: np.ndarray | None = None):
    

    if mh is None:
        mh = m_halo(z, m_uv, alpha_star, t_star, f_star10, omega_b, omega_c, h) # shape (n, r, q)

    alpha_star = convert_numpy(alpha_star)
    return - mh * np.log(10.0) * 0.4/(alpha_star[:, None, None]+1)
    


def f_duty(mh : float | np.ndarray, m_turn: float | np.ndarray):

    # result of shape (n, r, q)

    mh = convert_numpy(mh)
    m_turn = convert_numpy(m_turn)[:, None, None]

    return np.exp(-m_turn/mh)



def phi_uv(z: float | np.ndarray,
             m_uv: float | np.ndarray, 
             alpha_star: float | np.ndarray, 
             t_star: float | np.ndarray, 
             f_star10: float | np.ndarray,
             m_turn: float | np.ndarray,
             omega_b: float | np.ndarray,
             omega_c: float | np.ndarray,
             h: float | np.ndarray,
             *, 
             window: str = 'sharpk', 
             c: float = 2.5, 
             SHETH_A = 0.322, 
             SHETH_q = 1.0, 
             SHETH_p = 0.3):
    
    # result of shape (n, r, q) in Mpc^(-3)
    
    mh       = m_halo(z, m_uv, alpha_star, t_star, f_star10, omega_b, omega_c, h)
    dmh_dmuv = dmhalo_dmuv(z, m_uv, alpha_star, t_star, f_star10, omega_b, omega_c, h, mh = mh)
    dndmh    = dndM(z, mh, k, pk, omega_b+omega_c, h, window=window, c = c, SHETH_A=SHETH_A, SHETH_q=SHETH_q, SHETH_p=SHETH_p)

    return f_duty(mh, m_turn) * dndmh * np.abs(dmh_dmuv)