
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

from .cosmology import convert_array, dn_dm
from .constants import CONVERSIONS


def m_halo(hz:         float | np.ndarray,
           m_uv:       float | np.ndarray, 
           alpha_star: float | np.ndarray, 
           t_star:     float | np.ndarray, 
           f_star10:   float | np.ndarray,
           omega_b:    float | np.ndarray,
           omega_m:    float | np.ndarray) -> tuple[np.ndarray,  np.ndarray]:
    """
    Halo mass in term of the UV magnitude for a given astrophysical model

    Parameters
    ----------
    hz: float, np.ndarray 
        Shape (q, r). Hubble factor given in s^{-1}
    m_uv: float, np.ndarry (s,) or (r, s)
        UV magnitude.
    omega_b: float, np.ndaray (q,)
        Reduced abundance of baryons

    Returns
    -------
    numpy.ndarray with shape (q, r, s)
    """

    hz         = convert_array(hz)
    m_uv       = convert_array(m_uv)
    alpha_star = convert_array(alpha_star)[:, None, None]
    t_star     = convert_array(t_star)[:, None, None]
    f_star10   = convert_array(f_star10)[:, None, None]
    omega_b    = convert_array(omega_b)
    omega_m    = convert_array(omega_m)

    if len(hz.shape) == 1:
        hz = hz[None, :]

    hz = hz[..., None] # shape (q, r, 1)

    if len(m_uv.shape) == 1:
        m_uv = m_uv[None, None, :]

    if len(m_uv.shape) == 2:
        m_uv = m_uv[None, :, :]


    q: int = f_star10.shape[0]
    r: int = hz.shape[1]
    s: int = m_uv.shape[2]

    
    # conversion constant between star formation rate and magnitude
    gamma_UV = 1.15e-28 * 10**(0.4*51.63) / CONVERSIONS.yr_to_s
    
    # fraction of baryons over total matter
    fb = (omega_b/omega_m)[:, None, None]

    # define the return array and fill it with zeros
    res = np.zeros((q, r, s)) # shape (q, r, s)

    # mh_below are those for which f_star10 (m/1e+10) <= 1 (and f_star = f_b f_star10 (m/1e+10))
    # mh_below is of shape (q, r, s)
    mh_below = 1e+10 * ( gamma_UV/(hz*1e+10) * t_star / f_star10 / fb *  10**(-0.4*m_uv) )**(1.0/(alpha_star+1.0))

    # make a difference if f_star10 (m/1e+10) <= 1 or not
    # mask is of shape (q, r, s)
    mask = ((f_star10 * (mh_below/1e+10)**alpha_star) <= 1)

    # vals below are those for which f_star10 (m/1e+10) > 1  (and f_star = f_b)
    # mh_above is of shape (q, r, s)
    mh_above = gamma_UV / hz * t_star / fb *  10**(-0.4*m_uv) 

    # fill the return array with the correct values according to the mask
    res[mask]  = mh_below[mask]
    res[~mask] = mh_above[~mask]

    # check that all mh below statisfy the criterion (as it should)
    if np.any( ( np.tile(f_star10, (1, r, s))[~mask] * (res[~mask]/1e+10)**( np.tile(alpha_star, (1, r, s))[~mask])   ) <= 1):
        raise ValueError("Halo mass value (index below) should satisfy f_star10 * (mh_below/1e+10)**alpha_star < 1")

    # the mask is True when we are below i.e. f_star10 (m/1e+10)^alpha_star < 1, and False when we are above
    return res, mask



def dmhalo_dmuv(hz: float | np.ndarray,
             m_uv: float | np.ndarray, 
             alpha_star: float | np.ndarray, 
             t_star: float | np.ndarray, 
             f_star10: float | np.ndarray,
             omega_b: float | np.ndarray,
             omega_m: float | np.ndarray,
             *,
             mh: np.ndarray | None = None,
             mask: np.ndarray | None = None):
    

    if (mh is None) or (mask is None):
        mh, mask = m_halo(hz, m_uv, alpha_star, t_star, f_star10, omega_b, omega_m) # shape (q, r, s)

    alpha_star = convert_array(alpha_star)
    
    res        = - mh * np.log(10.0) * 0.4/(alpha_star[:, None, None]+1) # if f_star10 (m/1e+10)^alpha_star <= 1
    res[~mask] = - mh[~mask] * np.log(10.0) * 0.4 # if f_star10 (m/1e+10)^alpha_star > 1
    
    return res 
    


def f_duty(mh : float | np.ndarray, m_turn: float | np.ndarray):

    # result of shape (q, r, s)

    mh = convert_array(mh)
    m_turn = convert_array(m_turn)[:, None, None]

    return np.exp(-m_turn/mh)


def phi_uv(z:          float | np.ndarray,
           hz:         float | np.ndarray,
           m_uv:       float | np.ndarray,
           k:          np.ndarray,
           pk:         np.ndarray, 
           alpha_star: float | np.ndarray, 
           t_star:     float | np.ndarray, 
           f_star10:   float | np.ndarray,
           m_turn:     float | np.ndarray,
           omega_b:    float | np.ndarray,
           omega_m:    float | np.ndarray,
           h:          float | np.ndarray,
           sheth_a:    float = 0.322,
           sheth_q:    float = 1.0,
           sheth_p:    float = 0.3,
           *, 
           window: str = 'sharpk', 
           c: float = 2.5,
           mh: np.ndarray | None = None,
           mask: np.ndarray | None = None,
           dndmh: np.ndarray | None = None):
    
    """
    UV flux in Mpc^{-3}

    Parameters
    ----------
    m_uv: float, np.ndarry (s,)
    omega_b: float, np.ndaray (q,)

    Returns
    -------
    result of shape (q, r, s)
    """

    
    # result of shape (n, r, q) in Mpc^(-3)

    if (mh is None) or (mask is None):
        mh, mask = m_halo(hz, m_uv, alpha_star, t_star, f_star10, omega_b, omega_m)               # shape (q, r, s)
    
    dmh_dmuv = dmhalo_dmuv(hz, m_uv, alpha_star, t_star, f_star10, omega_b, omega_m, mh = mh, mask=mask) # shape (q, r, s)
    
    if dndmh is None:
        dndmh = dn_dm(z, mh, k, pk, omega_m, h, sheth_a=sheth_a, sheth_q=sheth_q, sheth_p=sheth_p, window=window, c=c) # shape (q, r, s)

    return f_duty(mh, m_turn) * dndmh * np.abs(dmh_dmuv)

