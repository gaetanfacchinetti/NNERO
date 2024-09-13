
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

from .cosmology import convert_array, h_factor_no_rad, dn_dm
from .constants import CONVERTIONS


def m_halo(z:          float | np.ndarray,
           m_uv:       float | np.ndarray, 
           alpha_star: float | np.ndarray, 
           t_star:     float | np.ndarray, 
           f_star10:   float | np.ndarray,
           omega_b:    float | np.ndarray,
           omega_c:    float | np.ndarray,
           h:          float | np.ndarray):
    """
    Halo mass in term of the UV magnitude for a given astrophysical model

    Parameters:
    ----------
    z: float, np.ndarray (r,)
    m_uv: float, np.ndarry (s,) or (r, s)
    omega_b: float, np.ndaray (q,)

    Returns:
    -------
    result of shape (q, r, s)
    """

    z          = convert_array(z)
    m_uv       = convert_array(m_uv)
    alpha_star = convert_array(alpha_star)[:, None, None]
    t_star     = convert_array(t_star)[:, None, None]
    f_star10   = convert_array(f_star10)[:, None, None]
    omega_b    = convert_array(omega_b)
    omega_c    = convert_array(omega_c)
    h          = convert_array(h)

    if len(m_uv.shape) == 1:
        m_uv = m_uv[None, None, :]

    if len(m_uv.shape) == 2:
        m_uv = m_uv[None, :, :]


    gamma_UV = 1.15e-28 * 10**(0.4*51.63) / CONVERTIONS.yr_to_s
    hz       = (h_factor_no_rad(z, omega_b, omega_c, h) * CONVERTIONS.km_to_mpc)[:, :, None] # shape (q, r, 1)
    
    fb       = (omega_b/(omega_b+omega_c))[:, None, None]

    return 1e+10 * ( gamma_UV/(hz*1e+10) * t_star / f_star10 / fb *  10**(-0.4*m_uv) )**(1.0/(alpha_star+1.0))



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
        mh = m_halo(z, m_uv, alpha_star, t_star, f_star10, omega_b, omega_c, h) # shape (q, r, s)

    alpha_star = convert_array(alpha_star)
    return - mh * np.log(10.0) * 0.4/(alpha_star[:, None, None]+1)
    


def f_duty(mh : float | np.ndarray, m_turn: float | np.ndarray):

    # result of shape (q, r, s)

    mh = convert_array(mh)
    m_turn = convert_array(m_turn)[:, None, None]

    return np.exp(-m_turn/mh)


def phi_uv(z:          float | np.ndarray,
           m_uv:       float | np.ndarray,
           k:          np.ndarray,
           pk:         np.ndarray, 
           alpha_star: float | np.ndarray, 
           t_star:     float | np.ndarray, 
           f_star10:   float | np.ndarray,
           m_turn:     float | np.ndarray,
           omega_b:    float | np.ndarray,
           omega_c:    float | np.ndarray,
           h:          float | np.ndarray,
           sheth_a:    float = 0.322,
           sheth_q:    float = 1.0,
           sheth_p:    float = 0.3,
           *, 
           window: str = 'sharpk', 
           c: float = 2.5):
    
    """
    UV flux in Mpc^{-3}

    Parameters:
    ----------
    m_uv: float, np.ndarry (s,)
    omega_b: float, np.ndaray (q,)

    Returns:
    -------
    result of shape (q, r, s)
    """

    
    # result of shape (n, r, q) in Mpc^(-3)
    
    mh       = m_halo(z, m_uv, alpha_star, t_star, f_star10, omega_b, omega_c, h)               # shape (q, r, s)
    dmh_dmuv = dmhalo_dmuv(z, m_uv, alpha_star, t_star, f_star10, omega_b, omega_c, h, mh = mh) # shape (q, r, s)
    dndmh    = dn_dm(z, mh, k, pk, omega_b+omega_c, h, sheth_a=sheth_a, sheth_q=sheth_q, sheth_p=sheth_p, window=window, c = c) # shape (q, r, s)

    return f_duty(mh, m_turn) * dndmh * np.abs(dmh_dmuv)