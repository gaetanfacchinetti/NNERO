
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
# Definition of constants
# Use this method rather than using astropy units for efficiency
#
##################

from types import SimpleNamespace

CST_EV_M_S_K = SimpleNamespace()
CST_MSOL_MPC = SimpleNamespace()
CST_NO_DIM   = SimpleNamespace()
CONVERSIONS  = SimpleNamespace()

CST_EV_M_S_K.mass_hydrogen = 0.93878299831e+9 # in eV
CST_EV_M_S_K.mass_proton   = 0.938272e+9 # in eV
CST_EV_M_S_K.mass_helium   = 3.728400e+9 # in eV
CST_EV_M_S_K.sigma_Thomson = 6.6524616e-29 # in m^2
CST_EV_M_S_K.c_light       = 299792458 # in m / s
CST_EV_M_S_K.k_Boltz       = 8.617343e-5 # Boltzmann constant in eV/K
CST_EV_M_S_K.T0            = 2.7255 # K
CST_EV_M_S_K.Tnu0          = 1.9454 # K
CST_EV_M_S_K.rho_c_over_h2 = 1.0598e+10 # eV / m^3

CST_MSOL_MPC.rho_c_over_h2 = 2.7754e+11 # in units of h^2 M_odot Mpc^{-3}

CST_NO_DIM.YHe  = 0.245
CST_NO_DIM.Neff = 3.044

CONVERSIONS.yr_to_s   = 3600 * 24 * 365.25
CONVERSIONS.km_to_mpc = 3.2407792896393e-20