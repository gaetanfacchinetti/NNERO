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

from .cosmology    import (ShortPowerSpectrumRange, optical_depth_no_rad, h_factor_no_rad, n_baryons, n_hydrogen)
from .constants    import (CST_MSOL_MPC, CST_EV_M_S_K, CST_NO_DIM, CONVERSIONS)
from .astrophysics import (phi_uv)
from .data         import (DataSet, MetaData, DataPartition, latex_labels, uniform_to_true, true_to_uniform)
from .network      import (NeuralNetwork)
from .classifier   import (Classifier, train_classifier)
from .regressor    import (Regressor, train_regressor)
from .interpolator import (Interpolator, train_interpolator)
from .predictor    import (input_values, predict_classifier, predict_xHII, predict_tau, predict_tau_from_xHII,
                         predict_classifier_numpy, predict_xHII_numpy, predict_tau_numpy, predict_tau_from_xHII_numpy,
                         predict_Xe, predict_Xe_numpy, predict_tau_from_Xe, predict_tau_from_Xe_numpy,
                         predict_parameter, predict_parameter_numpy, predict_interpolator, predict_interpolator_numpy)
from .mcmc         import (log_prior, log_likelihood, log_probability, initialise_walkers, Likelihood, UVLFLikelihood)
from .analysis     import (MPChain, Samples, MPSamples, EMCEESamples, GaussianInfo, GaussianSamples, neutrino_masses)

from .analysis import MATPLOTLIB_IMPORTED

if MATPLOTLIB_IMPORTED:
    from .analysis import(AxesGrid, plot_data,  generate_contours, ProcessedData, 
                          compute_quantiles, prepare_data_plot,
                          get_Xe_stats, get_Xe_tanh_stats)

