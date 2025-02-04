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
import torch

from .data         import MetaData, true_to_uniform
from .cosmology    import optical_depth_no_rad
from .classifier   import Classifier
from .regressor    import Regressor
from .interpolator import Interpolator
from .data         import MP_KEY_CORRESPONDANCE


DEFAULT_VALUES = {'F_STAR10' : -1.5, 'ALPHA_STAR' : 0.5, 't_STAR' : 0.5, 'F_ESC10' : -1.0, 'ALPHA_ESC' : 0.3, 'M_TURN' : 8.7,
            'Omdmh2' : 0.11933, 'Ombh2' : 0.02242,  'hlittle' : 0.6736, 'Ln_1010_As' : 3.047, 'POWER_INDEX' : 0.9665, 
            'INVERSE_M_WDM' : 0.0, 'NEUTRINO_MASS_1' : 0.0, 'FRAC_WDM' : 0.0, 'M_WDM' : '20.0', 'L_X' : 40.0, 'NU_X_THRESH' : 500,
            'LOG10_PMF_SB' : -5.0, 'PMF_NB' : -2.0}



## Note that due to a strange naming convention in 21cmFAST, Omch2 actually corresponded to omega_dm
## This notation is deprecated today, prefer to use Omdmh2

# ---------------------------------------------------
# CHECKS AND PREPARATION OF THE DATA TO FED TO THE NN

def check_values(vals, metadata : MetaData):

    params_name = metadata.parameters_name
    min_params  = metadata.parameters_min_val
    max_params  = metadata.parameters_max_val

    if len(vals.shape) == 1:
        vals = vals[None, :]
    elif len(params_name) != vals.shape[-1]:
        raise ValueError("The input parameter array should have last dimension of size " + str(len(params_name)))

    # error handling, check that inputs are in the correct range
    if not (np.all(vals >= min_params) and np.all(max_params >= vals)):
        
        id_min_problem, pos_min_problem = np.where(vals < min_params)
        id_max_problem, pos_max_problem = np.where(max_params < vals)

        out_str = "Some parameters input are not in the correct range:\n"
        for i in range(len(id_min_problem)):
            out_str = out_str + str(id_min_problem[i]) +  " -> " +  params_name[pos_min_problem[i]] + " : " + str(vals[id_min_problem[i], pos_min_problem[i]]) + " < min_trained_value = " + str(min_params[pos_min_problem[i]]) + "\n"
        for i in range(len(id_max_problem)):
            out_str = out_str + str(id_max_problem[i]) +  " -> " +  params_name[pos_max_problem[i]] + " : " + str(vals[id_max_problem[i], pos_max_problem[i]]) + " > max_trained_value = " + str(max_params[pos_max_problem[i]]) + "\n"

        out_str = out_str.strip('\n')
        raise ValueError(out_str)


    
def input_values(metadata: MetaData, default: str = DEFAULT_VALUES, **kwargs):

    params_name = metadata.parameters_name

    # all parameters the neural network have been trained on
    iparams = {value: index for index, value in enumerate(params_name)}

    # predefined default values for most common parameters
    vals = np.array([default[p] for p in params_name])
        
    # check that the arguments passed in kwargs were trained on
    kw_keys = np.array(list(kwargs.keys()))
    kw_vals = np.array(list(kwargs.values()))

    # translate the keys that could be comming with a different naming convention
    kw_keys = np.array([MP_KEY_CORRESPONDANCE[key] if key in MP_KEY_CORRESPONDANCE.keys() else key for key in kw_keys])

    # error handling, check that inputs are in the trained parameters list
    # concatenate params_name and kw_keys and get unique input, if all goes well
    # the resulting array should have the same length as params_name
    if len(np.unique(np.concatenate((params_name, kw_keys)))) != len(params_name):
        raise ValueError("Some arguments of " + str(kw_keys) + " are not in the trained parameters list: " + str(params_name))

    # give their value to the parameters
    vals[[iparams[kw] for kw in kw_keys]] = kw_vals

    # error handling, check that inputs are in the correct range
    check_values(vals, metadata)


    #if not (np.all(vals >= min_params) and np.all(max_params >= vals)):
    #    min_problem = np.where(vals < min_params)[0]
    #    max_problem = np.where(max_params < vals)[0]
    #
    #    out_str = "Some parameters input are not in the correct range:\n"
    #    for i in min_problem:
    #        out_str = out_str + params[i] + " : " + str(vals[i]) + " < min_trained_value = " + str(min_params[i]) + "\n"
    #    for i in max_problem:
    #        out_str = out_str + params[i] + " : " + str(vals[i]) + " > max_trained_value = " + str(max_params[i]) + "\n"
    #
    #    out_str = out_str.strip('\n')
    #    raise ValueError(out_str)

    return vals


def uniform_input_values(metadata: MetaData, default:dict = DEFAULT_VALUES, **kwargs):
    vals = input_values(metadata, default, **kwargs)
    return true_to_uniform(vals, metadata.parameters_min_val, metadata.parameters_max_val)

def uniform_input_array(theta : np.ndarray, metadata: MetaData):
    check_values(theta, metadata)
    return true_to_uniform(theta, metadata.parameters_min_val, metadata.parameters_max_val)

# ---------------------------------------------------

# ---------------------------------------------------
# PREDICTION FUNCTIONS


def predict_classifier(classifier: Classifier | None = None, 
                       default : dict | None = None, 
                       **kwargs) -> bool:
    """
    Prediction of the classifier. See `predict_classifier_numpy` for a more
    advanced but faster method (working with numpy arrays).

    Parameters
    -----------
    classifier: nnero.Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    **kwargs
        Any value for a parameter the classifier has been trained on.
        Keys should corresponds to `classifier.parameters_name`.

    Returns
    --------
    bool
        True for an early reionization.
        False for a late reionization.
    """

    if default is None:
        default = DEFAULT_VALUES
    
    # if no classifier pass as input, load the default one
    if classifier is None:
        classifier = Classifier.load()

    u_vals = uniform_input_values(classifier.metadata, default, **kwargs)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = classifier.forward(u_vals)
        res = np.rint(res.numpy())
        res = res.astype(bool)

    return res[0]


def predict_classifier_numpy(theta: np.ndarray, 
                             classifier: Classifier | None = None) -> np.ndarray:
    """
    Prediction of the classifier. Same as `predict_classifier` but much faster
    as it work with numpy arrays.

    Parameters
    -----------
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        classifier for training. These parameters and their order are accesible 
        calling `classifier.info()`.
    classifier: nnero.Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.

    Returns
    --------
    np.ndarray
        Array of booleans.
        True for an early reionization.
        False for a late reionization.
    """

    if classifier is None:
        classifier = Classifier.load()

    u_vals = uniform_input_array(theta, classifier.metadata)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = classifier.forward(u_vals)
        res = np.rint(res.numpy())
        res = res.astype(bool)

    return res



def predict_regressor(regressor: Regressor | None = None, 
                      default: dict | None = None, 
                      **kwargs) -> np.ndarray:
    """
    Prediction of the regressor. See `predict_regressor_numpy` for a more
    advanced but faster method (working with numpy arrays).

    Parameters
    -----------
    regressor: nnero.Regressor | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.
    default : dict | None, optional
        Dictionnary of default values, by default None.
        The default dictionnaly should have keys corresponding to those of the trained dataset
        as accessible from `regressor.info()`.
    **kwargs
        Any value for a parameter the regressor has been trained on.
        Keys should corresponds to `regressor.parameters_name`.

    Returns
    --------
    np.ndarray
        Array of values for X_e.
    """

    if default is None:
        default = DEFAULT_VALUES

    # if no regressor passed as input, load the default one
    if regressor is None:
        regressor = Regressor.load()
     
    u_vals = uniform_input_values(regressor.metadata, default, **kwargs)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = regressor.forward(u_vals).numpy()

    return res




def predict_regressor_numpy(theta: np.ndarray, 
                            regressor: Regressor | None = None) -> np.ndarray:
    """
    Prediction of the regressor. Same as `predict_regressor` but much faster
    as it work with numpy arrays.

    Parameters
    -----------
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        regressor for training. These parameters and their order are accesible 
        calling `regressor.info()`.
    regressor: nnero.Regressor | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.

    Returns
    --------
    np.ndarray
        2D array of shape (n, q) for the values for X_e
        wit q the number of redshift bins. 
    """

    if regressor is None:
        regressor = Regressor.load()

    u_vals = uniform_input_array(theta, regressor.metadata)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = regressor.forward(u_vals).numpy()

    return res   



def predict_interpolator(interpolator: Interpolator | None = None, 
                         default: dict | None = None, 
                         parameter: str | None = None,
                         **kwargs) -> np.ndarray:
    """
    Prediction of the interpolator. See `predict_interpolator_numpy` for a more
    advanced but faster method (working with numpy arrays).

    Parameters
    -----------
    interpolator: nnero.Interpolator | None, optional
        Interpolator object already trained. Default is None. If None, the default
        interpolator `DefaultInterpolator_<parameter>` is used.
    default : dict | None, optional
        Dictionnary of default values, by default None.
        The default dictionnaly should have keys corresponding to those of the trained dataset
        as accessible from `interpolator.info()`.
    parameter: str | None, optional
        If no interpolator given in input, must be specified to know which interpolator to load.
        Default is None.
    **kwargs
        Any value for a parameter the interpolator has been trained on.
        Keys should corresponds to `interpolator.parameters_name`.

    Returns
    --------
    float
        Interpolated value.

    Raises
    ------
    ValueError
        If no interpolator given, need to specify on which parameter we cant to interpolate
        in order to load the associated default interpolator.
    """

    if default is None:
        default = DEFAULT_VALUES

    # if no interpolator passed as input, load the default one
    if interpolator is None:
        if parameter is not None:
            interpolator = Interpolator.load("DefaultInterpolator_" + parameter)
        else:
            raise ValueError("Need to pass the parameter to interpolate to the predictor.")
     
    u_vals = uniform_input_values(interpolator.metadata, default, **kwargs)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = interpolator.forward(u_vals).numpy()

    return res




def predict_interpolator_numpy(theta: np.ndarray, 
                               interpolator: Regressor | None = None,
                               parameter: str | None = None) -> np.ndarray:
    """
    Prediction of the interpolator. Same as `predict_interpolator` but much faster
    as it work with numpy arrays.

    Parameters
    -----------
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        regressor for training. These parameters and their order are accesible 
        calling `regressor.info()`.
    interpolator: nnero.Interpolator | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.
    parameter: str | None, optional
        If no interpolator given in input, must be specified to know which interpolator to load.
        Default is None.

    Returns
    --------
    np.ndarray
        1D array of shape n for the inteprolated values.

    Raises
    ------
    ValueError
        If no interpolator given, need to specify on which parameter we cant to interpolate
        in order to load the associated default interpolator.
    """

    # if no interpolator passed as input, load the default one
    if interpolator is None:
        if parameter is not None:
            interpolator = Interpolator.load("DefaultInterpolator_" + parameter)
        else:
            raise ValueError("Need to pass the parameter to interpolate to the predictor.")
        
    u_vals = uniform_input_array(theta, interpolator.metadata)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = interpolator.forward(u_vals).numpy()

    return res   





def predict_Xe(classifier: Classifier | None = None, 
               regressor:  Regressor  | None = None, 
               default: dict | None = None,
               **kwargs) -> np.ndarray | bool :
    """
    Prediction of the free electron fraction X_e, taking into account
    the selection from the classifier. See `predict_Xe_numpy` for a more
    advanced but faster method (working with numpy arrays).

    Parameters
    ----------
    classifier : Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    regressor : Regressor | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.
    default : dict | None, optional
        Dictionnary of default values, by default None.
        The default dictionnaly should have keys corresponding to those of the trained dataset
        as accessible from `regressor.info()` or `classifier.info()`.
    **kwargs
        Any value for a parameter the classifier and regressor have been trained on.
        Keys should corresponds to `regressor.parameters_name` or `classifier.parameters_name`.
        
    Returns
    -------
    np.ndarray | bool
        Returns `False` if the classifier returns `False`. Returns the prediction
        from the regressor for the free electron fraction X_e otherwise.
    """
    if default is None:
        default = DEFAULT_VALUES
    
    early = predict_classifier(classifier, default, **kwargs)

    if not early:
        return False
    
    xHII = predict_regressor(regressor, default, **kwargs)
    return xHII


def predict_Xe_numpy(theta: np.ndarray,
                     classifier: Classifier | None = None, 
                     regressor:  Regressor  | None = None) -> np.ndarray:
    """
    Prediction of the free electron fraction X_e, taking into account
    the selection from the classifier. Same as `predict_Xe` but much faster
    as it work with numpy arrays.

    Parameters
    ----------
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        regressor for training. These parameters and their order are accesible 
        calling `regressor.info()`.
    classifier : Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    regressor : Regressor | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.

    Returns
    -------
    np.ndarray
        2D array of shape (n, q) for the values for X_e
        with q the number for parameters.
        When the classifier outputs false, the array is filled with -1. 
    """
    
    if regressor is None:
        regressor = Regressor.load()
    
    mask = predict_classifier_numpy(theta, classifier)
    xHII = -np.ones((mask.shape[0], len(regressor.metadata.z)), dtype=np.float64)

    xHII[mask, :] = predict_regressor_numpy(theta[mask, :], regressor)

    return xHII





def predict_parameter(classifier: Classifier | None = None,  
                      interpolator:  Interpolator | None = None,  
                      default: dict | None = None,
                      parameter: str | None = None,
                      **kwargs) -> np.ndarray | bool :
    """
    Prediction of the interpolated parameter, taking into account
    the selection from the classifier. See `predict_parameter_numpy` for a more
    advanced but faster method (working with numpy arrays).

    Parameters
    ----------
    classifier : Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    interpolator : Interpolator | None, optional
        Interpolator object already trained. Default is None. If None, the default
        interpolator `DefaultInterpolator_<parameter>` is used.
    default : dict | None, optional
        Dictionnary of default values, by default None.
        The default dictionnaly should have keys corresponding to those of the trained dataset
        as accessible from `interpolator.info()` or `classifier.info()`.
    **kwargs
        Any value for a parameter the classifier and interpolator have been trained on.
        Keys should corresponds to `regressor.parameters_name` or `classifier.parameters_name`.
        
    Returns
    -------
    np.ndarray | bool
        Returns `False` if the classifier returns `False`. Returns the prediction
        from the interpolator for the parameter otherwise.
    """

    if default is None:
        default = DEFAULT_VALUES
    
    early = predict_classifier(classifier, default, **kwargs)

    if not early:
        return False
    
    return predict_interpolator(interpolator, default, parameter, **kwargs)


def predict_parameter_numpy(theta: np.ndarray,
                            classifier: Classifier | None = None, 
                            interpolator:  Interpolator  | None = None,
                            parameter: str | None = None) -> np.ndarray:
    """
    Prediction of the interpolated parameter, taking into account
    the selection from the classifier. Same as `predict_parameter` but much faster
    as it work with numpy arrays.

    Parameters
    ----------
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        regressor for training. These parameters and their order are accesible 
        calling `regressor.info()`.
    classifier : Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    interpolator : Interpolator | None, optional
        Interpolator object already trained. Default is None. If None, the default
        interpolator `DefaultInterpolator` is used.

    Returns
    -------
    np.ndarray
        1D array of shape (n) for the value of the parameter
        When the classifier outputs false, the array is filled with -1. 
    """
    
    # if no interpolator passed as input, load the default one
    if interpolator is None:
        if parameter is not None:
            interpolator = Interpolator.load("DefaultInterpolator_" + parameter)
        else:
            raise ValueError("Need to pass the parameter to interpolate to the predictor.")
    
    mask = predict_classifier_numpy(theta, classifier)
    res = -np.ones((mask.shape[0]), dtype=np.float64)

    res[mask] = predict_interpolator_numpy(theta[mask, :], interpolator, parameter)

    return res




def predict_tau_from_Xe(xe: np.ndarray, 
                        metadata: MetaData, 
                        default: dict | None = None, 
                        **kwargs) -> float:
    """
    Predict the optical depth to reionization from an array of X_e.
    See `predict_tau_from_Xe_numpy` for a more advanced but faster method 
    (working with numpy arrays).

    Parameters
    ----------
    xe : np.ndarray
        Array for the free electron fraction X_e = xHII(1+db).
    metadata : MetaData
        Metadata object (attached to the networks) describing properties
        of the data the networks have been trained on.
    default : dict | None, optional
        Dictionnary of default values, by default None.
        The default dictionnaly should have keys corresponding to those of the trained dataset
        as accessible from `metadata.parameters_name`.
    **kwargs
        Any value for a parameter in metadata.
        Keys should corresponds to `metadata.parameters_name`.
        
    Returns
    -------
    float
    """

    if default is None:
        default = DEFAULT_VALUES

    vals = input_values(metadata, default, **kwargs)
    
    omega_b = np.array([vals[metadata.pos_omega_b]])[None, :]
    omega_c = np.array([vals[metadata.pos_omega_dm]])[None, :]
    hlittle = np.array([vals[metadata.pos_hlittle]])[None, :]
    
    return optical_depth_no_rad(metadata.z[None, :], xe[None, :], omega_b, omega_c, hlittle)[0]


def predict_tau_from_Xe_numpy(xe: np.ndarray, 
                              theta : np.ndarray, 
                              metadata : MetaData) -> np.ndarray:
    """
    Predict the optical depth to reionization from an array of X_e.
    Same as `predict_tau_from_Xe` but much faster as it work with numpy arrays.

    Parameters
    ----------
    xe : np.ndarray
        Array for the free electron fraction X_e = xHII(1+db).
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        regressor for training. These parameters and their order are accesible 
        calling `metadata.parameters_name`.
    metadata : MetaData
        Metadata object (attached to the networks) describing properties
        of the data the networks have been trained on.

    Returns
    -------
    np.ndarray
        Array of values of the optical depth to reionization.
    """
    
    omega_b = theta[:, metadata.pos_omega_b]
    omega_c = theta[:, metadata.pos_omega_dm]
    hlittle = theta[:, metadata.pos_hlittle]

    return optical_depth_no_rad(metadata.z[None, :], xe, omega_b, omega_c, hlittle)



def predict_tau(classifier: Classifier | None = None,
                regressor: Regressor   | None = None,
                default: dict | None = None,
                **kwargs) -> float:
    """
    Predict the optical depth to reionization from a trained classifier and regressor
    as well as parameters passed in kwargs or default. See `predict_tau_numpy` 
    for a more advanced but faster method (working with numpy arrays).

    Parameters
    ----------
    classifier : Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    regressor : Regressor | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.
    default : dict | None, optional
        Dictionnary of default values, by default None.
        The default dictionnaly should have keys corresponding to those of the trained dataset
        as accessible from `regressor.info()` or `classifier.info()`.
    **kwargs
        Any value for a parameter the classifier and regressor have been trained on.
        Keys should corresponds to `regressor.parameters_name` or `classifier.parameters_name`.
        

    Returns
    -------
    float
    """
    
    if default is None:
        default = DEFAULT_VALUES

    if regressor is None:
        regressor = Regressor.load()

    xHII = predict_Xe(classifier, regressor, default, **kwargs)
    
    if xHII is False:
        return -1
    
    return predict_tau_from_Xe(xHII, regressor.metadata, default, **kwargs)



def predict_tau_numpy(theta: np.ndarray, 
                      classifier: Classifier | None = None,
                      regressor: Regressor   | None = None) -> np.ndarray:
    """
    Predict the optical depth to reionization from a trained classifier and regressor
    as well as parameters passed in kwargs or default. Same as `predict_tau` but much faster
    as it work with numpy arrays.

    Parameters
    ----------
    theta: np.ndarray
        Array of shape (n, p) of parameters, where p is the number of parameters.
        The p parameters values should be given in the order they were fed to the
        regressor for training. These parameters and their order are accesible 
        calling `regressor.info()` or `classifier.info()`.
    classifier : Classifier | None, optional
        Classifier object already trained. Default is None. If None, the default
        classifier `DefaultClassifier` is used.
    regressor : Regressor | None, optional
        Regressor object already trained. Default is None. If None, the default
        regressor `DefaultRegressor` is used.

    Returns
    -------
    np.ndarray
        Array of values of the optical depth to reionization.
    """
    
    if regressor is None:
        regressor = Regressor.load()
    
    xHII = predict_Xe_numpy(theta, classifier, regressor)
    
    res  = - np.ones(len(xHII))
    mask = (xHII[:, -1] != -1)
    
    res[mask]  = predict_tau_from_Xe_numpy(xHII[mask, :], theta[mask, :], regressor.metadata)

    return res

###########################################

# Defining equivalent names for functions that are now deprecated

def predict_xHII(classifier: Classifier | None = None, 
                 regressor:  Regressor  | None = None, 
                 default: dict | None = None,
                 **kwargs) -> np.ndarray:
    """
    .. deprecated::
        See `predict_Xe` instead.
    """
    return predict_Xe(classifier, regressor, default, **kwargs)

def predict_xHII_numpy(theta: np.ndarray,
                     classifier: Classifier | None = None, 
                     regressor:  Regressor  | None = None) -> np.ndarray:
    """
    .. deprecated::
        See `predict_Xe_numpy` instead.
    """
    return predict_Xe_numpy(theta, classifier, regressor)

def predict_tau_from_xHII(Xe, metadata : MetaData, default: dict | None = None, **kwargs) -> float:
    """
    .. deprecated::
        See `predict_tau_from_Xe` instead.
    """
    return predict_tau_from_Xe(Xe, metadata, default, **kwargs)

def predict_tau_from_xHII_numpy(Xe, theta : np.ndarray, metadata : MetaData) -> np.ndarray:
    """
    .. deprecated::
        See `predict_tau_from_Xe_numpy` instead.
    """
    return predict_tau_from_Xe_numpy(Xe, theta, metadata)
