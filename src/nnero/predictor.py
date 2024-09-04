import numpy as np
import torch


from .data       import MetaData, true_to_uniform
from .cosmology  import optical_depth_no_rad_numpy
from .classifier import Classifier
from .regressor  import Regressor



DEFAULT_VALUES = {'F_STAR10' : -1.5, 'ALPHA_STAR' : 0.5, 't_STAR' : 0.5, 'F_ESC10' : -1.0, 'ALPHA_ESC' : -0.5, 'M_TURN' : 8.7,
            'Omch2' : 0.11933, 'Ombh2' : 0.02242, 'hlittle' : 0.6736, 'Ln_1010_As' : 3.047, 'POWER_INDEX' : 0.9665, 
            'INVERSE_M_WDM' : 0.05, 'NEUTRINO_MASS_1' : 0.02, 'FRAC_WDM' : 0.0, 'M_WDM' : '20.0', 'L_X' : 40.0, 'NU_X_THRESH' : 500}



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


    
def input_values(metadata: MetaData, **kwargs):

    params_name = metadata.parameters_name

    # all parameters the neural network have been trained on
    iparams = {value: index for index, value in enumerate(params_name)}

    # predefined default values for most common parameters
    vals = np.array([DEFAULT_VALUES[p] for p in params_name])
        
    # check that the arguments passed in kwargs were trained on
    kw_keys = np.array(list(kwargs.keys()))
    kw_vals = np.array(list(kwargs.values()))

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


def uniform_input_values(metadata: MetaData, **kwargs):
    vals = input_values(metadata, **kwargs)
    return true_to_uniform(vals, metadata.parameters_min_val, metadata.parameters_max_val)

def uniform_input_array(parameters : np.ndarray, metadata: MetaData):
    check_values(parameters, metadata)
    return true_to_uniform(parameters, metadata.parameters_min_val, metadata.parameters_max_val)

# ---------------------------------------------------

# ---------------------------------------------------
# PREDICTION FUNCTIONS


def predict_classifier(classifier:Classifier | None = None, **kwargs):
    """
        prediction of the classifier

    Parameters:
    -----------
    - classifier: nnero.Classifier
        classifier object already trained
    - **kwargs:
        any value for a parameter the classifier 
        has been trained on

    Returns:
    --------
    returns boolean value:
    True for an early reionization
    False for a late reionization
    """
    
    # if no classifier pass as input, load the default one
    if classifier is None:
        classifier = Classifier.load()

    u_vals = uniform_input_values(classifier.metadata, **kwargs)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = classifier.forward(u_vals)
        res = np.rint(res.numpy())
        res = res.astype(bool)

    return res[0]


def predict_classifier_numpy(parameters: np.ndarray, classifier: Classifier | None = None):

    if classifier is None:
        classifier = Classifier.load()

    u_vals = uniform_input_array(parameters, classifier.metadata)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = classifier.forward(u_vals)
        res = np.rint(res.numpy())
        res = res.astype(bool)

    return res



def predict_regressor(regressor: Regressor | None = None, **kwargs):
    """
        prediction of the regressor

    Parameters:
    -----------
    - classifier: nnero.Classifier
        classifier object already trained
    - **kwargs:
        any value for a parameter the classifier 
        has been trained on

    Returns:
    --------
    returns boolean value:
    True for an early reionization
    False for a late reionization
    """

    # if no regressor passed as input, load the default one
    if regressor is None:
        regressor = Regressor.load()
     
    u_vals = uniform_input_values(regressor.metadata, **kwargs)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = regressor.forward(u_vals).numpy()

    return res




def predict_regressor_numpy(parameters: np.ndarray, regressor: Classifier | None = None):

    if regressor is None:
        regressor = Regressor.load()

    u_vals = uniform_input_array(parameters, regressor.metadata)
    u_vals = torch.tensor(u_vals, dtype=torch.float32)
    
    with torch.no_grad():
        res = regressor.forward(u_vals).numpy()

    return res   



def predict_xHII(classifier: Classifier | None = None, 
            regressor:  Regressor  | None = None, 
            **kwargs):
    
    early = predict_classifier(classifier, **kwargs)

    if not early:
        return False
    
    xHII = predict_regressor(regressor, **kwargs)
    return xHII


def predict_xHII_numpy(parameters: np.ndarray,
            classifier: Classifier | None = None, 
            regressor:  Regressor  | None = None):
    
    early = predict_classifier_numpy(parameters, classifier)    
    xHII = predict_regressor_numpy(parameters, regressor)

    xHII[early == True] = -np.ones(xHII.shape[-1])

    return xHII



def predict_tau_from_xHII(xHII, metadata : MetaData, **kwargs):

    vals = input_values(metadata, **kwargs)
    
    omega_b = np.array([vals[metadata.pos_omega_b]])[None, :]
    omega_c = np.array([vals[metadata.pos_omega_c]])[None, :]
    hlittle = np.array([vals[metadata.pos_hlittle]])[None, :]
    
    return optical_depth_no_rad_numpy(metadata.z[None, :], xHII[None, :], omega_b, omega_c, hlittle)[0]


def predict_tau_from_xHII_numpy(xHII, parameters : np.ndarray, metadata : MetaData):
    
    omega_b = parameters[:, metadata.pos_omega_b]
    omega_c = parameters[:, metadata.pos_omega_c]
    hlittle = parameters[:, metadata.pos_hlittle]

    return optical_depth_no_rad_numpy(metadata.z[None, :], xHII, omega_b, omega_c, hlittle)


def predict_tau(classifier: Classifier | None = None,
                regressor: Regressor   | None = None,
                **kwargs):
    
    xHII = predict_xHII(classifier, regressor, **kwargs)
    
    if xHII is False:
        return -1
    
    return predict_tau_from_xHII(xHII, regressor.metadata, **kwargs)


