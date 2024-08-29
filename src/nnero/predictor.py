import numpy as np
import torch


from .data       import MetaData, true_to_uniform
from .classifier import Classifier
from .regressor  import Regressor



DEFAULT_VALUES = {'F_STAR10' : -1.5, 'ALPHA_STAR' : 0.5, 't_STAR' : 0.5, 'F_ESC10' : -1.0, 'ALPHA_ESC' : -0.5, 'M_TURN' : 8.7,
            'Omch2' : 0.11933, 'Ombh2' : 0.02242, 'hlittle' : 0.6736, 'Ln_1010_As' : 3.047, 'POWER_INDEX' : 0.9665, 
            'INVERSE_M_WDM' : 0.05, 'NEUTRINO_MASS_1' : 0.02, 'FRAC_WDM' : 0.0, 'M_WDM' : '20.0', 'L_X' : 40.0, 'NU_X_THRESH' : 500}


def input_values(metadata: MetaData, **kwargs):

    params     = metadata.parameters_name
    min_params = metadata.parameters_min_val
    max_params = metadata.parameters_max_val

    # all parameters the neural network have been trained on
    iparams = {value: index for index, value in enumerate(params)}

    # predefined default values for most common parameters
    vals = np.array([DEFAULT_VALUES[p] for p in params])
        
    # check that the arguments passed in kwargs were trained on
    kw_keys = np.array(list(kwargs.keys()))
    kw_vals = np.array(list(kwargs.values()))

    # error handling, check that inputs are in the trained parameters list
    # concatenate params and kw_keys and get unique input, if all goes well
    # the resulting array should have the same length as params
    if len(np.unique(np.concatenate((params, kw_keys)))) != len(params):
        raise ValueError("Some arguments of " + str(kw_keys) + " are not in the trained parameters list: " + str(params))

    # give their value to the parameters
    vals[[iparams[kw] for kw in kw_keys]] = kw_vals

    # error handling, check that inputs are in the correct range
    if not (np.all(vals >= min_params) and np.all(max_params >= vals)):
        min_problem = np.where(vals < min_params)[0]
        max_problem = np.where(max_params < vals)[0]

        out_str = "Some parameters input are not in the correct range:\n"
        for i in min_problem:
            out_str = out_str + params[i] + " : " + str(vals[i]) + " < min_trained_value = " + str(min_params[i]) + "\n"
        for i in max_problem:
            out_str = out_str + params[i] + " : " + str(vals[i]) + " > max_trained_value = " + str(max_params[i]) + "\n"

        out_str = out_str.strip('\n')
        raise ValueError(out_str)

    return vals


def uniform_input_values(metadata: MetaData, **kwargs):
    vals = input_values(metadata, **kwargs)
    return true_to_uniform(vals, metadata.parameters_min_val, metadata.parameters_max_val)


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


def predict(classifier: Classifier | None = None, 
            regressor:  Regressor  | None = None, 
            **kwargs):
    
    early = predict_classifier(classifier, **kwargs)

    if not early:
        return False
    
    xHII = predict_regressor(regressor, **kwargs)
    return xHII