import numpy as np
import torch

from .data       import true_to_uniform
from .classifier import Classifier
from .regressor  import Regressor



DEFAULT_VALUES = {'F_STAR10' : -1.5, 'ALPHA_STAR' : 0.5, 't_STAR' : 0.5, 'F_ESC10' : -1.0, 'ALPHA_ESC' : -0.5, 'M_TURN' : 8.7,
            'Omch2' : 0.11933, 'Ombh2' : 0.02242, 'hlittle' : 0.6736, 'Ln_1010_As' : 3.047, 'POWER_INDEX' : 0.9665, 
            'INVERSE_M_WDM' : 0.05, 'NEUTRINO_MASS_1' : 0.02, 'FRAC_WDM' : 0.0}



def input_values(params, **kwargs):

    # all parameters the neural network have been trained on
    iparams = {value: index for index, value in enumerate(params)}

    # predefined default values for most common parameters
    vals = np.array([DEFAULT_VALUES[p] for p in params])
        
    # check that the arguments passed in kwargs were trained on
    kw_keys = np.array(list(kwargs.keys()))
    kw_vals = np.array(list(kwargs.values()))

    if len(np.unique([params, kw_keys])) > len(params):
        raise ValueError("Some arguments" + kw_keys + "are not in the trained parameters list" + str(params))

    vals[[iparams[kw] for kw in kw_keys]] = kw_vals

    #uvals = 

    return vals

"""
def predict_classifier(classifier: Classifier | None = None, **kwargs) -> bool:
    
    if classifier is None:
        classifier = Classifier.load("./DefaultClassifier")

    vals_classifier = values(classifier.metadata.parameters_name)

    # compute the reduced value of the parameter, in the range [0, 1]
    u_vals_classifier = np.zeros(len(params))

    for ikey, key in enumerate(params):
        
        min = classifier.metadata.parameters_min_val[ikey]
        max = classifier.metadata.parameters_max_val[ikey]
        val = vals_classifier[key]

        u_vals_classifier[ikey] = true_to_uniform(val, min, max)

    res_classifier = np.argmax(classifier.predict(np.array([u_vals_classifier]))[0])
    
    return res_classifier
    
    """