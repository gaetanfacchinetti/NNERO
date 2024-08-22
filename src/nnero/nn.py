import numpy as np
import torch
import torch.nn as nn

from .cosmology import optical_depth

from .data import true_to_uniform, uniform_to_true








class Regressor(nn.Module):

    def __init__(self, dim = 80):

        super(Regressor, self).__init__()

        self._model = nn.Sequential(
                    nn.Linear(16, dim),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, dim), nn.ReLU(),
                    nn.Linear(dim, 50),
                    )
                    

    def forward(self, x): 
        return torch.clamp(self._model(x), max=1.0)
    
    def tau_ion(self, x, y):
        omega_b = uniform_to_true(x[:, ib], parameters_min_val[ib], parameters_max_val[ib])
        omega_c = uniform_to_true(x[:, ic], parameters_min_val[ic], parameters_max_val[ic])
        h       = uniform_to_true(x[:, ih], parameters_min_val[ih], parameters_max_val[ih])
        res     = nnero.cosmology.optical_depth_no_rad_torch(z_tensor, y, omega_b, omega_c, h)
        return res



class McGreerRegressor(NeuralNetwork):

    def __init__(self, n_input = 12, *, model = None, name = "DefaultMcGreerRegressor", check_name = True):

        if check_name and (name == "DefaultMcGreerRegressor" and (model is not None)):
            raise AttributeError("Plase give your custom model a name different from the default value")

        if model is None:

            # Define a simple default classifier
            model = keras.Sequential(
                [
                keras.Input(shape=(n_input,), name="input"),
                keras.layers.Dense(32, name="layer_1_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(16, name="layer_2_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(8, name="layer_3_nn", activation = "relu", kernel_initializer='normal'),
                #keras.layers.Dense(32, name="layer_4_nn", activation = "relu", kernel_initializer='normal'),
                #keras.layers.Dense(32, name="layer_5_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(1,  name="output", kernel_initializer='normal'),
                ]
            )
        
        super().__init__(model, name)


    def compile(self, optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5), **kwargs):
        self._compile(rel_abs_loss, optimizer=optimizer, **kwargs)

    def train(self, db, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):
        self._train(db.u_train_valid_noreio, db.McGreer_train_valid_noreio, db.metadata, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultMcGreerRegressor"):

        model = keras.models.load_model(path + "/" + name + '.keras',  custom_objects={"rel_abs_loss": rel_abs_loss})
        regressor = McGreerRegressor(model=model, name = name, check_name=False)
        regressor._load_history(path)
        regressor._metadata = MetaData.load(path + "/" + name)
        return regressor


    def test(self, database):

        prediction = self.predict(database.u_test_valid_noreio)[:, 0]
        return 100 * (1 - prediction / database.McGreer_test_valid_noreio)




DEFAULT_VALUES = {'F_STAR10' : -1.5, 'ALPHA_STAR' : 0.5, 't_STAR' : 0.5, 'F_ESC10' : -1.0, 'ALPHA_ESC' : -0.5, 'M_TURN' : 8.7,
            'Omch2' : 0.11933, 'Ombh2' : 0.02242, 'hlittle' : 0.6736, 'Ln_1010_As' : 3.047, 'POWER_INDEX' : 0.9665, 'M_WDM' : 25.0}


# main functions

def predict_classifier(classifier = None, **kwargs):
    
    if classifier is None:
        classifier = Classifier.load(".", "DefaultClassifier")

    
    # all parameters the neural network have been trained on
    classifier_params = classifier._metadata.param_names

    # predefined default values for most common parameters
    vals_classifier = {}
    for key in classifier_params:
        vals_classifier[key] = DEFAULT_VALUES[key]
        
    # modify the value to that given in **kwargs
    for kw, val in kwargs.items():
        if kw in classifier_params:
            vals_classifier[kw] = val
        else :
            raise ValueError(str(kw) + " in argument is not a parameter on which " + str(classifier._name) 
                             + " has trained. Available parameters are :" + str(classifier_params))

    # compute the reduced value of the parameter, in the range [0, 1]
    u_vals_classifier = np.zeros(len(classifier_params))

    for ikey, key in enumerate(classifier_params):
        
        min = np.min(classifier._metadata.param_ranges[key])
        max = np.max(classifier._metadata.param_ranges[key])
        val = vals_classifier[key]

        u_vals_classifier[ikey] = true_to_uniform(val, min, max)

    res_classifier = np.argmax(classifier.predict(np.array([u_vals_classifier]))[0])

    # 0 reionized
    # 1 valid but not reionized
    # 2 not valid
    
    return res_classifier
    


def predict_regressor(regressor = None, **kwargs):
    
    if regressor is None:
        regressor = Regressor.load(".", "DefaultRegressor")
    
    # all parameters the neural network have been trained on
    regressor_params = regressor._metadata.param_names

    # predefined default values for most common parameters
    vals_regressor = {}
    for key in regressor_params:
        vals_regressor[key] = DEFAULT_VALUES[key]
        
    # modify the value to that given in **kwargs
    for kw, val in kwargs.items():
        if kw in regressor_params:
            vals_regressor[kw] = val
        else :
            raise ValueError(str(kw) + " in argument is not a parameter on which " + str(regressor._name) + 
                             " has trained. Available parameters are :\n" + str(regressor_params))

    # compute the reduced value of the parameter, in the range [0, 1]
    u_vals_regressor = np.zeros(len(regressor_params))

    for ikey, key in enumerate(regressor_params):
        
        max = np.max(regressor._metadata.param_ranges[key])
        min = np.min(regressor._metadata.param_ranges[key])
        val = vals_regressor[key]

        u_vals_regressor[ikey] = true_to_uniform(val, min, max)

    return regressor.predict(np.array([u_vals_regressor]))[0]

    

def predict_odtor_regressor(odtor_regressor = None, **kwargs):
    
    if odtor_regressor is None:
        odtor_regressor = ODToRRegressor.load(".", "DefaultODToRRegressor")
    
    # all parameters the neural network have been trained on
    regressor_params = odtor_regressor._metadata.param_names

    # predefined default values for most common parameters
    vals_regressor = {}
    for key in regressor_params:
        vals_regressor[key] = DEFAULT_VALUES[key]
        
    # modify the value to that given in **kwargs
    for kw, val in kwargs.items():
        if kw in regressor_params:
            vals_regressor[kw] = val
        else :
            raise ValueError(str(kw) + " in argument is not a parameter on which " + str(regressor._name) + 
                             " has trained. Available parameters are :\n" + str(regressor_params))

    # compute the reduced value of the parameter, in the range [0, 1]
    u_vals_regressor = np.zeros(len(regressor_params))

    for ikey, key in enumerate(regressor_params):
        
        max = np.max(odtor_regressor._metadata.param_ranges[key])
        min = np.min(odtor_regressor._metadata.param_ranges[key])
        val = vals_regressor[key]

        u_vals_regressor[ikey] = true_to_uniform(val, min, max)

    return odtor_regressor.predict(np.array([u_vals_regressor]))[0]


# predict the evolution of xHII

def predict_xHII(classifier = None, regressor = None, **kwargs):

    res_classifier = predict_classifier(classifier, **kwargs)
    
    if res_classifier == 2:
        return None
    
    return predict_regressor(regressor, **kwargs)


# predict the evolution of optical_depth

def predict_tau(classifier = None, regressor = None, **kwargs):
    
    if regressor is None:
        regressor = Regressor.load(".", "DefaultRegressor")

    x_HII = predict_xHII(classifier, regressor, **kwargs)

    if x_HII is None:
        return -1

    #ombh2 = kwargs.get('Ombh2', DEFAULT_VALUES['Ombh2'])
    #omch2 = kwargs.get('Ombh2', DEFAULT_VALUES['Omch2'])
    #h = kwargs.get('hlittle', DEFAULT_VALUES['hlittle'])
    #return optical_depth(regressor._metadata.redshifts, x_HII, ombh2, omch2 + ombh2, h)
    
    return predict_odtor_regressor(regressor, **kwargs)
