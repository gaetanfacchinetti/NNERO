import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from .data import MetaData
from .data import DataBase
from .data import true_to_uniform, tau_ion, uniform_to_true

class NeuralNetwork:

    def __init__(self, model, name):
        self._model = model
        self._name = name
        self._history = {'loss' : np.zeros(0), 'val_loss' : np.zeros(0)}
        self._metadata  = None

    def _compile(self, loss, *, optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3), **kwargs):
        self._history = {'loss' : np.zeros(0), 'val_loss' : np.zeros(0)}
        self._model.compile(optimizer, loss=loss, **kwargs)

    def _train(self, x, y, metadata, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):

        # set the metadata if it is new
        if (self._metadata is not None) and (self._metadata != metadata):
            raise ValueError("The metadata should be the same when training successively")
        
        if self._metadata is None:
            self._metadata = metadata

        # train the model according to the specification wanted by the user
        new_history = self._model.fit(x, y, epochs = epochs, validation_split = validation_split, verbose = verbose, batch_size = batch_size, **kwargs)

        # combine the histories
        for key in ['loss', 'val_loss']:
            self._history[key] = np.concatenate((self._history[key], new_history.history[key]), axis=0)

    # -------------------------------
    # redefine very simple functions from keras in this Class for convenience

    def save(self, path = ".") -> None:
        
        with open(path + "/" + self._name + '_history.npz', 'wb') as file:
            np.savez(file, history = self._history)

        # save the metadata and the model
        self._metadata.save(path + "/" + self._name)
        self._model.save(path + "/" + self._name + '.keras')

    def _load_history(self, path) -> None:
        
        with open(path + "/" + self._name + '_history.npz', 'rb') as file: 
            data = np.load(file, allow_pickle=True)
            self._history = data['history'].tolist()

    
    def postprocess(self, prediction):
        return prediction

    
    def predict(self, x, **kwargs):
        
        if len(x.shape) < 2:
            x = np.array([x])
        
        return self.postprocess(self._model.predict(x, verbose=0), **kwargs)



    @property
    def model(self):
        return self._model
    
    @property
    def history(self):
        return self._history
    
    @property
    def name(self):
        return self._name



class Classifier(NeuralNetwork):

    def __init__(self, n_input = 12, *, model = None, name = "DefaultClassifier", check_name = True):

        if check_name and (name == "DefaultClassifier" and (model is not None)):
            raise AttributeError("Plase give your custom model a name different from the default value")

        if model is None:
    
            # Define a simple default classifier
            model = keras.Sequential(
                [
                keras.Input(shape=(n_input,), name="input"),
                keras.layers.Dense(32, name="layer_1_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(32, name="layer_2_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(32, name="layer_3_nn", activation = "sigmoid", kernel_initializer='normal'),
                keras.layers.Dense(3,  name="output", kernel_initializer='normal', activation = "softmax"),
                ]
            )

        super().__init__(model, name)

    def compile(self, optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3), **kwargs):
        self._compile("categorical_crossentropy", optimizer=optimizer, **kwargs)

    def train(self, db, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):
        self._train(db.u_train, db.c_train, db.metadata, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultClassifier"):
        
        model = keras.models.load_model(path + "/" + name + '.keras')
        classifier = Classifier(model = model, name = name, check_name = False)
        classifier._load_history(path)
        classifier._metadata = MetaData.load(path + "/" + name)
        return classifier
    
    def test(self, database):

        res = np.zeros((3, 3))

        indices = [[[] for j in range(0, 3)] for i in range(0, 3)]
        prediction = self.predict(database.u_test)

        for i in range(0, database.metadata.ntest):

            index = np.argmax(prediction[i, :])

            for j in range(0, 3):
                if database.c_test[i, j] == 1:
                    res[index, j] = res[index, j] + 1
                    indices[index][j].append(i)
                    
        print("\t | actually reionized | actrually valid (not reionized) | actually not valid")
        print("predicted reionised             | " + str(res[0, 0]) + " | " + str(res[0, 1]) + " | " + str(res[0, 2]))
        print("predicted valid (not reionized) | " + str(res[1, 0]) + " | " + str(res[1, 1]) + " | " + str(res[1, 2]))
        print("predicted not valid             | " + str(res[2, 0]) + " | " + str(res[2, 1]) + " | " + str(res[2, 2]))

        return res

    


def rel_abs_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(1.0-y_pred/y_true))

keras.utils.get_custom_objects()['rel_abs_loss'] = rel_abs_loss



class Regressor(NeuralNetwork):

    # what worked so far:
    # - directly put xHIIdb as y_train
    # - relu activation function
    # - can do large dense layers with some dropout layers

    def __init__(self, n_input = 12, n_output = 32, *, model = None, name = "DefaultRegressor", check_name = True):

        if check_name and (name == "DefaultRegressor" and (model is not None)):
            raise AttributeError("Plase give your custom model a name different from the default value")

        if model is None:

            # Define a simple default classifier
            model = keras.Sequential(
                [
                keras.Input(shape=(n_input,), name="input"),
                keras.layers.Dense(128, name="layer_1_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_2_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_3_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_4_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_5_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_6_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_7_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_8_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(128, name="layer_9_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(n_output,  name="output", kernel_initializer='normal'),
                ]
            )
        
        super().__init__(model, name)


    def compile(self, optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5), **kwargs):
        self._compile(rel_abs_loss, optimizer=optimizer, **kwargs)

    def train(self, db, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):
        self._train(db.u_train_valid, db.y_train_valid, db.metadata, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultRegressor"):

        model = keras.models.load_model(path + "/" + name + '.keras',  custom_objects={"rel_abs_loss": rel_abs_loss})
        regressor = Regressor(model=model, name = name, check_name=False)
        regressor._load_history(path)
        regressor._metadata = MetaData.load(path + "/" + name)
        return regressor


    def postprocess(self, prediction, bias = 0.0):
        # prediction must be a 2D-array as returned by model.predict()
        
        res = np.zeros(prediction.shape)
        res[:, :] = prediction[:, :] / (1.0 + 0.01 * bias)

        for i in range(0, res.shape[0]):
            
            indexes_reio = np.where(res[i, :] >= 1.0)[0]
            
            if len(indexes_reio > 0):
                index_max = indexes_reio[-1]
                res[i, 0:index_max + 1] = 1.0

        return res


    def test(self, database, **kwargs):

        # compute the average relative difference error
        prediction = self.predict(database.u_test_valid, **kwargs)
        error_rel = 100 * (prediction / database.y_test_valid-1)
        

        # compute the error for the optical depth
        omch2 = database.x_test_valid[:, database.metadata.param_names.index('Omch2')]
        ombh2 = database.x_test_valid[:, database.metadata.param_names.index('Ombh2')]
        h = database.x_test_valid[:, database.metadata.param_names.index('hlittle')]

        error_tau = np.zeros(database.metadata.ntest_valid)
        for i in range(0, database.metadata.ntest_valid):
            tau_pred = tau_ion(database.metadata.redshifts, prediction[i, :], ombh2[i], omch2[i] + ombh2[i], h[i])
            tau_true = tau_ion(database.metadata.redshifts, database.y_test_valid[i, :], ombh2[i], omch2[i] + ombh2[i], h[i])
            error_tau[i] = 100 * (tau_pred/tau_true-1)

        return error_rel, error_tau
    




class ODToRRegressor(NeuralNetwork):

    def __init__(self, n_input = 12, *, model = None, name = "DefaultODToRRegressor", check_name = True):

        if check_name and (name == "DefaultODToRRegressor" and (model is not None)):
            raise AttributeError("Plase give your custom model a name different from the default value")

        if model is None:

            # Define a simple default classifier
            model = keras.Sequential(
                [
                keras.Input(shape=(n_input,), name="input"),
                keras.layers.Dense(32, name="layer_1_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(32, name="layer_2_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(32, name="layer_3_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(32, name="layer_4_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(32, name="layer_5_nn", activation = "relu", kernel_initializer='normal'),
                keras.layers.Dense(1,  name="output", kernel_initializer='normal'),
                ]
            )
        
        super().__init__(model, name)


    def compile(self, optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5), **kwargs):
        self._compile(rel_abs_loss, optimizer=optimizer, **kwargs)

    def train(self, db, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):
        self._train(db.u_train_valid, db.ttau_train_valid, db.metadata, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultODToRRegressor"):

        model = keras.models.load_model(path + "/" + name + '.keras',  custom_objects={"rel_abs_loss": rel_abs_loss})
        regressor = ODToRRegressor(model=model, name = name, check_name=False)
        regressor._load_history(path)
        regressor._metadata = MetaData.load(path + "/" + name)
        return regressor

    def postprocess(self, prediction):
        prediction[prediction > 1.0] = 1.0
        prediction[prediction < 0.69] = 0.69
        return prediction
    

    def test(self, database):

        prediction = self.predict(database.u_test_valid)[:, 0]
        return (100 * (prediction / database.ttau_test_valid - 1))




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


# predict the evolution of tau_ion

def predict_tau(classifier = None, regressor = None, **kwargs):
    
    if regressor is None:
        regressor = Regressor.load(".", "DefaultRegressor")

    x_HII = predict_xHII(classifier, regressor, **kwargs)

    if x_HII is None:
        return -1

    #ombh2 = kwargs.get('Ombh2', DEFAULT_VALUES['Ombh2'])
    #omch2 = kwargs.get('Ombh2', DEFAULT_VALUES['Omch2'])
    #h = kwargs.get('hlittle', DEFAULT_VALUES['hlittle'])
    #return tau_ion(regressor._metadata.redshifts, x_HII, ombh2, omch2 + ombh2, h)
    
    return predict_odtor_regressor(regressor, **kwargs)
