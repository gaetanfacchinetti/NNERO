import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class NeuralNetwork:

    def __init__(self, model, name):
        self._model = model
        self._name = name
        self._history = {'loss' : np.zeros(0), 'val_loss' : np.zeros(0)}

        self._params_list  = []
        self._params_range = {} # range of parameter values on which the nn is trained (true value not the unirformly drawn ones between 0 and 1)

    def _compile(self, loss, *, optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3), **kwargs):
        self._history = {'loss' : np.zeros(0), 'val_loss' : np.zeros(0)}
        self._model.compile(optimizer, loss=loss, **kwargs)

    def _train(self, x, y, params_list, params_range, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):

        # update the range of parameters
        if self._params_range == {} :
            self._params_range = params_range
            self._params_list  = params_list

        if self._params_range != params_range:
            # need to check this piece of code (not sure it is necessary to have it though)
            for key, value in self._params_range.items() : 
                self._params_range.value[0] = np.min(np.array([*params_range[key], *value]))
                self._params_range.value[1] = np.max(np.array([*params_range[key], *value]))

        # train the model according to the specification wanted by the user
        new_history = self._model.fit(x, y, epochs = epochs, validation_split = validation_split, verbose = verbose, batch_size = batch_size, **kwargs)

        # combine the histories
        for key in ['loss', 'val_loss']:
            self._history[key] = np.concatenate((self._history[key], new_history.history[key]), axis=0)

    # -------------------------------
    # redefine very simple functions from keras in this Class for convenience

    def save(self, path = ".") -> None:
        
        with open(path + "/" + self._name + '.npz', 'wb') as file:
            np.savez(file, history = self._history, params_range = self._params_range, params_list = self._params_list)

        self._model.save(path + "/" + self._name + '.keras')

    def predict(self, x):
        return self._model.predict(x, verbose=0)

    # -------------------------------

    def _load_history(self, path) -> None:
        
        with open(path + "/" + self._name + '.npz', 'rb') as file: 
            data = np.load(file, allow_pickle=True)
            self._history = data['history']
            self._params_range = data['params_range'].tolist()
            self._params_list  = data['params_list']


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
                keras.layers.Dense(10, name="layer_1_nn", activation = "sigmoid", kernel_initializer='normal'),
                keras.layers.Dense(10, name="layer_2_nn", activation = "sigmoid", kernel_initializer='normal'),
                keras.layers.Dense(3,  name="output", kernel_initializer='normal', activation = "softmax"),
                ]
            )

        super().__init__(model, name)

    def compile(self, optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3), **kwargs):
        self._compile("categorical_crossentropy", optimizer=optimizer, **kwargs)

    def train(self, db, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):
        self._train(db._x_train, db._y_train_class, db._params_keys, db._params_range, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultClassifier"):
        try:
            model = keras.models.load_model(path + "/" + name + '.keras')
            classifier =  Classifier(model = model, name = name, check_name = False)
            classifier._load_history(path)
            return classifier
        except Exception as e:
            pass
    



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
                keras.layers.Dense(n_output,  name="output", kernel_initializer='normal'),
                ]
            )

        super().__init__(model, name)

    def square_rel_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(1.0-y_pred/y_true))

    def compile(self, optimizer = tf.keras.optimizers.AdamW(learning_rate=5e-5), **kwargs):
        self._compile(self.square_rel_loss, optimizer=optimizer, **kwargs)

    def train(self, db, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):
        self._train(db._x_train_valid, db._y_train_valid, db._params_keys, db._params_range, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultRegressor"):
        try:
            model = keras.models.load_model(path + "/" + name + '.keras')
            regressor = Regressor(model, name, check_name=False)
            regressor._load_history(path)
            return regressor
        except Exception as e:
            pass
    


DEFAULT_VALUES = {'F_STAR10' : -1.5, 'ALPHA_STAR' : 0.5, 't_STAR' : 0.5, 'F_ESC10' : -1.0, 'ALPHA_ESC' : -0.5, 'M_TURN' : 8.7,
            'Omch2' : 0.11933, 'Ombh2' : 0.02242, 'hlittle' : 0.6736, 'Ln_1010_As' : 3.047, 'POWER_INDEX' : 0.9665, 'M_WDM' : 25.0}

# main functions

def predict_classifier(classifier = None, **kwargs):
    
    if classifier is None:
        classifier = Classifier.load(".", "DefaultClassifier")

    
    # all parameters the neural network have been trained on
    classifier_params = classifier._params_list

    # predefined default values for most common parameters
    vals_classifier = {}
    for key in classifier_params:
        vals_classifier[key] = DEFAULT_VALUES[key]
        
    # modify the value to that given in **kwargs
    for kw, val in kwargs.items():
        if kw in classifier_params:
            vals_classifier[kw] = val
        else :
            raise ValueError(str(kw) + " in argument is not a parameter on which " + str(classifier._name) + " has trained. Available parameters are :" + str(classifier_params))

    # compute the reduced value of the parameter, in the range [0, 1]
    u_vals_classifier = np.zeros(len(classifier_params))

    for ikey, key in enumerate(classifier_params):
        

        max = np.max(classifier._params_range[key])
        min = np.min(classifier._params_range[key])
        val = vals_classifier[key]

        u_vals_classifier[ikey] =  (val - min) / (max - min)

    res_classifier = np.argmax(classifier.predict(np.array([u_vals_classifier]))[0])

    # 0 reionized
    # 1 valid but not reionized
    # 2 not valid
    
    return res_classifier
    


def predict_regressor(regressor = None, **kwargs):
    
    if regressor is None:
        regressor = Regressor.load(".", "DefaultRegressor")
    
    # all parameters the neural network have been trained on
    regressor_params = regressor._params_list

    # predefined default values for most common parameters
    vals_regressor = {}
    for key in regressor_params:
        vals_regressor[key] = DEFAULT_VALUES[key]
        
    # modify the value to that given in **kwargs
    for kw, val in kwargs.items():
        if kw in regressor_params:
            vals_regressor[kw] = val
        else :
            raise ValueError(str(kw) + " in argument is not a parameter on which " + str(regressor._name) + " has trained. Available parameters are :" + str(regressor_params))

    # compute the reduced value of the parameter, in the range [0, 1]
    u_vals_regressor = np.zeros(len(regressor_params))

    for ikey, key in enumerate(regressor_params):
        
        max = np.max(regressor._params_range[key])
        min = np.min(regressor._params_range[key])
        val = vals_regressor[key]

        u_vals_regressor[ikey] =  (val - min) / (max - min)

    res_regressor = regressor.predict(np.array([u_vals_regressor]))[0]

    # 0 reionized
    # 1 valid but not reionized
    # 2 not valid
    
    return res_regressor
    


# predict the evolution of xHII
def predict_xHII(classifier = None, regressor = None, **kwargs):

    res_classifier = predict_classifier(classifier, **kwargs)
    
    if res_classifier == 2:
        return -1
    
    res_regressor = predict_regressor(regressor, **kwargs)

    return res_regressor


# work in progress
def predict_tau(classifier = None, regressor = None, **kwargs)
    
    res_classifier = predict_classifier(classifier, **kwargs)
    
    if res_classifier == 2:
        return -1
    
    res_regressor = predict_regressor(regressor, **kwargs)

    return None