import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class NeuralNetwork:

    def __init__(self, model, name):
        self._model = model
        self._name = name
        self._history = {'loss' : np.zeros(0), 'val_loss' : np.zeros(0)}

    def _compile(self, loss, *, optimizer = tf.keras.optimizers.AdamW(learning_rate=1e-3), **kwargs):
        self._history = {'loss' : np.zeros(0), 'val_loss' : np.zeros(0)}
        self._model.compile(optimizer, loss=loss, **kwargs)

    def _train(self, x, y, *, epochs = 500, validation_split=0.1, verbose='auto', batch_size=32, **kwargs):

        # train the model according to the specification wanted by the user
        new_history = self._model.fit(x, y, epochs = epochs, validation_split = validation_split, verbose = verbose, batch_size = batch_size, **kwargs)

        # combine the histories
        for key in ['loss', 'val_loss']:
            self._history[key] = np.concatenate((self._history[key], new_history.history[key]), axis=0)

    # -------------------------------
    # redefine very simple functions from keras in this Class for convenience

    def save(self, path = ".") -> None:
        self._model.save(path + "/" + self._name + '.keras')
    
    def predict(self, x):
        self._model.predict(x)

    # -------------------------------
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
        self._train(db._x_train, db._y_train_class, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultClassifier"):
        model = keras.models.load_model(path + "/" + name + '.keras')
        return Classifier(model = model, name = name, check_name = False)
    



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
        self._train(db._x_train_valid, db._y_train_valid, epochs=epochs, validation_split=validation_split, verbose=verbose, batch_size=batch_size, **kwargs )
    
    @classmethod
    def load(cls, path = ".", name = "DefaultRegressor"):
        model = keras.models.load_model(path + "/" + name + '.keras')
        return Regressor(model, name; check_name=False)
    