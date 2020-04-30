# ********* starskøpe.spacekit.models ********* #

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import models, layers, optimizers
from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #

class Keras:

    @staticmethod
    def build_cnn(X_train, X_test, y_train, y_test, kernel_size=11, activation='relu', input_shape=None, strides=4, optimizer=Adam, 
                    learning_rate=1e-5, loss='binary_crossentropy', 
                    metrics=['accuracy'], validation_data=None, 
                    verbose=2, epochs=5, steps_per_epoch=None, 
                    batch_size=32):
        """
        Builds, compiles and fits a linear CNN using Keras API

        """
        import keras
        from keras.utils.np_utils import to_categorical
        # from keras.preprocessing.text import Tokenizer
        from keras import models, layers, optimizers
        from keras.models import Sequential, Model
        from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
        BatchNormalization, Input, concatenate, Activation
        from keras.optimizers import Adam
        from keras.wrappers.scikit_learn import KerasClassifier
        from sklearn.model_selection import cross_val_score
        #setting defaults
        if input_shape is None:
            input_shape = X_train.shape[1:]
        if kernel_size is None:
            kernel_size=11
        if activation is None:
            activation='relu'
        if strides is None:
            strides = 4
        if learning_rate is None:
            learning_rate = 1e-5
        if loss is None:
            loss='binary_crossentropy'
        if metrics is None:
            metrics=['accuracy']
        if verbose is None:
            verbose=2
        if epochs is None:
            epochs = 5
        if validation_data is None:
            validation_data=(X_test, y_test)
        if steps_per_epoch is None:
            steps_per_epoch = (X_train.shape[1]//batch_size)
        if batch_size is None:
            batch_size = 32
        

        model=Sequential()
        #layer1: takes input shape
        model.add(Conv1D(filters=8, kernel_size=kernel_size, 
                        activation=activation, input_shape=input_shape))
        model.add(MaxPool1D(strides=strides))
        model.add(BatchNormalization())
        #layer2
        model.add(Conv1D(filters=16, kernel_size=kernel_size, 
                        activation=activation))
        model.add(MaxPool1D(strides=strides))
        model.add(BatchNormalization())
        #layer3
        model.add(Conv1D(filters=32, kernel_size=kernel_size, 
                        activation=activation))
        model.add(MaxPool1D(strides=strides))
        model.add(BatchNormalization())
        #layer4
        model.add(Conv1D(filters=64, kernel_size=kernel_size, 
                        activation=activation))
        model.add(MaxPool1D(strides=strides))
        model.add(Flatten())
        print("FULL CONNECTION")
        
        # Full Connection
        model.add(Dropout(0.5))
        model.add(Dense(64, activation=activation))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation=activation))

        # sigmoid function
        model.add(Dense(1, activation='sigmoid'))

        print("COMPILING")   
        ##### COMPILE #####
        model.compile(optimizer=optimizer(learning_rate), loss=loss, 
                    metrics=metrics)
        

        def batch_maker(X_train, y_train, batch_size=batch_size):
            """
            Gives equal number of positive and negative samples rotating randomly
            
            generator: A generator or an instance of `keras.utils.Sequence`
                
            The output of the generator must be either
            - a tuple `(inputs, targets)`
            - a tuple `(inputs, targets, sample_weights)`.

            This tuple (a single output of the generator) makes a single
            batch. Therefore, all arrays in this tuple must have the same
            length (equal to the size of this batch). Different batches may have 
            different sizes. 

            For example, the last batch of the epoch
            is commonly smaller than the others, if the size of the dataset
            is not divisible by the batch size.
            The generator is expected to loop over its data
            indefinitely. An epoch finishes when `steps_per_epoch`
            batches have been seen by the model.
            
            """
            import numpy
            import random
            # hb: half-batch
            hb = batch_size // 2
            
            # Returns a new array of given shape and type, without initializing.
            # x_train.shape = (5087, 3197, 2)
            xb = np.empty((batch_size, X_train.shape[1], X_train.shape[2]), dtype='float32')
            
            #y_train.shape = (5087, 1)
            yb = np.empty((batch_size, y_train.shape[1]), dtype='float32')
            
            pos = np.where(y_train[:,0] == 1.)[0]
            neg = np.where(y_train[:,0] == 0.)[0]

            # rotating each of the samples randomly
            while True:
                np.random.shuffle(pos)
                np.random.shuffle(neg)
            
                xb[:hb] = X_train[pos[:hb]]
                xb[hb:] = X_train[neg[hb:batch_size]]
                yb[:hb] = y_train[pos[:hb]]
                yb[hb:] = y_train[neg[hb:batch_size]]
            
                for i in range(batch_size):
                    size = np.random.randint(xb.shape[1])
                    xb[i] = np.roll(xb[i], size, axis=0)
            
                yield xb, yb


        print("FITTING")
        history = model.fit_generator(batch_maker(X_train, y_train, batch_size),
                                    validation_data=validation_data, 
                                    verbose=verbose, epochs=epochs, 
                                    steps_per_epoch=steps_per_epoch)

        
        return model,history 

