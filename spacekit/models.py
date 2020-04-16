# ********* starskøpe.spacekit.models ********* #

import numpy as np
from sklearn.preprocessing import FunctionTransformer

# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #
class process:

    @staticmethod
    def fTransformer(X, func=None, fit=True, inverse=False, validate=True):
        """
        returns `transformer`

        + log1p : applies a log transformation
        >>> func=np.log1p(X)
        """
        from sklearn.preprocessing import FunctionTransformer

        if func = 'log1p':
            TF = 
            
        transformer = FunctionTransformer(TF, validate=True)
        return X_transformed


#     .. _function_transformer:

# Custom transformers
# ===================

# Often, you will want to convert an existing Python function into a transformer
# to assist in data cleaning or processing. You can implement a transformer from
# an arbitrary function with :class:`FunctionTransformer`. For example, to build
# a transformer that applies a log transformation in a pipeline, do::

#     >>> import numpy as np
#     >>> from sklearn.preprocessing import FunctionTransformer
#     >>> transformer = FunctionTransformer(np.log1p, validate=True)
#     >>> X = np.array([[0, 1], [2, 3]])
#     >>> transformer.transform(X)
#     array([[0.        , 0.69314718],
#            [1.09861229, 1.38629436]])

# You can ensure that ``func`` and ``inverse_func`` are the inverse of each other
# by setting ``check_inverse=True`` and calling ``fit`` before
# ``transform``. Please note that a warning is raised and can be turned into an
# error with a ``filterwarnings``::

#   >>> import warnings
#   >>> warnings.filterwarnings("error", message=".*check_inverse*.",
#   ...                         category=UserWarning, append=False)

# For a full code example that demonstrates using a :class:`FunctionTransformer`
# to do custom feature selection,
# see :ref:`sphx_glr_auto_examples_preprocessing_plot_function_transformer.py`



    # NUMPY_TRAIN_TEST_SPLIT
    @staticmethod
    def numpy_train_test_split(data_folder, train_set, test_set):
        """
        create target classes for training and test data using numpy
        returns x_train, y_train, x_test, y_test
        """
        import numpy as np
        
        train = np.loadtxt(data_folder+train_set, skiprows=1, delimiter=',')
        x_train = train[:, 1:]
        y_train = train[:, 0, np.newaxis] - 1.
        
        test = np.loadtxt(data_folder+test_set, skiprows=1, delimiter=',')
        x_test = test[:, 1:]
        y_test = test[:, 0, np.newaxis] - 1.
        
        return x_train, y_train, x_test, y_test


    # ZERO_SCALER
    @staticmethod
    def zero_scaler(x_train, x_test):
        """
        Scales each observation of an array to zero mean and unit variance.
        Takes array for train and test data separately.
        """
        import numpy as np
            
        x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / 
            np.std(x_train, axis=1).reshape(-1,1))
        
        x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / 
                np.std(x_test, axis=1).reshape(-1,1))
    
        return x_train, x_test


    # TIME_FILTER
    @staticmethod
    def time_filter(x_train, x_test, step_size=None, axis=2):
        """
        Adds an input corresponding to the running average over a set number
        of time steps. This helps the neural network to ignore high frequency 
        noise by passing in a uniform 1-D filter and stacking the arrays. 
        
        **ARGS
        step_size: integer, # timesteps for 1D filter. defaults to 200
        axis: which axis to stack the arrays
        """
        import numpy as np
        from scipy.ndimage.filters import uniform_filter1d
        
        if step_size is None:
            step_size=200
        
        train_filter = uniform_filter1d(x_train, axis=1, size=step_size)
        test_filter = uniform_filter1d(x_test, axis=1, size=step_size)
        
        x_train = np.stack([x_train, train_filter], axis=2)
        x_test = np.stack([x_test, test_filter], axis=2)

        return x_train, x_test

    @staticmethod
    # NUDGE_DATASET
    # variation of sklearn's `nudge_dataset` function 
    def nudge(X, Y, ):
        """
        This produces a dataset 5 times bigger than the original one,
        by moving the 8x8 images in X around by 1px to left, right, down, up
        """



        direction_vectors = [
            [[0, 1, 0],
            [0, 0, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [1, 0, 0],
            [0, 0, 0]],

            [[0, 0, 0],
            [0, 0, 1],
            [0, 0, 0]],

            [[0, 0, 0],
            [0, 0, 0],
            [0, 1, 0]]]

        def shift(x, w):
            return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

        X = np.concatenate([X] +
                        [np.apply_along_axis(shift, 1, X, vector)
                            for vector in direction_vectors])
        Y = np.concatenate([Y for _ in range(5)], axis=0)
        return X, Y

# 
#Load Data
# digits = datasets.load_digits()
# X = np.asarray(digits.data, 'float32')

# X, Y = nudge_dataset(X, digits.target)

# X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling
# 
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# 
#Models we will use
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000, multi_class='multinomial')
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm_features_classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
