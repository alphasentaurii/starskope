# ********* starskøpe.spacekit.models ********* #

import numpy as np
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split

# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #
class Process:

    # TRANSFORM SPLIT
    @staticmethod
    def transform_split(dataset, test_size=0.2):
        """
        create target classes for training and test data
        returns x_train, y_train, x_test, y_test
        """
        import numpy as np
        from sklearn.model_selection import train_test_split

        X = np.asarray(dataset.iloc[:,1:])
        y = np.asarray(dataset.iloc[:,0]) - 1
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        return X_train, X_test, y_train, y_test 



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

    

# 
# 
# Setting up
# 
# def nudge_dataset(X, Y):
    # """
    # This produces a dataset 5 times bigger than the original one,
    # by moving the 8x8 images in X around by 1px to left, right, down, up
    # """
    # direction_vectors = [
        # [[0, 1, 0],
        #  [0, 0, 0],
        #  [0, 0, 0]],
# 
        # [[0, 0, 0],
        #  [1, 0, 0],
        #  [0, 0, 0]],
# 
        # [[0, 0, 0],
        #  [0, 0, 1],
        #  [0, 0, 0]],
# 
        # [[0, 0, 0],
        #  [0, 0, 0],
        #  [0, 1, 0]]]
# 
    # def shift(x, w):
        # return convolve(x.reshape((9, 9)), mode='constant', weights=w).ravel()
# 
    # X = np.concatenate([X] +
                    #    [np.apply_along_axis(shift, 1, X, vector)
                        # for vector in direction_vectors])
    # Y = np.concatenate([Y for _ in range(5)], axis=0)
    # return X, Y
# 

# Models we will use
# logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000,
                                        #    multi_class='multinomial')
# rbm = BernoulliRBM(random_state=0, verbose=True)
# 
# rbm_features_classifier = Pipeline(
    # steps=[('rbm', rbm), ('logistic', logistic)])
# 




#     # TRANSFORMER
#     ### WORK IN PROGRESS
#     @staticmethod
#     def Transformer(X, func=None, inverse_func=None, validate=None, accept_sparse=False, 
#                     pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None):
        
#         """
#         FunctionTransformer
#         default: returns `XF` transformer ready for fitting
#         fit : (boolean) transforms,fits and returns data Xt 
#         inverse: (bool) runs inverse transformation, returns Xi

#         Parameters
#         func : callable, optional default=None

#             The callable to use for the transformation. This will be passed  
#             the same arguments as transform, with args and kwargs forwarded.  
#             If func is None, then func will be the identity function.  
#         inverse_func : callable, optional default=None

#             The callable to use for the inverse transformation. This will be  
#             passed the same arguments as inverse transform, with args and  
#             kwargs forwarded. If inverse_func is None, then inverse_func  
#             will be the identity function.  
        
#         class FunctionTransformer(func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None)
#         Constructs a transformer from an arbitrary callable.

#          This is useful for stateless transformations such as taking the log of frequencies, doing custom scaling, etc.

#         Note: If a lambda is used as the function, then the resulting transformer will not be pickleable.


#         validate : bool, optional default=True

#             Indicate that the input X array should be checked before calling  
#             `func`. The possibilities are:  

#             - If False, there is no input validation.  
#             - If True, then X will be converted to a 2-dimensional NumPy array or  
#             sparse matrix. If the conversion is not possible an exception is  
#             raised.  

#             `validate=True` as default will be replaced by  
#             `validate=False` in 0.22.  
#         accept_sparse : boolean, optional

#             Indicate that func accepts a sparse matrix as input. If validate is  
#             False, this has no effect. Otherwise, if accept_sparse is false,  
#             sparse matrix inputs will cause an exception to be raised.  
#         pass_y : bool, optional default=False

#             Indicate that transform should forward the y argument to the  
#             inner callable.  
#         check_inverse : bool, default=True

#         Whether to check that or `func` followed by `inverse_func` leads to  
#         the original inputs. It can be used for a sanity check, raising a  
#         warning when the condition is not fulfilled.  
#         kw_args : dict, optional

#             Dictionary of additional keyword arguments to pass to func.  
#         inv_kw_args : dict, optional

#             Dictionary of additional keyword arguments to pass to inverse_func. 

# #         """
#         from sklearn.preprocessing import FunctionTransformer

#         # functions = dict('rbf'=RBFSampler(),
#         #              'fourier'=np.fft.rfft(),
#         #              'SkewedChi2',
#         #              'log1p'=np.log1p(X))

#         for k,v in functions.items():
#             if k == func:
#                 print(f'Building transformer: {k}')
#                 transformer = functions[func]
#             else:
#                 print(f"Couldn't find Transformer named {func}")
#                 print(f"Try one of these instead: {functions.items()}")
        
#         if check_inverse is True:
#             Xt = transformer.fit(X)
#         if fit:
#             Xt = transformer.fit(X)
#             Xt = transformer.transform(Xt)
#         else: 
#             Xt = transformer.transform(X)

#         return transformer, Xt 


# #     .. _function_transformer:

# # Custom transformers
# # ===================

# # Often, you will want to convert an existing Python function into a transformer
# # to assist in data cleaning or processing. You can implement a transformer from
# an arbitrary function with :class:`FunctionTransformer`. For example, to build
# a transformer that applies a log transformation in a pipeline, do::

#     >>> import numpy as np
#     >>> from sklearn.preprocessing import FunctionTransformer
#     >>> transformer = FunctionTransformer(np.log1p, validate=True)
#     >>> X = np.array([[0, 1], [2, 3]])
#     >>> 
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

