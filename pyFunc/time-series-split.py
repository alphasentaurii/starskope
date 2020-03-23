

# TODO: incorporate some of below code to build version of sklearn's _BaseKFold

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html


from sklearn.model_selection._split import _BaseKFold


class TimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.

    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    n_splits : int, default=3
        Number of splits. Must be at least 1.

    Examples
    --------
    >>> from sklearn.model_selection import TimeSeriesSplit
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> tscv = TimeSeriesSplit(n_splits=3)
    >>> print(tscv)  # doctest: +NORMALIZE_WHITESPACE
    TimeSeriesSplit(n_splits=3)
    >>> for train_index, test_index in tscv.split(X):
    ...    print("TRAIN:", train_index, "TEST:", test_index)
    ...    X_train, X_test = X[train_index], X[test_index]
    ...    y_train, y_test = y[train_index], y[test_index]
    TRAIN: [0] TEST: [1]
    TRAIN: [0 1] TEST: [2]
    TRAIN: [0 1 2] TEST: [3]

    Notes
    -----
    The training set has size ``i * n_samples // (n_splits + 1)
    + n_samples % (n_splits + 1)`` in the ``i``th split,
    with a test set of size ``n_samples//(n_splits + 1)``,
    where ``n_samples`` is the number of samples.
    """
    def __init__(self, n_splits=3):
        super(TimeSeriesSplit, self).__init__(n_splits,
                                              shuffle=False,
                                              random_state=None)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like, with shape (n_samples,), optional
            Always ignored, exists for compatibility.

        Returns
        -------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        if n_folds > n_samples:
            raise ValueError(
                ("Cannot have number of folds ={0} greater"
                 " than the number of samples: {1}.").format(n_folds,
                                                             n_samples))
        indices = np.arange(n_samples)
        test_size = (n_samples // n_folds)
        test_starts = range(test_size + n_samples % n_folds,
                            n_samples, test_size)
        for test_start in test_starts:
            yield (indices[:test_start],
                   indices[test_start:test_start + test_size])





########

## IMPORT CUSTOM CAPSTONE FUNCTIONS
import functions_combined_BEST as ji
import functions_io as io

from functions_combined_BEST import ihelp, ihelp_menu,\
reload, inspect_variables

## IMPORT MY PUBLISHED PYPI PACKAGE 
import bs_ds as  bs
from bs_ds.imports import *

## IMPORT CONVENIENCE FUNCTIONS
from pprint import pprint
import qgrid
import json

# Import plotly and cufflinks for iplots
import plotly
import cufflinks as cf
from plotly import graph_objs as go
from plotly.offline import iplot
cf.go_offline()

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

#Set pd.set_options for tweet visibility
pd.set_option('display.max_colwidth',100)
pd.set_option('display.max_columns',50)



######### LOAD IN TESTDATA
df_combined = pd.read_csv('data/_combined_stock_data_with_tweet_preds.csv', index_col=0,parse_dates=True)
model_col_list = ['price','price_shifted', 'ma7', 'ma21', '26ema', '12ema', 'MACD', '20sd', 'upper_band','lower_band', 'ema', 'momentum',
                  'has_tweets','num_tweets','case_ratio', 'compound_score','pos','neu','neg','sentiment_class',
                  'pred_classes','pred_classes_int','total_favorite_count','total_retweet_count']

df_combined = ji.set_timeindex_freq(df_combined,fill_nulls=False)
df_combined.sort_index(inplace=True, ascending=True)
df_combined['price_shifted'] =df_combined['price'].shift(-1)

df_to_model = df_combined[model_col_list].copy()
del df_combined
df_to_model.head()


##### DEF BlockTimeSeriesSPlit()


class BlockTimeSeriesSplit(_BaseKFold): #sklearn.model_selection.TimeSeriesSplit):
    """A variant of sklearn.model_selection.TimeSeriesSplit that keeps train_size and test_size
    constant across folds. 
    Requires n_splits,train_size,test_size. train_size/test_size can be integer indices or float ratios """
    def __init__(self, n_splits=5,train_size=None, test_size=None, step_size=None, method='sliding'):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        if 'sliding' in method or 'normal' in method:
            self.method = method
        else:
            raise  Exception("Method may only be 'normal' or 'sliding'")
        
    def split(self,X,y=None, groups=None):
        import math 
        method = self.method
        ## Get n_samples, trian_size, test_size, step_size
        n_samples = len(X)
        test_size = self.test_size
        train_size =self.train_size
      
                
        ## If train size and test sze are ratios, calculate number of indices
        if train_size<1.0:
            train_size = math.floor(n_samples*train_size)
        
        if test_size <1.0:
            test_size = math.floor(n_samples*test_size)
            
        ## Save the sizes (all in integer form)
        self._train_size = train_size
        self._test_size = test_size
        
        ## calcualte and save k_fold_size        
        k_fold_size = self._test_size + self._train_size
        self._k_fold_size = k_fold_size    
        

    
        indices = np.arange(n_samples)
        
        ## Verify there is enough data to have non-overlapping k_folds
        if method=='normal':
            import warnings
            if n_samples // self._k_fold_size <self.n_splits:
                warnings.warn('The train and test sizes are too big for n_splits using method="normal"\n\
                switching to method="sliding"')
                method='sliding'
                self.method='sliding'
                              
                  
            
        if method=='normal':

            margin = 0
            for i in range(self.n_splits):

                start = i * k_fold_size
                stop = start+k_fold_size

                ## change mid to match my own needs
                mid = int(start+self._train_size)
                yield indices[start: mid], indices[mid + margin: stop]
        

        elif method=='sliding':
            
            step_size = self.step_size
            if step_size is None: ## if no step_size, calculate one
                ## DETERMINE STEP_SIZE
                last_possible_start = n_samples-self._k_fold_size #index[-1]-k_fold_size)\
                step_range =  range(last_possible_start)
                step_size = len(step_range)//self.n_splits
            self._step_size = step_size
                
            
            for i in range(self.n_splits):
                if i==0:
                    start = 0
                else:
                    start = prior_start+self._step_size #(i * step_size)

                stop =  start+k_fold_size            
                ## change mid to match my own needs
                mid = int(start+self._train_size)
                prior_start = start
                yield indices[start: mid], indices[mid: stop]





split_ts = BlockTimeSeriesSplit(n_splits = 5, train_size=0.3,test_size=0.1,method='sliding')#train_size=840, test_size=10*7)
master_date_index=df_to_model.index.to_series()
n=0
dashes = '---'*20
for train_index, test_index in split_ts.split(df_to_model):  
    
    print(f'\n{dashes}\nsplit {n}')
    train_date_index = master_date_index.iloc[train_index]
    test_date_index = master_date_index.iloc[test_index]
    ji.index_report(train_date_index)
    ji.index_report(test_date_index)
    n+=1





    ########## Use Pipeline to prepare modeling

    ## Using ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit,train_test_split, GridSearchCV,cross_val_score,KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer, make_column_transformer 
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


import xgboost as xgb
target_col= 'price_shifted'
## Sort_index
cols_to_drop =['price','pred_classes_int']
cols_to_drop.append(target_col)

# df_to_model.drop(cols_to_drop,axis=1,inplace=True)

features = df_to_model.drop(cols_to_drop, axis=1)
target = df_to_model[target_col]



## Get boolean masks for which columns to use
numeric_cols = features.dtypes=='float'
category_cols = ~numeric_cols
# target_col = df_to_model.columns=='price_shifted'


price_transformer = Pipeline(steps=[
    ('scaler',MinMaxScaler())
])


## define pipeline for preparing numeric data
numeric_transformer = Pipeline(steps=[
#     ('imputer',SimpleImputer(strategy='median')),
    ('scaler',MinMaxScaler())
])

category_transformer = Pipeline(steps=[
#     ('imputer',SimpleImputer(missing_values=np.nan,
#                              strategy='constant',fill_value='missing')),
    ('onehot',OneHotEncoder(handle_unknown='ignore'))
])


## define pipeline for preparing categorical data
preprocessor = ColumnTransformer(remainder='passthrough',
                                 transformers=[
                                     ('num',numeric_transformer, numeric_cols),
                                     ('cat',category_transformer,category_cols)])






# ANOTHER TRANSFORMER--- for getting feature importance

### ADDING MY OWN TRANSFORMATION SO CAN USE FEATUREA IMPROTANCE
df_tf =pd.DataFrame()
num_cols_list = numeric_cols[numeric_cols==True]
cat_cols_list = category_cols[category_cols==True]
# num_cols = df_to_model.columns

for col in df_to_model.columns:
    
    if col in num_cols_list:
        print(f'{col} is numeric')
        vals = df_to_model[col].values
        tf_num = numeric_transformer.fit_transform(vals.reshape(-1,1))
        
        try:
            df_tf[col] = tf_num.flatten()
            print(f"{col} added")
        except:
            print('Error')
            print(tf_num.shape)
#             print(tf_num[:10])
        
    if col in cat_cols_list:
        print(f'{col} is categorical')
#         colnames=[]
#         vals = df_to_model[col].values
#         print(vals.shape)
#         tf_cats = category_transformer.fit_transform(vals.reshape(-1,1))
#         print(tf_cols.shape)
#         print(col,'\n',tf_cats)
        
#         [colnames.append(f"{col}_{i}") for i in range(tf_cats.shape[1])]
#         print(colnames)
        
        df_temp = pd.get_dummies(df_to_model[col])#DataFrame(data=tf_cats[:],index=df_to_model.index)
#         display(df_temp.head())
#         df_temp.columns = 
#         colnames = [for i in range(tf_cols.shape[1])]
        df_tf = pd.concat([df_tf,df_temp],axis=1)

#     ('target',price_transformer,target_col)])
    

# reg = Pipeline(steps=[('preprocessor',preprocessor),
#                      ('regressor',xgb.XGBRegressor(random_state=42))])
df_tf.head()


# def BlockTimeSeriesSplit

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import xgboost as xgb
from xgboost import plot_importance,plot_tree
reg= xgb.XGBRegressor(random_state=42, n_estimators=500)

# split_ts = TimeSeriesSplit(n_splits=5)#,max_train_size=7*5*6)
split_ts = BlockTimeSeriesSplit(n_splits=5,train_size=0.3,test_size=0.1,method='normal')#train_size=7*5*4*3, test_size = 7*10)

results_list=[]
k=0
date_index = features.index.to_series()

for train_index, test_index in split_ts.split(features):  
    
    df_train = features.iloc[train_index]
    df_test = features.iloc[test_index]
    
    y_train = target.iloc[train_index].values
    y_test = target.iloc[test_index].values
    
    train_date_index = date_index.iloc[train_index]
    test_date_index = date_index.iloc[test_index]
    
    ## Fitting preprocessor to training data, transforming both
    preprocessor.fit(features)
    X_train = preprocessor.transform(df_train)
    X_test = preprocessor.transform(df_test)
    
   
        
    reg.fit(X_train, y_train)
    pred = reg.predict(X_test)
    
    
    
    true_train_series = pd.Series(y_train, index=train_date_index,name='true_train_price')
    true_test_series = pd.Series(y_test, index=test_date_index,name='true_test_price')
    pred_price_series = pd.Series(pred,index=test_date_index,name='pred_test_price')#.plot()
    
    df_xgb = pd.concat([true_train_series,true_test_series,pred_price_series],axis=1)


    try:
    
        df_results = ji.evaluate_regression(true_test_series,pred_price_series,show_results=False);
    
    except:
        print(f"Trouble with k={k}")      
    finally:

        fold_dict = {'k':k,
                     'train_index':train_date_index,
                     'test_index':test_date_index,
                     'results':df_results,
                     'df_model':df_xgb,
                     'model':reg}
        results_list.append(fold_dict)
        k+=1
    #     print('Model Score:',reg.score(X_test,y_test))
    

###### PLOTTING

from pandas import plotting
## PLOT ALL TRAIN/TEST SAMPLES ON ONE PLT
for i in range(len(results_list)):
    
    ## get results items 
    k = results_list[i]['k']
    df_model = results_list[i]['df_model'].copy()#
    df_results = results_list[i]['results']

    ## rename columns to identify on same plot
    new_colnames = df_model.columns+f"_{k}"
    df_model.columns  = new_colnames
    
    ## add to df_ind_plots and plot
    df_ind_plots = pd.concat([target.copy(),df_model],axis=1) # reset each time
    

    ## Create fig, axes     #     display(df_results.style.set_caption(f'Fold#{i}'))
    ax1= plt.subplot2grid((1,3),(0,0), colspan=2)
    ax2 =plt.subplot2grid((1,3),(0,2), colspan=1)
    ax2.axis('off')
    fig = plt.gcf()
    
    # Plot data
    df_ind_plots.plot(title=f'Fold #{k}',rot=45,figsize=(12,4),ax=ax1) #table=df_results) 
    # Plot results table
    plotting.table(ax2, colWidths=[0.6, 0.2],data=df_results,loc='right')
    plt.show()   