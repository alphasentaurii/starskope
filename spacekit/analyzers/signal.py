
# ********* starskøpe.Spacekit.modelMetrics ********* #
"""
model_metriks() 
helper functions for generating predictions, 
calculating scores, and evaluating a machine learning model.
"""
# -----------------
# STATIC CLASS METHODS 
# -----------------
# * predictions 
#   get_preds()
#
# * Plots
#   plot_keras_history()
#   plot_confusion_matrix()
#   roc_plots()
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #

import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt

### Flux Signals
### 
class flux:

    @staticmethod
    def signal_plots(signal, label_col=None, classes=None, 
                    class_names=None, figsize=(15,5), y_units=None, x_units=None):
        """
        Plots scatter and line plots of time series signal values.  
        
        **ARGS
        signal: pandas series or numpy array
        label_col: name of the label column if using labeled pandas series
            -use default None for numpy array or unlabeled series.
            -this is simply for customizing plot Title to include classification    
        classes: (optional- req labeled data) tuple if binary, array if multiclass
        class_names: tuple or array of strings denoting what the classes mean
        figsize: size of the figures (default = (15,5))
        
        ******
        
        Ex1: Labeled timeseries passing 1st row of pandas dataframe
        > first create the signal:
        star_signal_alpha = x_train.iloc[0, :]
        > then plot:
        star_signals(star_signal_alpha, label_col='LABEL',classes=[1,2], 
                    class_names=['No Planet', 'Planet']), figsize=(15,5))
        
        Ex2: numpy array without any labels
        > first create the signal:
        
        >then plot:
        star_signals(signal, figsize=(15,5))
        
        ######
        TODO: 
        -`signal` should take an array rather than pdseries
        -could allow either array or series to be passed, conv to array if series 
        ######
        """
        
        # pass None to label_col if unlabeled data, creates generic title
        if label_col is None:
            label = None
            title_scatter = "Scatterplot of Star Flux Signals"
            title_line = "Line Plot of Star Flux Signals"
            color='black'
            
        # store target column as variable 
        elif label_col is not None:
            label = signal[label_col]
            # for labeled timeseries
            if label == 1:
                cn = class_names[0]
                color='red'

            elif label == 2:
                cn = class_names[1] 
                color='blue'
        #create appropriate title acc to class_names    
            title_scatter = f"Scatterplot for Star Flux Signal: {cn}"
            title_line = f"Line Plot for Star Flux Signal: {cn}"
        
        # Set x and y axis labels according to units
        # if the units are unknown, we will default to "Flux"
        if y_units == None:
            y_units = 'Flux'
        else:
            y_units = y_units
        # it is assumed this is a timeseries, default to "time"   
        if x_units == None:
            x_units = 'Time'
        else:
            x_units = x_units
        
        # Scatter Plot 
        
        plt.figure(figsize=figsize)
        plt.scatter(pd.Series([i for i in range(1, len(signal))]), 
                    signal[1:], marker=4, color=color)
        plt.ylabel(y_units)
        plt.xlabel(x_units)
        plt.title(title_scatter)
        plt.show();

        # Line Plot
        plt.figure(figsize=figsize)
        plt.plot(pd.Series([i for i in range(1, len(signal))]), 
                signal[1:], color=color)
        plt.ylabel(y_units)
        plt.xlabel(x_units)
        plt.title(title_line)
        plt.show();



# check for change in relative flux vs position 
    # also subtract off bad pixels 
    #dflux = pdcflux[~bad_data]/np.nanmedian(pdcflux[~bad_data])






### IN PROG 



    # # SIGNAL_SPLITS #### WORK IN PROGRESS
    # @staticmethod
    # def signal_splits(signal, figsize=(21,11), **subplot_kws):

    #     fig, axes = plt.subplots(ncols=3, figsize=figsize, **subplot_kws)
    #     axes = axes.flatten()
# 