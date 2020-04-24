
# ********* starskøpe.spacekit.metriks ********* #
""" 
helper functions for generating predictions, 
calculating scores, and evaluating a machine learning model.

TODO       
- save metriks to textfile/pickle objs and/or dictionaries



"""
# -----------------
# STATIC CLASS METHODS 
# -----------------
# * predictions 
#   get_preds()
#
# * Plots
#   keras_history()
#   plot_confusion_matrix()
#   roc_plots()
# ********* /\/\/\/\/\/\/\/\/\/\/\/\/\/\/\/ ********* #

import pandas as pd
import numpy as np
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, roc_curve, roc_auc_score, accuracy_score, jaccard_score

#PREDICTIONS
### GET_PREDS()
class Predictor():

    @staticmethod
    def get_preds(x_test,y_test,store=False,model=None,verbose=False,**kwargs):
    #y_true = (y_test[:, 0] + 0.5).astype("int") # flatten and make integer
    #y_hat = model.predict(x_test)[:,0] 
    # import pandas as pd
    # import numpy as np
    # from sklearn import metrics
    # from sklearn.metrics import accuracy_score, recall_score
    if model is None:
        model=model
        # class predictions 
        y_true = y_test.flatten()
        y_hat = model.predict(x_test)
        y_pred = model.predict_classes(x_test).flatten() 
        preds = pd.Series(y_pred).value_counts(normalize=False)
        
        if verbose:
            print(f"y_pred:\n {preds}")
            print("\n")

        if store:
            # store in dict
            pre_dict = dict('yt'=y_true, 'yh'=y_hat,'yp'=y_pred,'preds'=preds)
            
            # save to textfile/pickle obj?

        return y_true, y_pred

### PLOTS
class Plotter:

    ### PLOT_KERAS_HISTORY()
    ## Based on code by James Irving PhD / Flatiron School Study Group Notes
    ## James Irving: 
    @staticmethod
    def keras_history(model=None,history=None, figsize=(21,11),subplot_kws={}):
        """
        returns fig
        side by side sublots of training val accuracy and loss (left and right respectively)
        minor adjustments made to James Irving's code include;
        -'model' which model iteration, eg m1
        -`history` arg set to None so we can pass in a specific model's history, e.g. `h1`
        -returns `fig`
        -increased default figsize arg
        -changed figsize=figsize in first initialization (figsize=(10,4) was hardcoded)
        """
        if model is None:
            model=model
        
        if hasattr(history,'history'):
            history=history.history
        figsize=figsize
        subplot_kws={}

        acc_keys = list(filter(lambda x: 'acc' in x,history.keys()))
        loss_keys = list(filter(lambda x: 'loss' in x,history.keys()))

        fig, axes = plt.subplots(ncols=2, figsize=figsize, **subplot_kws)
        axes = axes.flatten()

        y_labels= ['Accuracy','Loss']
        for a, metric in enumerate([acc_keys,loss_keys]):
            for i in range(len(metric)):
                ax = pd.Series(history[metric[i]],
                            name=metric[i]).plot(ax=axes[a],label=metric[i])

        [ax.legend() for ax in axes]
        [ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True)) for ax in axes]
        [ax.set(xlabel='Epochs') for ax in axes]
        plt.suptitle('Training Results', y=1.01)
        plt.tight_layout()
        plt.show()

        return fig

    # FUSION_MATRIX()
    @staticmethod
    def fusion_matrix(matrix, classes=None, normalize=False, title='Fusion Matrix', cmap='Blues',
        print_raw=False,figsize=(7,8)): 
        """
        FUSION MATRIX
        -------------
        matrix: can pass in matrix or a tuple (ytrue,ypred) to create on the fly 
        classes: class names for target variables

        *** Don't be Confused! (what is a Fusion Matrix?)
        *** 
        *** Well, it's a confusion matrix, but...
        *** 
        *** This function is designed to make the classifier's predictions easy to interpret
        *** CMs of the past were quite confusing... Fusion  is bnet
        *** Fusion makes more sense because in a way, we're smashing all our
        *** little atoms of data together and measuring the energy of outputs
        *** (as far as neural networks go this is literally true)
        ***
        *** So...
        *** This is a Fusion Matrix!

        """
        from sklearn import metrics                       
        from sklearn.metrics import confusion_matrix #ugh
        import itertools
        import numpy as np
        import matplotlib.pyplot as plt
        
        # make matrix if tuple passed to matrix:
        if isinstance(matrix, tuple):
            y_true = matrix[0].copy()
            y_pred = matrix[1].copy()
            
            if y_true.ndim>1:
                y_true = y_true.argmax(axis=1)
            if y_pred.ndim>1:
               y_pred = y_pred.argmax(axis=1)
            fusion = metrics.confusion_matrix(y_true, y_pred)
        else:
            fusion = matrix
        
        # INTEGER LABELS
        if classes is None:
            classes=list(range(len(matrix)))

        #NORMALIZING
        # Check if normalize is set to True
        # If so, normalize the raw confusion matrix before visualizing
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt='.2f'
        else:
            fmt='d'
        
        # PLOT
        fig, ax = plt.subplots(figsize=(10,10))
        # mask = np.zeros_like(matrix, dtype=np.bool)
        # idx = np.true_indices_from(mask)
        # mask[idx] = True
        plt.imshow(cm, cmap=cmap, aspect='equal')
        
        # Add title and axis labels 
        plt.title('FUSION Matrix') 
        plt.ylabel('TRUE') 
        plt.xlabel('PRED')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        ax.set_ylim(len(cm), -.5,.5)
        
        # Text formatting
        fmt = '.2f' if normalize else 'd'
        # Add labels to each cell
        thresh = cm.max() / 2.
        # iterate thru matrix and append labels  
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    size=14, weight='bold')
        
        # Add a legend
        plt.colorbar()
        plt.show() 
        return cm, fig


    @staticmethod
    def roc_plots(y_test, y_hat):
        # from sklearn import metrics
        # from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
        #y_true = (y_test[:, 0] + 0.5).astype("int") 
        y_true=y_test.flatten()
          
        fpr, tpr, thresholds = roc_curve(y_true, y_hat) 
        fpr, tpr, thresholds = roc_curve(y_true, y_hat)

        # Threshold Cutoff for predictions
        crossover_index = np.min(np.where(1.-fpr <= tpr))
        crossover_cutoff = thresholds[crossover_index]
        crossover_specificity = 1.-fpr[crossover_index]
        #print("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
        
        fig, ax1, ax2 = plt.figure(ncols=2, figsize=(15,5))
        ax1.plot(thresholds, 1.-fpr)
        ax1.plot(thresholds, tpr)
        plt.title("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
        plt.show()

        ax2.plot(fpr, tpr)
        ax2.title("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
        plt.show()
        
        roc = roc_auc_score(y_true,y_hat)
        print("ROC_AUC SCORE:",roc)
        #print("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
        return roc


    ### EVALUATE_MODEL ###
    @staticmethod
    def score(x_test, y_test, model=None, get_preds=False, fusion_matrix=True, 
        ROC=True, hist=True, **kwargs):
        """
        evaluates model predictions and stores the output in a dictionary `score`
        """
        from spacekit.metriks import get_preds, fusion_matrix, keras_history, roc_plots
        from sklearn import metrics
        from sklearn.metrics import jaccard_score,confusion_matrix

        score = {}
        if model is None:
            model = model
        if get_preds: 
        # class predictions 
            y_true = y_test.flatten()
            y_pred = model.predict_classes(x_test).flatten() 

        y_true, y_pred = get_preds(x_test,y_test,model,**kwargs)
        summary = model.summary

        # save in dictionary
        model_idx = {'model':model, 'summary':summary,'y_true':y_true,'y_pred':y_pred}
        score['model']=model_idx

        # CLASSIFICATION REPORT
        num_dashes=20
        print('\n')
        print('---'*num_dashes)
        print('\tCLASSIFICATION REPORT:')
        print('---'*num_dashes)
        # generate report
        report = metrics.classification_report(y_true,y_pred)
        
        # calculate scores
        jacc = jaccard_score(y_test, y_pred)
        acc = accuracy_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)

        # save in dictionary
        scores = {}
        scores['jaccard'] = jacc
        scores['accuracy'] = acc
        scores['recall'] = rec

        print('Jaccard Similarity Score:',jacc)
        print('\nAccuracy Score:', acc)
        print('\nRecall Score:', rec)

        # save in dictionary
        score_idx = {'report':report,'scores':scores}
        score['report'] = score_idx

        # FUSION MATRIX
        if fusion_matrix:
            #fusion = confusion_matrix(y_true, y_pred, labels=[0,1])
            
            # Plot confusion matrix
            fusion = fusion_matrix(matrix=(y_true, y_pred), classes=['No Planet', 'Planet')
            plt.show()

            return fig


        # ROC Area Under Curve
        if roc:
            ROC = roc_plots(y_test, y_pred)

        
        #Plot Model Training Results (PLOT KERAS HISTORY)
        HIST = plot_keras_history(history)
        
        plot_idx = {'Fusion':fusion,'HIST':HIST,'ROC':ROC}

        score['plots'] = plot_idx

        return score

