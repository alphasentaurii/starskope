
# ********* starskøpe.spacekit.metriks ********* #
""" 
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
class predictions:

    @staticmethod
    def get_preds(x_test,y_test,model=None,verbose=False,**kwargs):
    #y_true = (y_test[:, 0] + 0.5).astype("int") # flatten and make integer
    #y_hat = model.predict(x_test)[:,0] 
    # import pandas as pd
    # import numpy as np
    # from sklearn import metrics
    # from sklearn.metrics import accuracy_score, recall_score

        # class predictions 
        y_true = y_test.flatten()
        y_hat = model.predict(x_test)
        y_pred = model.predict_classes(x_test).flatten() 
        preds=pd.Series(y_pred).value_counts(normalize=False)
        
        
        if verbose:
            print(f"y_pred:\n {preds}")
            print("\n")

        return y_true, y_pred

### PLOTS
class plots:

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
        plt.suptitle('Model Training Results', y=1.01)
        plt.tight_layout()
        plt.show()

        return fig

    # PLOT_CONFUSION_MATRIX()
    @staticmethod
    def plot_confusion_matrix(cmatrix, classes=None, normalize=False, title='Confusion Matrix', cmap='Blues',
        print_raw=False,figsize=(7,8)): 
        from sklearn import metrics                       
        from sklearn.metrics import confusion_matrix
        import itertools
        import numpy as np
        import matplotlib.pyplot as plt
        
        # make confusion matrix if tuple passed to cm:
        if isinstance(cmatrix, tuple):
            y_true = cmatrix[0].copy()
            y_pred = cmatrix[1].copy()
            cm = metrics.confusion_matrix(y_true, y_pred)
        else:
            cm = cmatrix
        
        # INTEGER LABELS
        if classes is None:
            classes=list(range(len(cm)))

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
        #mask = np.zeros_like(cm, dtype=np.bool)
        #idx = np.triu_indices_from(mask)
        #mask[idx] = True
        plt.imshow(cm, cmap=cmap, aspect='equal')
        # Add title and axis labels 
        plt.title('Confusion Matrix') 
        plt.ylabel('True label') 
        plt.xlabel('Pred label')
        
        # Add appropriate axis scales
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        #ax.set_ylim(len(cm), -.5,.5)
        
        # Text formatting
        fmt = '.2f' if normalize else 'd'
        # Add labels to each cell
        thresh = cm.max() / 2.
        # iterate thru matrix and append labels  
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment='center',
                    color='darkgray' if cm[i, j] > thresh else 'black',
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
    def score(x_test, y_test, model=None, get_preds=False, confusion_matrix=True, ROC=True, hist=True, **kwargs):
        """
        evaluates the model predictions and stores the output in a dictionary `score`
        """
        from spacekit.metriks import get_preds, plot_confusion_matrix, keras_history, roc_plots
        from sklearn import metrics
        from sklearn.metrics import jaccard_score
        from sklearn.metrics import confusion_matrix

        score = {}
        if model is None:
            model = model
        if get_preds: 
        # class predictions 
            y_true = y_test.flatten()
            y_pred = model.predict_classes(x_test).flatten() 

        y_true, y_pred = get_preds(x_test,y_test,model,**kwargs)
        summary = model.summary


        model_idx = {'model':model, 'summary':summary,'y_true':y_true,'y_pred':y_pred}
        score['model']=model_idx

        # CLASSIFICATION REPORT
        num_dashes=20
        print('\n')
        print('---'*num_dashes)
        print('\tCLASSIFICATION REPORT:')
        print('---'*num_dashes)
        report = metrics.classification_report(y_true,y_pred)
        
        # scores
        jaccard = jaccard_score(y_test, y_pred)
        acc = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        scores = {}
        scores['jaccard'] = jaccard
        scores['accuracy'] = acc
        scores['recall'] = recall

        print('Jaccard Similarity Score:',jaccard)
        print('\nAccuracy Score:', acc)
        print('\nRecall Score:', recall_score)

        
        score_idx = {'report':report,'scores':scores}
        score['report'] = score_idx

        # CONFUSION MATRIX
        if confusion_matrix:
            CM = confusion_matrix(y_true, y_pred, labels=[0,1])
            
            # Plot confusion matrix
            fig = plot_confusion_matrix(cm, classes=['No Planet', 'Planet'], 
                                        normalize=False,                               
                                        title='Confusion matrix')
            plt.show()

            return CM


        # ROC Area Under Curve
        if roc:
            ROC = roc_plots(y_test, y_pred)

        
        #Plot Model Training Results (PLOT KERAS HISTORY)
        HIST = plot_keras_history(history)
        
        plot_idx = {'CM':CM,'HIST':HIST,'ROC':ROC}

        score['plots'] = plot_idx

        return score

