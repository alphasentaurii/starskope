
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
from sklearn.metrics import confusion_matrix #ugh



def get_preds(X,y,model=None,verbose=False):
    if model is None:
        model=model
    # class predictions 
    y_true = y.flatten()
    y_hat = model.predict(X)
    y_pred = model.predict_classes(X).flatten() 
    preds = pd.Series(y_pred).value_counts(normalize=False)
    
    if verbose:
        print(f"y_pred:\n {preds}")
        print("\n")

    return y_true, y_pred



def fnfp(model=None,X,y):
    if model is None:
        model = model
    y_pred = np.round( model.predict(X) )

    pos_idx = y==1
    neg_idx = y==0

    tp = np.sum(y_pred[pos_idx]==1)/y_pred.shape[0]
    fn = np.sum(y_pred[pos_idx]==0)/y_pred.shape[0]

    tn = np.sum(y_pred[neg_idx]==0)/y_pred.shape[0]
    fp = np.sum(y_pred[neg_idx]==1)/y_pred.shape[0]

    return fn,fp

    ### PLOT_KERAS_HISTORY()
    ## Based on code from James Irving PhD / Flatiron School Study Group Notes 

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
    if history is None:
        history=history

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
    
def fusion_matrix(matrix=(y_true,y_pred), classes=None, normalize=False, title='FUSION Matrix', cmap='Blues',
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
    # from sklearn import metrics                       
    # from sklearn.metrics import confusion_matrix #ugh
    # import itertools
    # import numpy as np
    # import matplotlib.pyplot as plt
    
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
    # If so, normalize the raw fusion matrix before visualizing
    if normalize:
        fusion = fusion.astype('float') / fusion.sum(axis=1)[:, np.newaxis]
        fmt='.2f'
    else:
        fmt='d'
    
    # PLOT
    fig, ax = plt.subplots(figsize=(10,10))
    # mask = np.zeros_like(matrix, dtype=np.bool)
    # idx = np.true_indices_from(mask)
    # mask[idx] = True
    plt.imshow(fusion, cmap=cmap, aspect='equal')
    
    # Add title and axis labels 
    plt.title('FUSION Matrix') 
    plt.ylabel('TRUE') 
    plt.xlabel('PRED')
    
    # Add appropriate axis scales
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    ax.set_ylim(len(fusion), -.5,.5)
    
    # Text formatting
    fmt = '.2f' if normalize else 'd'
    # Add labels to each cell
    thresh = fusion.max() / 2.
    # iterate thru matrix and append labels  
    for i, j in itertools.product(range(fusion.shape[0]), range(fusion.shape[1])):
        plt.text(j, i, format(fusion[i, j], fmt),
                horizontalalignment='center',
                color='white' if fusion[i, j] > thresh else 'black',
                size=14, weight='bold')
    
    # Add a legend
    plt.colorbar()
    plt.show() 
    return fusion, fig



def roc_plots(X,y):
    # from sklearn import metrics
    # from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

    y_true = y.flatten()
    y_hat = model.predict(X)

    fpr, tpr, thresholds = roc_curve(y_true, y_hat) 
    fpr, tpr, thresholds = roc_curve(y_true, y_hat)

    # Threshold Cutoff for predictions
    crossover_index = np.min(np.where(1.-fpr <= tpr))
    crossover_cutoff = thresholds[crossover_index]
    crossover_specificity = 1.-fpr[crossover_index]

    plt.plot(thresholds, 1.-fpr)
    plt.plot(thresholds, tpr)
    plt.title("Crossover at {0:.2f} with specificity {1:.2f}".format(crossover_cutoff, crossover_specificity))
    plt.show()

    plt.plot(fpr, tpr)
    plt.title("ROC area under curve is {0:.2f}".format(roc_auc_score(y_true, y_hat)))
    plt.show()
    
    roc = roc_auc_score(y_true,y_hat)
    print("ROC_AUC SCORE:",roc)
    return roc


    ### COMPUTE SCORES ###

def compute(X, y, model=None, hist=None, preds=True, summary=True, fusion=True, 
    roc=True, **kwargs):
    """
    evaluates model predictions and stores the output in a dataframe
    returns `improbability_drive`
    """
    # import pandas as pd
    # from sklearn import metrics
    # from sklearn.metrics import jaccard_score,confusion_matrix

    if model is None:
        model = model

    # initialize a spare improbability drive
    improbability_drive = pd.DataFrame(columns=['models','hist','preds','summary'])

    if hist is None:
        hist=history.history

    # class predictions 
    if preds:
        y_true = y.flatten()
        y_pred = model.predict_classes(X).flatten() 

    if summary:
        summary = model.summary

    # save in df
    improbability_drive['models'] = model
    improbability_drive['summary'] = model.summary
    improbability_drive['preds'] = y_pred

    # FUSION MATRIX
    if fusion:
        # Plot fusion matrix
        fusion = fusion_matrix(matrix=(y_true,y_pred), classes=['No Planet', 'Planet'])
        plt.show()

    # ROC Area Under Curve
    if roc:
        ROC = roc_plots(X, y)

    #Plot Model Training Results (PLOT KERAS HISTORY)
    if hist:
        HIST = keras_history(model, hist)
    
    improbability_drive['fusion'] = fusion
    improbability_drive['roc'] = ROC
    improbability_drive['hist'] = HIST

    # CLASSIFICATION REPORT
    num_dashes=20
    print('\n')
    print('---'*num_dashes)
    print('\tCLASSIFICATION REPORT:')
    print('---'*num_dashes)
    # generate report
    report = metrics.classification_report(y_true,y_pred)
    print(report)

    # calculate scores
    # jacc = jaccard_score(y_true y_pred)
    # acc = accuracy_score(y_true, y_pred)
    # rec = recall_score(y_true, y_pred)

    # save to df:
    improbability_drive['report'] = report
    improbability_drive['jaccard'] = jaccard_score(y_true y_pred)
    improbability_drive['accuracy'] = accuracy_score(y_true, y_pred)
    improbability_drive['recall'] = recall_score(y_true, y_pred)

    return improbability_drive

