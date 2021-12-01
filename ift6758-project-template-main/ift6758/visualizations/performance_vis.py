
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score


def plot_model_performance(model_list):
    """
    model_list: list of dictionaries
    X_test, y_test: Test dataframe, Label Series
    returns a combined roc curve for all the models in the list
    """
    fig = plt.figure(figsize=(8, 8))
    for m in model_list:
            y_pred = m["y_pred"]
            y_test = m["y_test"]
            y_pred_proba = m["y_pred_proba"]
            # Compute False postive rate, and True positive rate
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:,1])
            # Calculate Area under the curve to display on the plot
            auc = roc_auc_score(y_test,y_pred)
            # Now, plot the computed values
            plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], auc))
            
    # Custom settings for the plot 
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity(False Positive Rate)')
    plt.ylabel('Sensitivity(True Positive Rate)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()   
    
    return fig

"""
sample model_list:
[
{
    'label': 'Gradient Boosting also',
    'y_test':  y_test,
    'y_pred': y_pred,
    #without reshaping to [:,1]
    'y_pred_proba' : y_pred_proba
}
]
"""
        
