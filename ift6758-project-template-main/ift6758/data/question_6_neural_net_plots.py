from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve, CalibrationDisplay

def plot_roc(preds_valid, y_valid, save_path = 'Q6_neural_net_ROC.Curve.png'):
    fpr, tpr, _ = roc_curve(y_valid, preds_valid)
    fig = plt.figure(figsize=(12.5,7.5))
    roc_auc = auc(fpr, tpr)
    
    lw = 3
    color_list = ['darkorange','green','navy','red']
        
    plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Neural Net (area = {round(roc_auc,3)})')

    plt.plot([0, 1], [0, 1], color="black", lw=lw, label="Ideal Random Baseline",linestyle="--")
    plt.legend(fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.title("ROC CURVE", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.savefig(save_path, bbox_inches = 'tight')
    return fig


def prep_percentile_df(y_valid, preds_valid, n_buckets = 20):
    
    df = pd.DataFrame(
        {'goal_ind':  y_valid,
         'Goal_Prob': preds_valid
        }
    )

    df['shot_count'] = 1
    df['Goal_Prob_bucket'] = pd.qcut(df['Goal_Prob'], n_buckets, labels = False) + 0
    df = df.groupby(['Goal_Prob_bucket']).sum().reset_index()
    df['goal_rate'] = df['goal_ind']/df['shot_count']
    df['pred_percentile'] = df['Goal_Prob_bucket']*(100/n_buckets)
    df = df.sort_values('pred_percentile', ascending = False)
    df['cumul_goal_pct'] = np.cumsum(df['goal_ind'])/sum(df['goal_ind'])
    
    return df


def plot_goal_rate(df, save_path = 'Q6_neural_net_Goal_Rate.png'):

    fig = plt.figure(figsize=(12.5,7.5))
    plt.plot(
        df['pred_percentile'],
        df['goal_rate'],
        color='darkorange', 
        lw = 3,
        label = 'Neural Net'
    )
    plt.xlim(left=105, right=-5)
    plt.ylim(bottom=0, top=1)

    plt.legend(fontsize=12)
    plt.ylabel('Goal / (Shot + Goal)', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.title(f"Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    fig.savefig(save_path, bbox_inches = 'tight')
    return fig


def plot_cum_goal_rate(df, save_path = 'Q6_neural_net_Cum_Goal.png'):

    fig = plt.figure(figsize=(12.5,7.5))
    plt.plot(
        df['pred_percentile'],
        df['cumul_goal_pct'],
        color='darkorange', 
        lw = 3,
        label = 'Neural Net'
    )
    plt.xlim(left=105, right=-5)
    plt.ylim(bottom=0, top=df['cumul_goal_pct'].max()+0.05)

    plt.legend(fontsize=12)
    plt.ylabel('Cumulative Goal Proportion', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.title(f"Cumulative Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    fig.savefig(save_path, bbox_inches = 'tight')
    return fig


def plot_calibration_curve(y_valid, preds_valid, n_bins=20, save_path = 'Q6_neural_net_Calibration_Curve.png'):
    fig = plt.figure(figsize=(8, 8))
    plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    prob_true, prob_pred = calibration_curve(y_valid, preds_valid, n_bins=n_bins)
    plt.plot(prob_pred, prob_true, "s-",label='Neural Net')

    plt.ylabel("Fraction of positives")
    plt.ylim([-0.05, 1.05])
    plt.legend(loc="lower right")
    plt.title('Calibration plots (reliability curve)',size=24)
    plt.xlabel("Mean predicted value",size=24)
    plt.ylabel("Fraction of Positives",size=24)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    plt.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(save_path, bbox_inches = 'tight')
    return fig