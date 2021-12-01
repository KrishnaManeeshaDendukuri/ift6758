from comet_ml import Experiment

import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay

question_no = 7
###############################################
#######           ROC                   #######
###############################################

def roc_auc_plot(df_roc,df_auc, question_no = question_no, n_models = 6,list_labels = None,games=None):
    # list_labels =  ["1","2","3","4"]
    fig = plt.figure(figsize=(12.5,7.5))
    lw = 3
    color_list = ['darkorange','green','navy','red', 'magenta']
    
    for i in range(n_models-1):
        plt.plot(df_roc.FPR[i], df_roc.TPR[i],color=color_list[i],lw=lw,label=f'{list_labels[i]} (area = {round(df_auc.AUC[i],3)})')

    plt.plot([0, 1], [0, 1], color="black", lw=lw, label="Ideal Random Baseline",linestyle="--")
    plt.legend(fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.title("ROC CURVE", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{games}_combined_ROC_Curve.png',bbox_inches = 'tight')
    plt.show()
    
    return fig

def get_roc_auc_plot(df_prob,list_labels=None, question_no = question_no,games=None):
        """
        structure dataframes before plotting
        Saves a plot for the ROC AUC curve
        """
        if not list_labels:
            raise Exception("Expecting an argument 'list_labels' Model Name Labels for the plots")
        n_cols = df_prob.shape[1]

        fpr_list = []
        tpr_list = []
        roc_auc_list = []

        for i in range(1,n_cols):
            fpr, tpr, _ = roc_curve(df_prob.iloc[:,0], df_prob.iloc[:,i])

            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
           
            #  Call for function to plot roc
            #----------------------------------------------------------------------------------------------- 
        df_roc=pd.DataFrame(list(zip(fpr_list,tpr_list)),columns = ['FPR','TPR'])
        df_auc=pd.DataFrame(list(zip(roc_auc_list)),columns = ['AUC'])
        fig = roc_auc_plot(df_roc,df_auc, question_no = question_no, list_labels = list_labels,games=games)
        return fig

###############################################
#######           Goal Rate             #######
###############################################

def goal_rate_plot(df_perc_prop,n_bins, question_no = question_no, list_labels = None, n_models = 6,games=None):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(12.5,7.5))
    
    color_list = ['darkorange','green','navy','red', 'magenta']
    
    # ax2 = sns.lineplot(x = df_perc_prop.pctile[:n_bins-1], y = df_perc_prop.goal_rate[:n_bins-1], label=f'{list_lables[0]}', color=color_list[0], legend = False, linewidth = 3)
    for i in range(n_models-1):

        ax2 = sns.lineplot(x = df_perc_prop.pctile[i*n_bins:n_bins*(i+1)-1], y = df_perc_prop.goal_rate[i*n_bins:n_bins*(i+1)-1], label=f'{list_labels[i]}', color=color_list[i], legend = False, linewidth = 3)
    ax2.set_xlim(left=105, right=-5)
    ax2.set_ylim(bottom=0, top=1)
    plt.legend(fontsize=12)
    plt.ylabel('Goal / (Shot + Goal)', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.title(f"Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{games}_combined_Goal_Rate.png',bbox_inches = 'tight')
    return fig

def get_goal_rate_plot(df_prob=None, n_bins = 20, quant = 5,list_labels = None, question_no = question_no,games=None):
        from matplotlib import pyplot as plt

        """
        df_prob: dataframe with model predictions
        Saves a plot for goal rate as a function of the shot probability model percentile
        """
        if not list_labels:
            raise Exception("Expecting an argument 'list_labels' Model Name Labels for the plots")
        df_prob = df_prob
        n_bins = n_bins
        quant= quant
        
        df_perc,goal_count,shot_count,goal_rate,cum_goal_rate,pctile,pctile_prop = [],[],[],[],[],[],[]
        
        df_prob_1 = df_prob.copy()
        cols = df_prob.columns[1:-1] #'Goal_Prob'
        
        for col in cols:
            df_prob_1['percentile'] = df_prob_1[col].rank(pct=True)
            quantile_list = np.linspace(0,1,n_bins*quant+1).round(4).tolist()
            q = df_prob_1.quantile(quantile_list)
            for i in np.arange(quant,(quant*n_bins)+1,quant):
                df_perc = df_prob_1[((df_prob_1[col]>=q[col][(i-quant)/100]) & (df_prob_1[col]<q[col][i/100]))]
                goal_count.append(df_perc['goal_ind'].sum())
                shot_count.append(df_perc['shot_count'].sum())
                goal_rate.append(df_perc['goal_ind'].sum()/df_perc['shot_count'].sum())
                pctile.append(i)
            
        #  Call for function to plot goal rate
        #----------------------------------------------------------------------------------------------- 
        df_perc_prop = pd.DataFrame(list(zip(goal_count,shot_count,goal_rate,pctile)),columns=['goal_count',"sum_shot_count",'goal_rate','pctile'])
        fig = goal_rate_plot(df_perc_prop = df_perc_prop,n_bins = n_bins, question_no = question_no, list_labels = list_labels, n_models = 6,games=games)    
        return fig

###############################################
#######        Cumulative Goal Rate     #######
###############################################

def cum_rate_plot(df_perc_prop_cum,n_bins, question_no = question_no, n_models= 6,list_labels = None,games=None):
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(12.5,7.5))
    color_list = ['darkorange','green','navy','red', 'magenta']

    for i in range(n_models-1):
        
        ax3 = sns.lineplot(x = df_perc_prop_cum.pctile[i*n_bins:n_bins*(i+1)-1], y = df_perc_prop_cum.cum_goal_rate[i*n_bins:n_bins*(i+1)-1], label=f'{list_labels[i]}', color=color_list[i], legend = False, linewidth = 3)
 
    ax3.set_xlim(left=105, right=-5)
    ax3.set_ylim(bottom=0, top=df_perc_prop_cum['cum_goal_rate'].max()+0.05)
    plt.ylabel('Cumulative Goal Proportion', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.title(f"Cumulative Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.legend(fontsize=12)
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{games}_combined_Cum_Goal.png',bbox_inches = 'tight')
    
    return fig

def get_cum_rate_plot( df_prob=None, n_bins = 20, quant = 5,list_labels = None, question_no = question_no,games=None):
        from matplotlib import pyplot as plt

        """
        df_prob: dataframe with model predictions
        Saves a plot for cumulative proportion of goals as a function of the shot probability model percentile.
        """
        df_prob = df_prob
        n_bins = n_bins
        quant= quant
        
        goal_count2,pctile2,cum_goal_rate2 = [],[],[]
        
        df_prob = df_prob.copy()
        # df_prob['percentile'] = df_prob['Goal_Prob'].rank(pct=True)
        # q = df_prob.quantile(quantile_list)
        # total = df_prob['goal_ind'].sum()

        
        cols = df_prob.columns[1:-1]        
        for col in cols:
            temp,j=0,100
            df_prob['percentile'] = df_prob[col].rank(pct=True)
            quantile_list = np.linspace(0,1,n_bins*quant+1).round(4).tolist()
            q = df_prob.quantile(quantile_list)
            total = df_prob['goal_ind'].sum()
            # to fetch the graph the graph in reverse order (100 ->0)
            for j in np.arange((quant*n_bins),0,-quant):
                df_perc2 = df_prob[((df_prob[col]>q[col][(j-quant)/100]) & (df_prob[col]<=q[col][j/100]))]
                goal_count2.append(df_perc2.goal_ind.sum())
                temp+=df_perc2.goal_ind.sum()
                cum_goal_rate2.append(temp/total)
                pctile2.append(j)
        
        #  Call for function to plot cumulative proportion 
        #----------------------------------------------------------------------------------------------- 
        df_perc_prop_cum = pd.DataFrame(list(zip(goal_count2,cum_goal_rate2,pctile2)),columns=['goal_count','cum_goal_rate','pctile'])
        fig = cum_rate_plot(df_perc_prop_cum, n_bins, question_no = question_no,n_models= 6,list_labels = list_labels,games=games)
        return fig

###############################################
#######           Calibration Curve     #######
###############################################
def calibration_plot(df_calib, n_bins, question_no = question_no, list_labels = None, n_models =6,games=None):
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    cols = df_calib.columns[1:-1] 
    for i in range(n_models-1):
            prob_true, prob_pred = calibration_curve(df_calib.goal_ind[i:], df_calib[cols[i]][i:], n_bins=n_bins)
            ax1.plot(prob_pred, prob_true, "s-",label=list_labels[i])

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)',size=24)
    ax1.set_xlabel("Mean predicted value",size=24)
    ax1.set_ylabel("Fraction of Positives",size=24)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    ax1.legend(loc="upper left", ncol=2)
    fig.savefig(f"../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{games}_Calibration_Curve.png",bbox_inches = 'tight')

    return fig

def get_calibration_plot(df_prob=None,n_bins = 20, question_no = question_no, list_labels = None,games=None):
        """
        df_prob: dataframe with model predictions
        Saves a plot for reliability diagram (calibration curve)
        """
        df_prob = df_prob
        n_bins = n_bins
        
        #  Call for function to plot calibration curve
        #----------------------------------------------------------------------------------------------- 
        df_calib = df_prob.copy()
        fig = calibration_plot(df_calib,n_bins, question_no = question_no, list_labels = list_labels, n_models =6,games=games)
        return fig