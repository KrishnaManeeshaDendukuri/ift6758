from comet_ml import Experiment

import pandas as pd
import numpy as np
import os
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay

################################# get_roc_auc_plot outside class is currently in use(down below) ##########

#### Usage indicated at the end of the script

###############################################
#######           PLOT FUNCTIONS        #######
###############################################

def roc_auc_plot(name,df_roc,df_auc, question_no = None):
    list_labels = [name,'Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    lw = 3
    color_list = ['darkorange','green','navy','red']
        
    plt.plot(df_roc.FPR[0], df_roc.TPR[0],color=color_list[0],lw=lw,label=f'{list_labels[0]} (area = {round(df_auc.AUC[0],3)})')

    plt.plot([0, 1], [0, 1], color="black", lw=lw, label="Ideal Random Baseline",linestyle="--")
    plt.legend(fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.title("ROC CURVE", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
# <<<<<<< Updated upstream
    plt.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_ROC_Curve.png',bbox_inches = 'tight')
    plt.show()
# =======
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}'+name+'_ROC_Curve.png',bbox_inches = 'tight')
# >>>>>>> Stashed changes
    
    return plt
    
def goal_rate_plot(name,df_perc_prop,n_bins, question_no = None):
    list_lables = [name,'Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    
    color_list = ['darkorange','green','navy','red']
    
    ax2 = sns.lineplot(x = df_perc_prop.pctile[:n_bins-1], y = df_perc_prop.goal_rate[:n_bins-1], label=f'{list_lables[0]}', color=color_list[0], legend = False, linewidth = 3)
    ax2.set_xlim(left=105, right=-5)
    ax2.set_ylim(bottom=0, top=1)
    plt.legend(fontsize=12)
    plt.ylabel('Goal / (Shot + Goal)', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.title(f"Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_Goal_Rate.png',bbox_inches = 'tight')
    
    return fig

def cum_rate_plot(name,df_perc_prop_cum,n_bins, question_no = None):
    list_lables = [name,'Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    color_list = ['darkorange','green','navy','red']

    ax3 = sns.lineplot(x = df_perc_prop_cum.pctile[:n_bins-1], y = df_perc_prop_cum.cum_goal_rate[:n_bins-1], label=f'{list_lables[0]}', color=color_list[0], legend = False, linewidth = 3)
 
    ax3.set_xlim(left=105, right=-5)
    ax3.set_ylim(bottom=0, top=df_perc_prop_cum['cum_goal_rate'].max()+0.05)
    plt.ylabel('Cumulative Goal Proportion', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.title(f"Cumulative Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.legend(fontsize=12)
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_Cum_Goal.png',bbox_inches = 'tight')
    
    return fig

def calibration_plot(name,df_calib,n_bins, question_no = None):
    list_lables = [name,'Random Baseline']
    fig = plt.figure(figsize=(8, 8))
    ax4 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    ax4.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    prob_true, prob_pred = calibration_curve(df_calib.goal_ind, df_calib.Goal_Prob, n_bins=n_bins)
    ax4.plot(prob_pred, prob_true, "s-",label=list_lables[0])

    ax4.set_ylabel("Fraction of positives")
    ax4.set_ylim([-0.05, 1.05])
    ax4.legend(loc="lower right")
    ax4.set_title('Calibration plots (reliability curve)',size=24)
    ax4.set_xlabel("Mean predicted value",size=24)
    ax4.set_ylabel("Fraction of Positives",size=24)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    ax4.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_Calibration_Curve.png',bbox_inches = 'tight')
    plt.show()
    return fig


#########################################################
#######      INFER MODEL FOR EVALUATION    ##############
#########################################################

class Performance_Eval:
    def __init__(self, df_prob, n_bins = 20, quant = 5, question_no = None):
        self.n_bins = n_bins
        self.quant = quant 
        self.df_prob = df_prob
        # # self.name = name_of_model
        # self.question_no = question_no
        # # self.y_pred_proba = y_pred_proba
        print("new updated version")
        
    # def infer_model(self):
    #     """
    #     Infer Model -> Fetch model predictions and prediction probabilities
    #     """
        
    #     model = self.model
    #     X_train = self.X_train
    #     y_train = self.y_train
    #     X_valid = self.X_valid
    #     y_valid = self.y_valid
        
    #     self.temp_y_valid = []
    #     self.y_pred = model.predict(X_valid)
    #     self.predicted_prob = model.predict_proba(X_valid)
    #     self.temp_y_valid += y_valid.tolist()
    #     self.predicted_prob_goal = []
    #     for prob in self.predicted_prob:
    #         self.predicted_prob_goal.append(prob[1])

        
    def get_roc_auc_plot(self, df_prob, name = None):

        """
        structure dataframes before plotting
        Saves a plot for the ROC AUC curve
        """
        # self.infer_model()
        n_models = df_prob.shape[0]
        for i in n_models:
            
            # feature_list = self.X_train.columns
            fpr_list = []
            tpr_list = []
            roc_auc_list = []
            fpr, tpr, _ = roc_curve(self.df_prob[i]['goal_ind'], self.df_prob[i]['Goal_Prob'])
            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
            
            #  Call for function to plot roc
            #----------------------------------------------------------------------------------------------- 
        self.df_roc=pd.DataFrame(list(zip(fpr_list,tpr_list)),columns = ['FPR','TPR'])
        self.df_auc=pd.DataFrame(list(zip(roc_auc_list)),columns = ['AUC'])
        plt = roc_auc_plot(name,self.df_roc,self.df_auc, question_no = self.question_no)
        return fig
        
    def get_goal_rate_plot(self, df_prob=None, name = None):
        """
        df_prob: dataframe with model predictions
        Saves a plot for goal rate as a function of the shot probability model percentile
        """
        df_prob = self.df_prob
        n_bins = self.n_bins
        quant= self.quant
        
        df_perc,goal_count,shot_count,goal_rate,cum_goal_rate,pctile,pctile_prop = [],[],[],[],[],[],[]
        
        df_prob_1 = df_prob.copy()
        df_prob_1['percentile'] = df_prob_1['Goal_Prob'].rank(pct=True)
        self.quantile_list = np.linspace(0,1,n_bins*quant+1).round(4).tolist()
        q = df_prob_1.quantile(self.quantile_list)
        col = 'Goal_Prob'
        
        for i in np.arange(quant,(quant*n_bins)+1,quant):
            df_perc = df_prob_1[((df_prob_1[col]>=q[col][(i-quant)/100]) & (df_prob_1[col]<q[col][i/100]))]
            goal_count.append(df_perc['goal_ind'].sum())
            shot_count.append(df_perc['shot_count'].sum())
            goal_rate.append(df_perc['goal_ind'].sum()/df_perc['shot_count'].sum())
            pctile.append(i)
            
        #  Call for function to plot goal rate
        #----------------------------------------------------------------------------------------------- 
        self.df_perc_prop = pd.DataFrame(list(zip(goal_count,shot_count,goal_rate,pctile)),columns=['goal_count',"sum_shot_count",'goal_rate','pctile'])
        plt = goal_rate_plot(name,df_perc_prop = self.df_perc_prop,n_bins = self.n_bins, question_no = self.question_no)    
        return plt
    
    def get_cum_rate_plot(self, df_prob=None, name = None):
        """
        df_prob: dataframe with model predictions
        Saves a plot for cumulative proportion of goals as a function of the shot probability model percentile.
        """
        df_prob = self.df_prob
        n_bins = self.n_bins
        quant= self.quant
        
        goal_count2,pctile2,cum_goal_rate2 = [],[],[]
        
        df_prob = df_prob.copy()
        df_prob['percentile'] = df_prob['Goal_Prob'].rank(pct=True)
        q = df_prob.quantile(self.quantile_list)
        total = df_prob['goal_ind'].sum()

        temp,j=0,100
        col = 'Goal_Prob'
        
        # to fetch the graph the graph in reverse order (100 ->0)
        for j in np.arange((quant*n_bins),0,-quant):
            df_perc2 = df_prob[((df_prob[col]>q[col][(j-quant)/100]) & (df_prob[col]<=q[col][j/100]))]
            goal_count2.append(df_perc2.goal_ind.sum())
            temp+=df_perc2.goal_ind.sum()
            cum_goal_rate2.append(temp/total)
            pctile2.append(j)
        
        #  Call for function to plot cumulative proportion 
        #----------------------------------------------------------------------------------------------- 
        self.df_perc_prop_cum = pd.DataFrame(list(zip(goal_count2,cum_goal_rate2,pctile2)),columns=['goal_count','cum_goal_rate','pctile'])
        plt = cum_rate_plot(name,self.df_perc_prop_cum, self.n_bins, question_no = self.question_no)
        return plt
    
    def get_calibration_plot(self, df_prob=None, name = None):
        """
        df_prob: dataframe with model predictions
        Saves a plot for reliability diagram (calibration curve)
        """
        df_prob = self.df_prob
        n_bins = self.n_bins
        
        #  Call for function to plot calibration curve
        #----------------------------------------------------------------------------------------------- 
        self.df_calib = df_prob.copy()
        plt = calibration_plot(name,self.df_calib,n_bins, question_no = self.question_no)
        return plt


    def plot_multiple_models_roc(model_preds_json, question_no = 7):
        for model in model_preds_json:
            name = model["label"]
            y_test = model["y_test"]
            y_pred_proba = model["y_pred_proba"][:,1]
            perf_eval = Performance_Eval(y_test = y_test, y_pred_proba = y_pred_proba, question_no = question_no)
            plt = perf_eval.get_roc_auc_plot(name)
            

        # Custom settings for the plot 
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()   
            
        # return fig


######################################################

def roc_auc_plot(name, df_roc_list, df_auc_list, question_no = None):
    list_labels = [name,'Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    lw = 3
    color_list = ['darkorange','green','navy','red']
        
    for i in range(len(df_roc_list)):
        plt.plot(df_roc[i].FPR[0], df_roc[i].TPR[0],color=color_list[0],lw=lw,label=f'{list_labels[0]} (area = {round(df_auc[i].AUC[0],3)})')

    plt.plot([0, 1], [0, 1], color="black", lw=lw, label="Ideal Random Baseline",linestyle="--")
    plt.legend(fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.title("ROC CURVE", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
# <<<<<<< Updated upstream
    plt.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_ROC_Curve.png',bbox_inches = 'tight')
    plt.show()
# =======
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}'+name+'_ROC_Curve.png',bbox_inches = 'tight')
# >>>>>>> Stashed changes
    
    return plt
    
def goal_rate_plot(name,df_perc_prop,n_bins, question_no = None):
    list_lables = [name,'Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    
    color_list = ['darkorange','green','navy','red']
    
    ax2 = sns.lineplot(x = df_perc_prop.pctile[:n_bins-1], y = df_perc_prop.goal_rate[:n_bins-1], label=f'{list_lables[0]}', color=color_list[0], legend = False, linewidth = 3)
    ax2.set_xlim(left=105, right=-5)
    ax2.set_ylim(bottom=0, top=1)
    plt.legend(fontsize=12)
    plt.ylabel('Goal / (Shot + Goal)', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.title(f"Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_Goal_Rate.png',bbox_inches = 'tight')
    
    return fig

def cum_rate_plot(name,df_perc_prop_cum,n_bins, question_no = None):
    list_lables = [name,'Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    color_list = ['darkorange','green','navy','red']

    ax3 = sns.lineplot(x = df_perc_prop_cum.pctile[:n_bins-1], y = df_perc_prop_cum.cum_goal_rate[:n_bins-1], label=f'{list_lables[0]}', color=color_list[0], legend = False, linewidth = 3)
 
    ax3.set_xlim(left=105, right=-5)
    ax3.set_ylim(bottom=0, top=df_perc_prop_cum['cum_goal_rate'].max()+0.05)
    plt.ylabel('Cumulative Goal Proportion', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.title(f"Cumulative Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.legend(fontsize=12)
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_Cum_Goal.png',bbox_inches = 'tight')
    
    return fig

def calibration_plot(name,df_calib,n_bins, question_no = None):
    list_lables = [name,'Random Baseline']
    fig = plt.figure(figsize=(8, 8))
    ax4 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    ax4.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    
    prob_true, prob_pred = calibration_curve(df_calib.goal_ind, df_calib.Goal_Prob, n_bins=n_bins)
    ax4.plot(prob_pred, prob_true, "s-",label=list_lables[0])

    ax4.set_ylabel("Fraction of positives")
    ax4.set_ylim([-0.05, 1.05])
    ax4.legend(loc="lower right")
    ax4.set_title('Calibration plots (reliability curve)',size=24)
    ax4.set_xlabel("Mean predicted value",size=24)
    ax4.set_ylabel("Fraction of Positives",size=24)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    ax4.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    fig.savefig(f'../../ift6758-blog-template-main/figures/milestone2/Q{question_no}_{name}_Calibration_Curve.png',bbox_inches = 'tight')
    plt.show()
    return fig


#########################################################
#######      INFER MODEL FOR EVALUATION    ##############
#########################################################

# class Performance_Eval:
#     def __init__(self, y_test, y_pred_proba, n_bins = 20, quant = 5, question_no = None):
#         self.n_bins = n_bins
#         self.quant = quant 
#         self.y_test = y_test
#         # self.name = name_of_model
#         self.question_no = question_no
#         self.y_pred_proba = y_pred_proba
#         print("new updated version")



######## CURRENTLY IN USE

def get_roc_auc_plot(df_prob, name = None):

        """
        structure dataframes before plotting
        Saves a plot for the ROC AUC curve
        """
        # self.infer_model()
        n_models = df_prob.shape[0]
        for i in n_models:
            
            # feature_list = self.X_train.columns
            fpr_list = []
            tpr_list = []
            roc_auc_list = []
            fpr, tpr, _ = roc_curve(df_prob[i]['goal_ind'], df_prob[i]['Goal_Prob'])
            roc_auc = auc(fpr, tpr)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            roc_auc_list.append(roc_auc)
            
            #  Call for function to plot roc
            #----------------------------------------------------------------------------------------------- 
        self.df_roc=pd.DataFrame(list(zip(fpr_list,tpr_list)),columns = ['FPR','TPR'])
        self.df_auc=pd.DataFrame(list(zip(roc_auc_list)),columns = ['AUC'])
        plt = roc_auc_plot(name,self.df_roc,self.df_auc, question_no = self.question_no)
        return fig
