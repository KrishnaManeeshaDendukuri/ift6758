from comet_ml import Experiment

import pandas as pd
import numpy as np
import requests
import json
import os
import math
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


###############################################
#######           PLOT FUNCTIONS        #######
###############################################

def roc_auc_plot(list_model_iter, df_roc, df_auc, question_no = None):
    list_lables = ['Distance from Net','Absolute Shot Angle','Distance and Abs Shot Angle','Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    lw = 3
    color_list = ['darkorange','green','navy','red']
    for i in range(len(list_model_iter)):
        
        plt.plot(df_roc.FPR[i], df_roc.TPR[i],color=color_list[i],lw=lw,label=f'{list_lables[i]} (area = {round(df_auc.AUC[i],3)})')

    plt.plot([0, 1], [0, 1], color="black", lw=lw, label="Ideal Random Baseline",linestyle="--")
    plt.legend(fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.title("ROC CURVE", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    fig.savefig("../../ift6758-blog-template-main/figures/milestone2/Q{}LogisticReg_ROC_Curve.png".format(question_no),bbox_inches = 'tight')
 
    return fig
 
def goal_rate_plot(list_model_iter, df_perc_prop, n_bins, question_no = None):
    list_lables = ['Distance from Net','Absolute Shot Angle','Distance and Abs Shot Angle','Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    color_list = ['darkorange','green','navy','red']
    
    for i in range(len(list_model_iter)):

        ax = sns.lineplot(x = df_perc_prop.pctile[i*n_bins:n_bins*(i+1)-1], y = df_perc_prop.goal_rate[i*n_bins:n_bins*(i+1)-1], label=f'{list_lables[i]}', color=color_list[i], legend = False, linewidth = 3)

    ax.set_xlim(left=105, right=-5)
    ax.set_ylim(bottom=0, top=1)
    plt.legend(fontsize=12)
    plt.ylabel('Goal / (Shot + Goal)', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.title(f"Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    fig.savefig("../../ift6758-blog-template-main/figures/milestone2/Q{}LogisticReg_Goal_Rate.png".format(question_no),bbox_inches = 'tight')

    return fig

def cum_rate_plot(list_model_iter, df_perc_prop_cum, n_bins, question_no = None):
    list_lables = ['Distance from Net','Absolute Shot Angle','Distance and Abs Shot Angle','Random Baseline']
    fig = plt.figure(figsize=(12.5,7.5))
    color_list = ['darkorange','green','navy','red']
    
    for i in range(len(list_model_iter)):

        ax = sns.lineplot(x = df_perc_prop_cum.pctile[i*n_bins:n_bins*(i+1)-1], y = df_perc_prop_cum.cum_goal_rate[i*n_bins:n_bins*(i+1)-1], label=f'{list_lables[i]}', color=color_list[i], legend = False, linewidth = 3)
 
    ax.set_xlim(left=105, right=-5)
    ax.set_ylim(bottom=0, top=df_perc_prop_cum['cum_goal_rate'].max()+0.05)
    plt.ylabel('Cumulative Goal Proportion', fontsize=18)
    plt.xlabel('Shot Probability Model Percentile', fontsize=18)
    plt.title(f"Cumulative Goal Rate v.s. Shot Probability Model Percentile", fontsize=18)
    plt.yticks(size = 12)
    plt.xticks(size = 12)
    plt.legend(fontsize=12)
    fig.savefig("../../ift6758-blog-template-main/figures/milestone2/Q{}LogisticReg_Cum_Goal.png".format(question_no),bbox_inches = 'tight')

    return fig

def calibration_plot(list_model_iter, df_calib, n_bins, length, question_no = None):
    list_lables = ['Distance from Net','Absolute Shot Angle','Distance and Abs Shot Angle','Random Baseline']
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1)
    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for i in range(len(list_model_iter)):
        prob_true, prob_pred = calibration_curve(df_calib.goal_ind[int(i*length):int(length*(i+1)-1)], df_calib.Goal_Prob[int(i*length):int(length*(i+1)-1)], n_bins=n_bins)
        ax1.plot(prob_pred, prob_true, "s-",label=list_lables[i])

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots (reliability curve)',size=24)
    ax1.set_xlabel("Mean predicted value",size=24)
    ax1.set_ylabel("Fraction of Positives",size=24)
    plt.yticks(size = 20)
    plt.xticks(size = 20)
    ax1.legend(loc="upper left", ncol=2)
    fig.savefig("../../ift6758-blog-template-main/figures/milestone2/Q{}LogisticReg_Calibration_Curve.png".format(question_no),bbox_inches = 'tight')

    return fig