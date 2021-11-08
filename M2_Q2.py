#!/usr/bin/env python
# coding: utf-8

# 2. Feature Engineering I (10%)

# In[77]:


import pandas as pd
import numpy as np
import math
import requests
import json
import os
from matplotlib import pyplot as plt
import seaborn as sns


# In[78]:


df = pd.read_csv(r'C:\Users\chait\OneDrive\Documents\School\IFT_6758\2.Project\Milestone2_Draft\data\tidy_df.csv')


# In[79]:


# Angle of the shot added to the dataframe
df['angle_from_net'] = np.arcsin(df['y_coordinates'].abs()/df['distance_from_net'])*180/math.pi


# In[80]:


# Imputing empty net data in existing dataframe

df['empty_net'] = df['empty_net'].replace(np.nan,0)
df['empty_net'] = df['empty_net'].replace(True,1)
df['empty_net'] = df['empty_net'].replace(False,0)


# In[81]:


df = df[~df['distance_from_net'].isnull()] # distance was null for 17 rows
df = df[~df['angle_from_net'].isnull()] # angle was null for 18 rows


# In[84]:


# training (train+validation sets for model) and test tests creation

df_train = df[df['season'] != 20192020]
df_test = df[df['season'] == 20192020]


# In[83]:


df_train_nogoals = df_train[df_train['shot_ind']==1]
df_train_goals = df_train[df_train['goal_ind']==1]


# In[87]:


fig = plt.figure(figsize=(14,7))
plt.subplot(121)
plt.hist(df_train_nogoals['distance_from_net'], edgecolor="red", bins=math.ceil((df_train_nogoals['distance_from_net'].max() - df_train_nogoals['distance_from_net'].min())/5))
plt.title("shots (excluding goals) binned by distance")
plt.show()


# In[86]:


fig = plt.figure(figsize=(35,10))
plt.subplot(121)
plt.hist(df_train_goals['distance_from_net'], edgecolor="red", bins=math.ceil((df_train_goals['distance_from_net'].max() - df_train_goals['distance_from_net'].min())/5))
plt.title("goals binned by distance")
plt.show()


# In[88]:


fig = plt.figure(figsize=(35,10))
plt.subplot(121)
plt.hist(df_train_nogoals['angle_from_net'], edgecolor="yellow", bins=math.ceil((df_train_nogoals['angle_from_net'].max() - df_train_nogoals['angle_from_net'].min())/5))
plt.title("shots (excluding goals) binned by angle")
plt.show()


# In[89]:


fig = plt.figure(figsize=(35,10))
plt.subplot(121)
plt.hist(df_train_goals['angle_from_net'],edgecolor="red", bins=math.ceil((df_train_goals['angle_from_net'].max() - df_train_goals['angle_from_net'].min())/5))
plt.title("goals binned by distance")
plt.show()


# In[90]:


# Joint plot showing the relation between distance of the shot and angle of the shot
sns.jointplot(data=df_train, x="distance_from_net", y="angle_from_net", hue="goal_ind")


# In[91]:


# Relation between goal rate and distance


n_buckets = 20

df_train['distance_from_net_bucket'] = pd.qcut(df_train['distance_from_net'], n_buckets, labels = False) +1
df_train['angle_from_net_bucket'] = pd.qcut(df_train['angle_from_net'], n_buckets, labels = False) +1

intervals = list(set(pd.qcut(df_train['distance_from_net'], n_buckets)))
intervals.sort()
intervals = [str(interval) for interval in intervals]


intervals_ang = list(set(pd.qcut(df_train['angle_from_net'], n_buckets)))
intervals_ang.sort()
intervals_ang = [str(interval_ang) for interval_ang in intervals_ang]

df_train_copy = df_train.copy()
df_train_copy['shot_count'] = 1

df_train_copy1 = df_train_copy[['goal_ind','shot_count','distance_from_net_bucket']].groupby(['distance_from_net_bucket']).sum().reset_index()
df_train_copy1['goal_rate'] = df_train_copy1['goal_ind']/df_train_copy1['shot_count']

y1_max = max(df_train_copy1['goal_rate'])

fig = plt.figure(figsize = (14,7))
ax = sns.lineplot(x = 'distance_from_net_bucket', y = 'goal_rate', label='goal percentage', data = df_train_copy1, color='b', legend = False, linewidth = 2.5)
ax.set_xticks(range(1,n_buckets+1))
ax.set_xticklabels(intervals, rotation = 45)
ax.set_ylim(bottom=0, top=y1_max * 1.1)
fig.legend(loc="upper right")
plt.title(f"Relation between goal rate and distance")
fig.legend(loc="upper right")
plt.autoscale()
fig.show()
#fig.savefig("../../ift6758-blog-template-main/figures/question_5_2_"+str(i)+".png", bbox_inches = 'tight')


# In[64]:


df_train_copy1


# In[92]:


# Relation between goal rate and angle

df_train_copy2 = df_train_copy[['goal_ind','shot_count','angle_from_net_bucket']].groupby(['angle_from_net_bucket']).sum().reset_index()
df_train_copy2['goal_rate'] = df_train_copy2['goal_ind']/df_train_copy2['shot_count']

y2_max = max(df_train_copy2['goal_rate'])

fig = plt.figure(figsize = (14,7))
ax = sns.lineplot(x = 'angle_from_net_bucket', y = 'goal_rate', label='goal percentage', data = df_train_copy2, color='b', legend = False, linewidth = 2.5)
ax.set_xticks(range(1,n_buckets+1))
ax.set_xticklabels(intervals_ang, rotation = 45)
ax.set_ylim(bottom=0, top=y2_max * 1.1)
fig.legend(loc="upper right")
plt.title(f"Relation between goal rate and angle")
fig.legend(loc="upper right")
plt.autoscale()
fig.show()


# In[93]:


#goals only to be binned by distance for empty net and non-empty net events

df_train_empty_net = df_train[(df_train['empty_net']==1) & (df_train['goal_ind']==1)]
df_train_non_empty_net = df_train[(df_train['empty_net']==0) & (df_train['goal_ind']==1)]
df_train_empty_net.shape


# In[94]:


fig = plt.figure(figsize=(35,10))
plt.subplot(121)
plt.hist(df_train_empty_net['distance_from_net'], edgecolor="red", bins=math.ceil((df_train_empty_net['distance_from_net'].max() - df_train_empty_net['distance_from_net'].min())/5))
plt.title("Empty net goals binned by distance")
plt.show()


# In[95]:


fig = plt.figure(figsize=(35,10))
plt.subplot(121)
plt.hist(df_train_non_empty_net['distance_from_net'], edgecolor="red", bins=math.ceil((df_train_non_empty_net['distance_from_net'].max() - df_train_non_empty_net['distance_from_net'].min())/5))
plt.title("Non empty net goals binned by distance")
plt.show()


# In[97]:


# events that have incorrect features (e.g. wrong x/y coordinates) validated by the NHL gamecenter video clips


df_goals = df[df['goal_ind']==1]
df_goals_anomaly = df_goals[(df_goals['x_coordinates'].abs()>89)&(df_goals['distance_from_net']>=150)&(df_goals['empty_net']==0)]
df_goals_anomaly


# In[110]:


import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[104]:


# Class 0 (no-goal) vs. Class 1 (goal) of the whole training set
sns.countplot(x='goal_ind', data = df_train, palette = 'Set3')


# In[105]:


# Class 0 (no-goal) vs. Class 1 (goal) of the final test set

sns.countplot(x='goal_ind', data = df_test, palette = 'Set3')


# In[106]:


sns.countplot(x='shot_type', data = df_test, palette = 'Set3', hue = 'goal_ind')


# In[114]:


X = df_train['distance_from_net'].values.reshape(-1,1)
y = df_train['goal_ind']
#y = y.astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.3,random_state=10)
clf = LogisticRegression()
clf.fit(X_train, y_train)
print("accuracy score of the train set data is:",clf.score(X_train, y_train))

print("accuracy score of the validation set data is:",clf.score(X_valid, y_valid))

y_pred = clf.predict(X_valid)

cnt = 0
for i in range(len(X_valid)):
    if y_pred[i] == 1:
        cnt += 1
    
print('Number of goal events predicted by classifier is :', cnt)


# In[115]:


# Confusion Matrix

cm = sklearn.metrics.confusion_matrix(y_valid, y_pred, labels=None, sample_weight=None, normalize=None)
#print(cm)
# true negatives is C00 , false negatives is C10, true positives is C11 and false positives is C01.
print("TN = ", cm[0][0])
print("FN = ", cm[1][0])
print("TP = ", cm[1][1])
print("FP = ", cm[0][1])
print("TPR = ",cm[1][1]/(cm[1][1]+cm[1][0]))
print("FPR = ",cm[0][1]/(cm[0][1]+cm[0][0]))


# In[116]:


# Predictted probabilities of Class 0 (no goal) & Class 1 (goal)

predicted_prob = clf.predict_proba(X_valid)
predicted_prob


# In[117]:


# ROC curve

y_score = clf.fit(X_train, y_train).decision_function(X_valid)

from sklearn.metrics import roc_curve, auc
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_valid, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(
    fpr,
    tpr,
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.3f)" % roc_auc,
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic (ROC) curve")
plt.legend(loc="lower right")
plt.show()


# In[119]:


# a new df with goal_inx and probabilities of classes

dummy_goal = []
for i in y_valid:
    dummy_goal.append(i)
df_goals = pd.DataFrame(dummy_goal, columns = ["goal_ind"])

dummy_list = []
for i in predicted_prob:
    dummy_list.append(i)
df_prob = pd.DataFrame(dummy_list, columns = ["No_Goal_Prob","Goal_Prob"])
df_new = pd.concat([df_goals, df_prob], axis=1)
df_new['shot_count'] = 1
df_new
#df_new['Goal_Prob'].quantile(np.linspace(0,1,21))


# In[121]:


# The goal rate (#goals / (#no_goals + #goals)) as a function of the shot probability model percentile

n_buckets = 20
dummy = 0
j=0
temp_list = []

df_new['Goal_Prob_bucket'] = pd.qcut(df_new['Goal_Prob'], n_buckets, labels = False) + 0
# print(df_new.sample(10))
df_new1 = df_new.copy()
df_new1 = df_new1[['goal_ind','shot_count','Goal_Prob_bucket']].groupby(['Goal_Prob_bucket']).sum().reset_index()
df_new1['goal_rate'] = df_new1['goal_ind']/df_new1['shot_count']
df_new1['pctile'] = df_new1['Goal_Prob_bucket']*(100/n_buckets)
df_new1['pctile_cum'] = 100-(df_new1['Goal_Prob_bucket']+1)*(100/n_buckets)
for i in df_new1['goal_ind']:
    dummy = i + dummy
    temp = dummy / df_new1['goal_ind'].sum()
    temp_list.append(temp)
    j+=1
df_new1['Proportion'] = temp_list
print(df_new1.drop(['Goal_Prob_bucket'],axis=1))


# In[122]:


y1_max = max(df_new1['goal_rate'])

fig = plt.figure(figsize = (14,7))
ax = sns.lineplot(x = 'pctile', y = 'goal_rate', label='goal percentile function of distance from net', data = df_new1, color='b', legend = False, linewidth = 2.5)
ax.set_xlim(left=105, right=-5)
ax.set_ylim(bottom=0, top=1)
fig.legend(loc="upper right")
plt.title(f"goal rate (#goals / (#no_goals + #goals)) as a function of the shot probability model percentile")
fig.legend(loc="upper right")
#plt.autoscale()
fig.show()


# In[123]:


y1_max = max(df_new1['Proportion'])

fig = plt.figure(figsize = (14,7))
ax = sns.lineplot(x = 'pctile_cum', y = 'Proportion', label='goal percentile function of distance from net', data = df_new1, color='b', legend = False, linewidth = 2.5)
ax.set_xlim(left=105, right=-5)
ax.set_ylim(bottom=0, top=1)
fig.legend(loc="upper right")
plt.title(f"goal percentile function of distance from net")
fig.legend(loc="upper right")
#plt.autoscale()
fig.show()


# In[ ]:




