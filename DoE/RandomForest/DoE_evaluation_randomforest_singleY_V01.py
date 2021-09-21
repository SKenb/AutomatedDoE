# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 10:58:07 2021

@author: peter.sagmeister
"""

#%% import functions
import numpy as np
from numpy import mean
from numpy import std 
import os
import pandas as pd
from datetime import datetime
import sklearn
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.experimental import enable_hist_gradient_boosting 
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


#%% load the dataset and import data from csv
dirname = os.path.dirname(__file__)
file_name = os.path.join(dirname,'Sample_Data_DoE.csv')  
read_in = pd.read_csv(file_name, sep=";", header=None)   #time_data = pd.read_csv(file_name, sep=";", index_col=0, header=None)

#%% separate data set in training data set and validation set

# training_X = np.array(read_in.iloc[0:19,2:6], dtype=np.float32)
# training_Y = np.array(read_in.iloc[0:19,6:], dtype=np.float32)

# validation_X = np.array(read_in.iloc[20:,2:6], dtype=np.float32)
# validation_Y = np.array(read_in.iloc[20:,6:], dtype=np.float32)

#%% random training and validation set
all_data = np.array(read_in.iloc[0:,2:], dtype=np.float32)
np.random.shuffle(all_data)  #shuffel data

training_X = all_data[0:20,0:4]
training_Y = all_data[0:20,4:5 ]
validation_X = all_data[21:,0:4]
validation_Y = all_data[21:,4:5 ]

#%% add squared  terms and itneraction terms

#add squared terms
for 
f1_sq = 

#%%
# define the model
model = RandomForestRegressor(n_estimators=100, criterion='mse',)
# evaluate the model
cv = RepeatedKFold(n_splits=2, n_repeats=20, random_state=1)
n_scores = cross_val_score(model, training_X, training_Y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1, error_score='raise')
# report performance
print('MAE: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))

#%% fit the model on the whole dataset
model.fit(training_X, training_Y)
#%%
model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0, loss='ls').fit(training_X, training_Y)
R2_gbr_train = model_gbr.score(training_X,training_Y)
R2_gbr_val = model_gbr.score(validation_X,validation_Y)
print('r2_gbr_train: %.3f' % R2_gbr_train)
print('r2_gbr_val: %.3f' % R2_gbr_val)


#%% Validate model on training data
pred_train = model.predict(training_X)
rmse_train_STY = np.sqrt(np.mean((pred_train[:] - training_Y[:])**2)) 
R2_train = model.score(training_X,training_Y)
print('r2_train: %.3f' % R2_train)
a = [0, 2]
b = [0, 1]

plt.figure(1)
plt.plot(pred_train[:], training_Y[:],'r.', markersize=3, label='STY')
plt.legend()
plt.plot(a, a,'k-', markersize=1,)
plt.xlabel('predicted (STY)')
plt.ylabel('actual (STY)')
plt.title("Parity plot Training")



#%% Validate model on validation data

pred_val = model.predict(validation_X)
rmse_val_STY = np.sqrt(np.mean((pred_val[:] - validation_Y[:])**2)) 
R2_val = model.score(validation_X,validation_Y)
print('r2_val: %.3f' % R2_val)
a = [0, 2]
b = [0, 1]

plt.figure(2)
plt.plot(pred_val[:], validation_Y[:],'r.', markersize=3, label='STY')
plt.legend()
plt.plot(a, a,'k-', markersize=1,)
plt.xlabel('predicted (STY)')
plt.ylabel('actual (STY)')
plt.title("Parity plot Validation")



 
#%%
#%% Decision tree

from sklearn import tree
dtmodel = tree.DecisionTreeRegressor()
dtmodel.fit(training_X, training_Y)
pred_dtmodel_val = dtmodel.predict(validation_X)
R2_dt_train = dtmodel.score(training_X,training_Y)
R2_dt_val = dtmodel.score(validation_X,validation_Y)
print('r2_dt_train: %.3f' % R2_dt_train)
print('r2_dt_val: %.3f' % R2_dt_val)

#%% multi linear regression
from sklearn import linear_model
# Create linear regression object
regrmodel = linear_model.LinearRegression()
# Train the model using the training sets
regrmodel.fit(training_X,training_Y)
# Make predictions using the testing set
R2_regr_train = regrmodel.score(training_X,training_Y)
R2_regr_val = regrmodel.score(validation_X,validation_Y)
pred_regrmodel_val = regrmodel.predict(validation_X)
print('r2_regr_train: %.3f' % R2_regr_train)
print('r2_regr_val: %.3f' % R2_regr_val)
