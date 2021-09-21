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

training_X = all_data[0:30,0:4]
training_Y = all_data[0:30,4: ]
validation_X = all_data[31:,0:4]
validation_Y = all_data[31:,4: ]
#%% add squared  terms and itneraction terms
f_sq = np.zeros_like(training_X)
#add squared terms
for n in range(len(training_X[0])):
    f_sq[:,n] = training_X[:,n] * training_X[:,n] 
#add interaction terms
# for n in range(len(training_X[0])):
#     if n == 0:
#         f_int[:,n] = training_X[:,n] * training_X[:,len(training_X[0]] 
#     else:     
#         f_int[:,n] = training_X[:,n] * training_X[:,n] 

training_X = np.concatenate((training_X, f_sq),axis=1)
#%% add squared  terms and itneraction terms
f_val_sq = np.zeros_like(validation_X)
#add squared terms
for n in range(len(validation_X[0])):
    f_val_sq[:,n] = validation_X[:,n] * validation_X[:,n] 
#add interaction terms
# for n in range(len(training_X[0])):
#     if n == 0:
#         f_int[:,n] = training_X[:,n] * training_X[:,len(training_X[0]] 
#     else:     
#         f_int[:,n] = training_X[:,n] * training_X[:,n] 

validation_X = np.concatenate((validation_X, f_val_sq),axis=1)


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


#%% Validate model on training data
pred_train = model.predict(training_X)
rmse_train_STY = np.sqrt(np.mean((pred_train[:,0] - training_Y[:,0])**2)) 
rmse_train_conv = np.sqrt(np.mean((pred_train[:,1] - training_Y[:,1])**2)) 
R2_train = model.score(training_X,training_Y)
print('r2_train: %.3f' % R2_train)
a = [0, 2]
b = [0, 1]

plt.figure(1)
plt.plot(pred_train[:,0], training_Y[:,0],'r.', markersize=3, label='STY')
plt.legend()
plt.plot(a, a,'k-', markersize=1,)
plt.xlabel('predicted (STY)')
plt.ylabel('actual (STY)')
plt.title("Parity plot Training")

plt.figure(2)
plt.plot(pred_train[:,1], training_Y[:,1],'b.', markersize=3, label='Conversion')
plt.legend()
plt.xlabel('predicted (conversion)')
plt.ylabel('actual (conversion)')
plt.title("Parity plot Training")
plt.plot(b, b,'k-', markersize=1,)


#%% Validate model on validation data

pred_val = model.predict(validation_X)
rmse_val_STY = np.sqrt(np.mean((pred_val[:,0] - validation_Y[:,0])**2)) 
rmse_val_conv = np.sqrt(np.mean((pred_val[:,1] - validation_Y[:,1])**2)) 
R2_val = model.score(validation_X,validation_Y)
print('r2_val: %.3f' % R2_val)
a = [0, 2]
b = [0, 1]

plt.figure(3)
plt.plot(pred_val[:,0], validation_Y[:,0],'r.', markersize=3, label='STY')
plt.legend()
plt.plot(a, a,'k-', markersize=1,)
plt.xlabel('predicted (STY)')
plt.ylabel('actual (STY)')
plt.title("Parity plot Validation")

plt.figure(4)
plt.plot(pred_val[:,1], validation_Y[:,1],'b.', markersize=3, label='Conversion')
plt.legend()
plt.xlabel('predicted (conversion)')
plt.ylabel('actual (conversion)')
plt.title("Parity plot Validation")
plt.plot(b, b,'k-', markersize=1,)

#%%
# #%% plot design space
# equiv_space = np.linspace(0.9, 3, num=50)
# conc_space = np.linspace(0.2, 0.4, num=50)
# time_space = np.linspace(2.5, 6, num=50)
# temp_space = np.linspace(60, 160, num=50)
# design_space = np.zeros(0)
# # for n in range(50):
# #     if n == 0:
# #         design_space_a = np.reshape(np.stack((equiv_space[0], conc_space[0], time_space[n], temp_space[0])), (1,4))
# #     else:
# #         design_space_b = np.reshape(np.stack((equiv_space[0], conc_space[0], time_space[n], temp_space[0])), (1,4)) 
# #         design_space_a = np.vstack((design_space_a, design_space_b))                          

# for n in range(50):
#     if n == 0:
#         design_space_a = np.stack((np.full(50, 0.9), np.full(50,0.2), np.full(50, time_space[n]), temp_space[:]),axis = 1)
#     else:
#         design_space_b = np.stack((np.full(50, 0.9), np.full(50,0.2), np.full(50, time_space[n]), temp_space[:]),axis = 1)
#         design_space_a = np.vstack((design_space_a, design_space_b))     


# pred_design_space = model.predict(design_space_a)

# plt.figure(5)
# X, Y = np.meshgrid(temp_space[:], time_space[:])
# Z = np.reshape(pred_design_space[:,1], (50,50))
# plt.contour(X, Y, Z)

 
#%%
# #%% Decision tree

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
