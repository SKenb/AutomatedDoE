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
from sklearn.metrics import r2_score

#%% load the dataset and import data from csv
dirname = os.path.dirname(__file__)
file_name = os.path.join(dirname,'Sample_Data_DoE.csv')  
read_in = pd.read_csv(file_name, sep=";", header=None)   #time_data = pd.read_csv(file_name, sep=";", index_col=0, header=None)

#%% separate data set in training data set and validation set

training_X = np.array(read_in.iloc[0:19,2:6], dtype=np.float32)
training_Y = np.array(read_in.iloc[0:19,6:], dtype=np.float32)

validation_X = np.array(read_in.iloc[20:,2:6], dtype=np.float32)
validation_Y = np.array(read_in.iloc[20:,6:], dtype=np.float32)

#%% random training and validation set
# all_data = np.array(read_in.iloc[0:,2:], dtype=np.float32)
# np.random.shuffle(all_data)  #shuffel data

# training_X = all_data[0:30,0:4]
# training_Y = all_data[0:30,4:]
# validation_X = all_data[31:,0:4]
# validation_Y = all_data[31:,4:]


#%%
# define the model
model = RandomForestRegressor()
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
plt.xlabel('predicted (Conversion)')
plt.ylabel('actual (Conversion)')
plt.title("Parity plot Training")
plt.plot(b, b,'k-', markersize=1,)


#%% Validate model on validation data

pred_val = model.predict(validation_X)
rmse_val_STY = np.sqrt(np.mean((pred_val[:,0] - validation_Y[:,0])**2)) 
rmse_val_conv = np.sqrt(np.mean((pred_val[:,1] - validation_Y[:,1])**2)) 
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
plt.xlabel('predicted (Conversion)')
plt.ylabel('actual (Conversion)')
plt.title("Parity plot Validation")
plt.plot(b, b,'k-', markersize=1,)

STY_r2 = r2_score(validation_Y[:,0], pred_val[:,0], sample_weight=None,multioutput="uniform_average")
print('STY r^2:', (STY_r2))
Conversion_r2 = r2_score(validation_Y[:,1], pred_val[:,1], sample_weight=None,multioutput="uniform_average")
print('Conversion r^2:', (Conversion_r2))