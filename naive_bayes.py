# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:25:40 2019

@author: Samarth
"""

import numpy as np
import pandas as pd

# other dependencies that you might not need
# just for publishing image in notebook
# from IPython.display import Image
# from IPython.core.display import HTML


# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))
    
    # return p
    return p


# column has all the name of column name 
# our data is stored in dataframe: data

column = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
data = pd.read_csv('pima-indians-diabetes.data.csv',names=column)

# Number of patients of outcome 1
n_outcome1 = data['Outcome'][data['Outcome'] == 1].count()

# Number of patients of outcome 0
n_outcome0 = data['Outcome'][data['Outcome'] == 0].count()

# Total people
total_ppl = data['Outcome'].count()

P_outcome1 = n_outcome1 / total_ppl
P_outcome0 = n_outcome0 / total_ppl

# Now first calculate the means of the data according to outcome
# Group the data by gender and calculate the means of each feature
data_means = data.groupby('Outcome').mean()
# View the values
data_means


# Second calculate the variance of the data according to outcome
# Group the data by gender and calculate the variance of each feature
data_variance = data.groupby('Outcome').var()
# View the values
data_variance


# Means for outcome1 for all features
outcome1_Pregnancies_mean = data_means['Pregnancies'][data_variance.index == 1].values[0]
outcome1_Glucose_mean = data_means['Glucose'][data_variance.index == 1].values[0]
outcome1_BloodPressure_mean = data_means['BloodPressure'][data_variance.index == 1].values[0]
outcome1_SkinThickness_mean = data_means['SkinThickness'][data_variance.index == 1].values[0]
outcome1_Insulin_mean = data_means['Insulin'][data_variance.index == 1].values[0]
outcome1_BMI_mean = data_means['BMI'][data_variance.index == 1].values[0]
outcome1_DiabetesPedigreeFunction_mean = data_means['DiabetesPedigreeFunction'][data_variance.index == 1].values[0]
outcome1_Age_mean = data_means['Age'][data_variance.index == 1].values[0]

# Means for outcome0 for all features
outcome0_Pregnancies_mean = data_means['Pregnancies'][data_variance.index == 0].values[0]
outcome0_Glucose_mean = data_means['Glucose'][data_variance.index == 0].values[0]
outcome0_BloodPressure_mean = data_means['BloodPressure'][data_variance.index == 0].values[0]
outcome0_SkinThickness_mean = data_means['SkinThickness'][data_variance.index == 0].values[0]
outcome0_Insulin_mean = data_means['Insulin'][data_variance.index == 0].values[0]
outcome0_BMI_mean = data_means['BMI'][data_variance.index == 0].values[0]
outcome0_DiabetesPedigreeFunction_mean = data_means['DiabetesPedigreeFunction'][data_variance.index == 0].values[0]
outcome0_Age_mean = data_means['Age'][data_variance.index == 0].values[0]

# Variance for outcomeo for all features
outcome0_Pregnancies_variance = data_variance['Pregnancies'][data_variance.index == 0].values[0]
outcome0_Glucose_variance = data_variance['Glucose'][data_variance.index == 0].values[0]
outcome0_BloodPressure_variance = data_variance['BloodPressure'][data_variance.index == 0].values[0]
outcome0_SkinThickness_variance = data_variance['SkinThickness'][data_variance.index == 0].values[0]
outcome0_Insulin_variance = data_variance['Insulin'][data_variance.index == 0].values[0]
outcome0_BMI_variance = data_variance['BMI'][data_variance.index == 0].values[0]
outcome0_DiabetesPedigreeFunction_variance = data_variance['DiabetesPedigreeFunction'][data_variance.index == 0].values[0]
outcome0_Age_variance = data_variance['Age'][data_variance.index == 0].values[0]

# Variance for outcome1 for all features
outcome1_Pregnancies_variance = data_variance['Pregnancies'][data_variance.index == 1].values[0]
outcome1_Glucose_variance= data_variance['Glucose'][data_variance.index == 1].values[0]
outcome1_BloodPressure_variance = data_variance['BloodPressure'][data_variance.index == 1].values[0]
outcome1_SkinThickness_variance = data_variance['SkinThickness'][data_variance.index == 1].values[0]
outcome1_Insulin_variance = data_variance['Insulin'][data_variance.index == 1].values[0]
outcome1_BMI_variance = data_variance['BMI'][data_variance.index == 1].values[0]
outcome1_DiabetesPedigreeFunction_variance = data_variance['DiabetesPedigreeFunction'][data_variance.index == 1].values[0]
outcome1_Age_variance = data_variance['Age'][data_variance.index == 1].values[0]




# Create an empty dataframe that we have to predict 
person = pd.DataFrame()

# Create some feature values for this single row
person['Pregnancies'] = [7]
person['Glucose'] = [130]
person['BloodPressure'] = [86]
person['SkinThickness'] = [34]
person['Insulin'] = [0]
person['BMI'] = [33.5]
person['DiabetesPedigreeFunction'] = [0.564]
person['Age'] = [50]
# View the data 
person

# Numerator of the posterior probability if the unclassified observation is a Outcome1
d_out0 = P_outcome0 * \
p_x_given_y(person['Pregnancies'][0], outcome0_Pregnancies_mean, outcome0_Pregnancies_variance) * \
p_x_given_y(person['Glucose'][0], outcome0_Glucose_mean, outcome0_Glucose_variance) * \
p_x_given_y(person['BloodPressure'][0], outcome0_BloodPressure_mean, outcome0_BloodPressure_variance) * \
p_x_given_y(person['SkinThickness'][0], outcome0_SkinThickness_mean, outcome0_SkinThickness_variance) * \
p_x_given_y(person['Insulin'][0], outcome0_Insulin_mean, outcome0_Insulin_variance) * \
p_x_given_y(person['BMI'][0], outcome0_BMI_mean, outcome0_BMI_variance) * \
p_x_given_y(person['DiabetesPedigreeFunction'][0], outcome0_DiabetesPedigreeFunction_mean, outcome0_DiabetesPedigreeFunction_variance) *\
p_x_given_y(person['Age'][0], outcome0_Age_mean, outcome0_Age_variance) 

# So for now we will only calculate the numerator of the data and will predict based on numerator only

# Numerator of the posterior probability if the unclassified observation is a Outcome1
d_out1 = P_outcome1 * \
p_x_given_y(person['Pregnancies'][0], outcome1_Pregnancies_mean, outcome1_Pregnancies_variance) * \
p_x_given_y(person['Glucose'][0], outcome1_Glucose_mean, outcome1_Glucose_variance) * \
p_x_given_y(person['BloodPressure'][0], outcome1_BloodPressure_mean, outcome1_BloodPressure_variance) * \
p_x_given_y(person['SkinThickness'][0], outcome1_SkinThickness_mean, outcome1_SkinThickness_variance) * \
p_x_given_y(person['Insulin'][0], outcome1_Insulin_mean, outcome1_Insulin_variance) * \
p_x_given_y(person['BMI'][0], outcome1_BMI_mean, outcome1_BMI_variance) * \
p_x_given_y(person['DiabetesPedigreeFunction'][0], outcome1_DiabetesPedigreeFunction_mean, outcome1_DiabetesPedigreeFunction_variance) *\
p_x_given_y(person['Age'][0], outcome1_Age_mean, outcome1_Age_variance)

if d_out0 > d_out1:
    print("Person has no diabetes")
else:
    print("Given person has diabetes")

