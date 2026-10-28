import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual - predicted)))
def tryModel(inputs,target,model):
    model = model.fit(inputs,target)
    predictions = model.predict(inputs)
    return rmse(predictions, target)
df = pd.read_csv('Data/insurance_preprocessed.csv')
smoker = df[df['smoker'] == 1]
nonSmoker = df[df['smoker'] == 0]
modelBoth = LinearRegression()
modelSmoker = LinearRegression()
modelNonSmoker = LinearRegression()
inputsBoth =  df[['age','smoker','bmi','children','sex','northeast', 'northwest', 'southeast', 'southwest']]
targetsBoth = df['charges']
inputsSmoker =  smoker[['age','bmi','children','sex','northeast', 'northwest', 'southeast', 'southwest']]
targetsSmoker = smoker['charges']
inputsNonSmoker =  nonSmoker[['age','bmi','children','sex','northeast', 'northwest', 'southeast', 'southwest']]
targetsNonSmoker = nonSmoker['charges']
print("Both: ",tryModel(inputsBoth,targetsBoth,modelBoth))
print("Smoker only: ",tryModel(inputsSmoker,targetsSmoker,modelSmoker))
print("non Smoker only: ",tryModel(inputsNonSmoker,targetsNonSmoker,modelNonSmoker))

