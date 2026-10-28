import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('Data/insurance_preprocessed.csv')
df = df.drop(columns=['region'])
X = df.drop(columns="charges")
y = df["charges"]

x_train, x_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=X["smoker"]
)

def rmse(actual, predicted):
    return np.sqrt(np.mean(np.square(actual - predicted)))

def tryModel(x_train, y_train, x_test, y_test, model):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return rmse(y_test, predictions)

# 1. Evaluate on full dataset
modelBoth = LinearRegression()
print("Both: ", tryModel(x_train, y_train, x_test, y_test, modelBoth))

# 2. Evaluate on Smoker subset
smoker_train_mask = x_train['smoker'] == 1
smoker_test_mask = x_test['smoker'] == 1

modelSmoker = LinearRegression()
print("Smoker only: ", tryModel(
    x_train[smoker_train_mask], y_train[smoker_train_mask],
    x_test[smoker_test_mask], y_test[smoker_test_mask],
    modelSmoker
))

# 3. Evaluate on Non-Smoker subset
non_smoker_train_mask = x_train['smoker'] == 0
non_smoker_test_mask = x_test['smoker'] == 0

modelNonSmoker = LinearRegression()
print("Non-Smoker only: ", tryModel(
    x_train[non_smoker_train_mask], y_train[non_smoker_train_mask],
    x_test[non_smoker_test_mask], y_test[non_smoker_test_mask],
    modelNonSmoker
))