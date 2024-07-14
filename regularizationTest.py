
import pandas as pd
from matplotlib import pyplot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge

df = pd.read_csv('Melbourne_housing_FULL.csv')
dropColList = ['Address','Date','Postcode','YearBuilt','Lattitude','Longtitude']
dfNew = df.drop(dropColList, axis='columns', inplace=False)

# Fill Gaps
# print(f"Cols with NaNs:\n {dfNew.isna().sum()}")
colWithNaNs = ['Propertycount','Distance','Bedroom2', 'Bathroom', 'Car']
dfNew[colWithNaNs] = df[colWithNaNs].fillna(0)

dfNew['Landsize'] = dfNew['Landsize'].fillna(dfNew['Landsize'].mean())
dfNew['BuildingArea'] = dfNew['BuildingArea'].fillna(dfNew['BuildingArea'].mean())
dfNew.dropna(inplace=True)

# Encoding
dfEncoded = pd.get_dummies(dfNew, drop_first=True)

# Train test split
xInp = dfEncoded.drop(['Price'], axis='columns')
yInp = dfEncoded['Price']
xTrain, xTest, yTrain, yTest = train_test_split(xInp, yInp, train_size=0.2)
print(xInp.head())

# Linear Regression
modelObj = LinearRegression()
modelObj.fit(xTrain, yTrain)
print(f"Linear Reg Score: {modelObj.score(xTest, yTest)}")

# L-1 Lasso Regularization model
lassoObj = Lasso(alpha=20, max_iter=1000, tol=1e-2)
lassoObj.fit(xTrain, yTrain)
print(f"Lasso-L1 Reg Score: {lassoObj.score(xTest, yTest)}")

# L-2 Ridge Regularization model
ridgeObj = Ridge(alpha=20, max_iter=1000, tol=1e-2)
ridgeObj.fit(xTrain, yTrain)
print(f"Ridge-L2 Reg Score: {ridgeObj.score(xTest, yTest)}")
