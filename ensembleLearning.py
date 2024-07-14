
import pandas as pd

df = pd.read_csv('diabetes.csv')

# Read input data for training
xInp = df.drop('Outcome', axis='columns')
yInp = df.Outcome

# Scaling input
from sklearn.preprocessing import StandardScaler
scalerObj = StandardScaler()
xInpSc = scalerObj.fit_transform(xInp)

# Splitting data
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xInpSc, yInp, test_size=0.2, stratify=yInp)

# Classfication model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
scores = cross_val_score(DecisionTreeClassifier(), xTrain, yTrain, cv=5)
print(f"DTClassifier score: {scores}")
print(f"Avg DTClassifier score: {scores.mean()}")

# Bagging classifier
from sklearn.ensemble import BaggingClassifier
bagModel = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=100, max_samples=0.8, oob_score=True)
bagModel.fit(xTrain, yTrain)
print(f"Bag model OOB score: {bagModel.oob_score_}")

# Bag Model with cross-validation
scores = cross_val_score(bagModel, xTrain, yTrain, cv=5)
print(f"Bagging + CrossValidation score: {scores}")
print(f"Avg (Bagging + CrossValidation) score: {scores.mean()}")