
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dfFull = pd.read_csv("titanic.csv")
dfClean = dfFull.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'], axis="columns", inplace=False)
print(dfClean)

# Encoding labels
dummyCol = pd.get_dummies(dfClean.Sex)
dfClean = pd.concat([dfClean,dummyCol], axis='columns')
dfClean.drop('Sex', axis='columns', inplace=True)

# Filter NaNs
if not dfClean.columns[dfClean.isna().any()].isnull():
    dfClean.Age = dfClean.Age.fillna(dfClean.Age.mean())

# Split dataset
xInp = dfClean.drop('Survived', axis='columns')
yInp = dfClean['Survived']
xTrain, xTest, yTrain, yTest = train_test_split(xInp, yInp, test_size=0.2)

# Naive bayes classifier
nbObj = GaussianNB()
nbObj.fit(xTrain, yTrain)
print(nbObj.score(xTest, yTest))