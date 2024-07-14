
import pandas as pd
from sklearn import model_selection, linear_model
from matplotlib import pyplot

dfPrices = pd.read_csv("insurance_data.csv")
print(dfPrices)

# Plot the data
pyplot.scatter(dfPrices["age"], dfPrices["bought_insurance"], marker='*')
pyplot.show(block=False)
pyplot.pause(2)
pyplot.close()

# Split train-test and train logistic regression model
X = dfPrices[['age']]
y = dfPrices.bought_insurance
XTrain, XTest, yTrain, yTest = model_selection.train_test_split(X,y,test_size=0.3)
logRegObj = linear_model.LogisticRegression()
logRegObj.fit(XTrain, yTrain)
# print(logRegObj.coef_)
print(logRegObj.predict(XTest))
print(logRegObj.predict_proba(XTest))
print(yTest)