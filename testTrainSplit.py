

import pandas as pd
from matplotlib import pyplot
from sklearn.model_selection import train_test_split

dfPrices = pd.read_csv("carprices.csv")
print(dfPrices)

# Show the plots
pyplot.subplot(1,2,1)
pyplot.scatter(dfPrices['Mileage'], dfPrices['Sell Price($)'])
pyplot.subplot(1,2,2)
pyplot.scatter(dfPrices['Age(yrs)'], dfPrices['Sell Price($)'])
pyplot.show(block=False)
pyplot.pause(1)
pyplot.close()

# Train-test splitting
X = dfPrices[["Mileage", "Age(yrs)"]]
y = dfPrices["Sell Price($)"]
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2)

# Running linear regressor
from sklearn import linear_model
linRegObj = linear_model.LinearRegression()
linRegObj.fit(XTrain, yTrain)

print(linRegObj.predict(XTest))
print(yTest)
print(linRegObj.score(XTrain, yTrain))