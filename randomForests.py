import pandas as pd
import seaborn as sb
from matplotlib import pyplot
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Create pandas dataframe
digits = load_digits()
digitDF = pd.DataFrame(digits.data)
digitDF['target'] = digits.target
print(digitDF)

# Split train-test data
xInp = digitDF.drop('target',axis='columns')
yInp = digitDF['target']
xTrain, xTest, yTrain, yTest = train_test_split(xInp, yInp, train_size=0.6)

# Create model and train
modelObj = RandomForestClassifier(n_estimators=20)
modelObj.fit(xTrain, yTrain)
print(f"RandomForest Accuracy: {modelObj.score(xTest, yTest)}")

# Confusion Matrix and plotting
yPred = modelObj.predict(xTest)
confMat = confusion_matrix(yPred, yTest)
sb.heatmap(confMat, annot=True)
pyplot.show(block=False)
pyplot.pause(5)
pyplot.close()