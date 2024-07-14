
from sklearn.datasets import load_iris
import pandas as pd

# Create pandas dataframe from input data.
iris = load_iris()
petalDF = pd.DataFrame(iris.data, columns=iris.feature_names)
petalDF['target'] = iris.target
print(petalDF)

# Split the dataset
xInput = petalDF.drop("target", axis='columns')
yInput = petalDF["target"]
print(xInput)
print(yInput)
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xInput, yInput, test_size=0.2)

# Create SVM and train
from sklearn.svm import SVC
modelObj = SVC(gamma=15, C=5)
modelObj.fit(xTrain, yTrain)
print(f"SVM Accuracy: {modelObj.score(xTest, yTest)}")