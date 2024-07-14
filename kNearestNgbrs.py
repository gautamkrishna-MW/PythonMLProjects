
import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris()
petalDF = pd.DataFrame(iris.data, columns=iris.feature_names)
petalDF['target'] = iris.target
print(petalDF.columns)

# Split the dataset
xInput = petalDF.drop("target", axis='columns')
yInput = petalDF["target"]
print(xInput)
print(yInput)
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(xInput, yInput, test_size=0.2)

# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knnObj = KNeighborsClassifier(n_neighbors=3)
knnObj.fit(xTrain, yTrain)
print(f"KNN Score: {knnObj.score(xTest, yTest)}")
  