
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import pandas as pd

df = pd.read_csv("salaries.csv")

# Create input and outputs
X = df.drop("100k_greater", axis='columns')
y = df["100k_greater"]

# Convert labels to numerical inputs
leObj = LabelEncoder()
newDFComp = leObj.fit_transform(df['company'])
newDFJob = leObj.fit_transform(df['job'])
newDFDegree = leObj.fit_transform(df['degree'])
newDF = pd.DataFrame({"Company":newDFComp, "Job":newDFJob, "Degree":newDFDegree})

# Split dataset and perform classification using decision-trees
xTrain, xTest, yTrain, yTest = train_test_split(newDF, y, test_size=0.2)
modelObj = tree.DecisionTreeClassifier()
modelObj.fit(xTrain,yTrain)
print(f"Score of decision Tree: {modelObj.score(xTest, yTest)}")
print(modelObj.predict(xTest))
print(yTest)
