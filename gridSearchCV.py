
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['flower'] = iris.target
df['flower'] = df['flower'].apply(lambda x:iris.target_names[x])
print(df.columns)

# Create numerical values for categorical variables
from sklearn.preprocessing import LabelEncoder
lbObj = LabelEncoder()
dummyVals = lbObj.fit_transform(df.flower)
df['flower_categories'] = dummyVals
newDF = df.drop('flower',axis='columns', inplace=False)
print(newDF)

# Data splitting
xInp = newDF[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
yInp = newDF[['flower_categories']]
xTrain, xTest, yTrain, yTest = train_test_split(xInp, yInp, test_size=0.3)

# GridsearchCV of SVM (tries all parameter combinations)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
clf = GridSearchCV(SVC(gamma='auto'), {'C':[1,10,20], 'kernel':['rbf','linear']}, cv=5, return_train_score=False)
clf.fit(xTrain, yTrain)
print(pd.DataFrame(clf.cv_results_))

# RandomsearchCV of SVM (tries random parameter combinations, given the number of combinations (n_iter))
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
clf = RandomizedSearchCV(SVC(gamma='auto'), {'C':[1,10,20], 'kernel':['rbf','linear']}, cv=5, n_iter=5, return_train_score=False)
clf.fit(xTrain, yTrain)
print(pd.DataFrame(clf.cv_results_))