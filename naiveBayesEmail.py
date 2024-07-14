
import pandas as pd

df = pd.read_csv('spam.csv')

# Convert categorical variables into numbers
print(df.head())
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
print(df.head())

# Train-test split
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(df.Message, df.spam, test_size=0.3)

# Create features from text
from sklearn.feature_extraction.text import CountVectorizer
cvObj = CountVectorizer()
xTrainCount = cvObj.fit_transform(xTrain.values)
xTestCount = cvObj.transform(xTest.values)

# Run Classifier
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(xTrainCount, yTrain)
print(model.score(xTestCount, yTest))

# Pipeline for feature extraction and classification
from sklearn.pipeline import Pipeline
modelPipeline = Pipeline([('Vectoriser', CountVectorizer()),('nb', MultinomialNB())])
modelPipeline.fit(xTrain, yTrain) # Feed directly the input without vectorzing
print(f"Pipeline score: {modelPipeline.score(xTest, yTest)}")