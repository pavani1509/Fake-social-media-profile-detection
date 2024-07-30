import pandas as pd
import numpy as np
from nltk.corpus import names
import nltk
from sklearn.neighbors import KNeighborsClassifier

genuine_users = pd.read_csv("data/users.csv")
fake_users = pd.read_csv("data/fusers.csv")
x = pd.concat([genuine_users, fake_users])
X = pd.DataFrame(x)
# Label assignment
t = len(fake_users) * ['Genuine'] + len(genuine_users) * ['Fake']
er = pd.Series(t)
X['label'] = pd.DataFrame(er)
# Map labels to numerical values
label = {'Genuine': 0, 'Fake': 1}
X['label'] = [label[item] for item in X['label']]
# Gender classification using NLTK (you should have NLTK installed)
def gender_features(word):
 return {'last_letter': word[-1]}
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
[(name, 'female') for name in names.words('female.txt')])
featuresets = [(gender_features(n), gender) for (n, gender) in
labeled_names]
classifier = nltk.NaiveBayesClassifier.train(featuresets)
# Classify gender for names in the dataset
a = []
for i in X['name']:
 vf = classifier.classify(gender_features(i))
 a.append(vf)
X['gender'] = pd.DataFrame(a)
# Encoding language
lang_list = list(enumerate(np.unique(X['lang'])))
lang_dict = {name: i for i, name in lang_list}
X['lang_code'] = X['lang'].map(lang_dict).astype(int)
# Feature selection
feature_columns_to_use = ['name', 'gender', 'statuses_count',
'followers_count', 'friends_count', 'favourites_count', 'listed_count']
ty = X.loc[:, feature_columns_to_use].values
# KNN Classifier
knn = KNeighborsClassifier()
# User input for the new account
name = input("Enter the name:")
gender = classifier.classify(gender_features(name))
statuses_count = int(input("statuses_count:"))
followers_count = int(input("followers_count:"))
friends_count = int(input("friends_count:"))
favourites_count = int(input("favourites_count:"))
listed_count = int(input("listed_count:"))
lang_code = -int(input("lang_code:"))
new_data = {"name": name, "gender": gender, "statuses_count":
statuses_count, "followers_count": followers_count,
"friends_count": friends_count, "favourites_count":
favourites_count, "listed_count": listed_count,
"lang_code": lang_code}
# Create a DataFrame from the new data
new_df = pd.DataFrame(new_data, index=[0])
# Save the new data to a CSV file
new_df.to_csv("new.csv", index=False)

# Read the new data from the CSV file
re = pd.read_csv("new.csv")
rs = re.loc[:, feature_columns_to_use].values
# Fit the KNN classifier on the existing data
knn.fit(ty, X['label'])
# Predict using KNN
prediction = knn.predict(rs)
print("Prediction:", prediction)