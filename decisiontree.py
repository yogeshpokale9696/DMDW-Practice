import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.tree as tree

df=pd.read_csv("data.csv")
print(df)
print(df.info())
df.drop(["Unnamed: 32"],axis=1)

diagnosis =df['diagnosis']
features= df.drop('diagnosis', axis=1)
# splitting the data into training and testing sets
features_train, features_test, labels_train, labels_test =train_test_split(features, diagnosis, test_size = 0.1)



#  Delete NaN Items from Dataset
#indexNames = df[df['Column'] == "nan" ].index
#df.drop(indexNames , inplace=True)




# creating the classifier

clf = tree.DecisionTreeClassifier (min_samples_split = 10)
clf-clf.fit(features_train, labels_train)
clf.get_params()
