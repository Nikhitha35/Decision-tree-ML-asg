import pandas as pd
import numpy as np
data = pd.read_csv('/content/PlayTennis-2.csv')
print(data.head())
#(b)using lable encoding
from sklearn.preprocessing import LabelEncoder

# Encode categorical features into numeric
label_encoder = LabelEncoder()

for column in data.columns:
    data[column] = label_encoder.fit_transform(data[column])

print(data.head())
#(c)decision treeclassifier creation
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Separate features and target
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Target (the last column)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Decision Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy')  # Using 'entropy' for ID3 algorithm
clf.fit(X_train, y_train)

# Test the classifier
accuracy = clf.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
from sklearn.tree import export_graphviz
import graphviz
# Changed 'tree' to 'clf' to reference the trained classifier
dot_data=export_graphviz(clf,out_file=None,feature_names=X.columns,filled=True,rounded=True,special_characters=True) 
graph=graphviz.Source(dot_data)
graph
