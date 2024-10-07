import pandas as pd
import numpy as np

# read feature dataset
train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
train_feat_X = train_feat['features']
train_feat_Y = train_feat['label']

test_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
test_feat_X = test_feat['features']
test_feat_Y = test_feat['label']

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assuming X is your input data (size: n_samples x 786) and y is the binary target

X_train = train_feat_X.reshape(train_feat_X.shape[0], -1)
X_test = test_feat_X.reshape(test_feat_X.shape[0], -1)
y_train =   train_feat_Y
y_test = test_feat_Y

# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

