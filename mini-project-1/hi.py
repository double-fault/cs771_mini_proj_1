import pandas as pd
import matplotlib.pyplot as plt
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

def perform(X_train, X_test, y_train, y_test):  
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')
    return accuracy

# Assuming X is your input data (size: n_samples x 786) and y is the binary target

stps = [0.2, 0.4, 0.6, 0.8, 1.0]

print(train_feat_X.shape)

train_feat_X = np.max(train_feat_X, axis=-1)
test_feat_X = np.max(test_feat_X, axis=-1)

X_train = train_feat_X.reshape(train_feat_X.shape[0], -1)
X_test = test_feat_X.reshape(test_feat_X.shape[0], -1)
y_train =   train_feat_Y
y_test = test_feat_Y
size = len(X_train)

x = []
y = []

for stp in stps:
    n = int(size * stp)
    x.append(stp * 100)
    y.append(perform(X_train[0:n], X_test, y_train[0:n], y_test) * 100)

plt.plot(x, y, marker='o')

# Label the axes
plt.xlabel('X values')
plt.ylabel('Y values')

# Add a title
plt.title('Line Plot of (x, y) Values')

# Show the plot
plt.grid(True)
plt.show()


# Split data into train and test sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the logistic regression model

