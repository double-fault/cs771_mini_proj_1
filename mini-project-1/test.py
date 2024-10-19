import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# read feature dataset
train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")

train_seq_X = train_seq_df['input_str'].tolist()
train_seq_X = [[int(x) for x in e] for e in train_seq_X]
train_seq_Y = train_seq_df['label'].tolist()
test_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
test_seq_X = test_seq_df['input_str'].tolist()
test_seq_X = [[int(x) for x in e] for e in test_seq_X]
test_seq_Y = test_seq_df['label'].tolist()

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

stps = [1.0]

X_train = np.array(train_seq_X, dtype='float64')
X_test = np.array(test_seq_X, dtype='float64')
y_train =np.array(train_seq_Y, dtype='float64')
y_test = np.array(test_seq_Y, dtype='float64')
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)
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

