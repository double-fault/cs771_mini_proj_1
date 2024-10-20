import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense
from tensorflow.keras import layers, models
import tensorflow as tf

# these are dummy models
class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        NotImplemented
    
    def predict(self, X):
        NotImplemented
    
class TextSeqModel(MLModel):
    def __init__(self, pct=100) -> None:
        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")
        train_seq_X = train_seq_df['input_str'].tolist()
        train_seq_Y = train_seq_df['label'].tolist()

        n = len(train_seq_X)
        n = int(n * pct / 100)

        train_seq_X = train_seq_X[:n]
        train_seq_Y = train_seq_Y[:n]

        X_train = [[int(char) for char in sequence] for sequence in train_seq_X]

        self.model = self.build_cnn()

        X_train = np.array(X_train)
        Y_train = np.array(train_seq_Y)

        self.model.fit(X_train, Y_train, epochs=100, batch_size=32, verbose=0)

    def build_cnn(self):
        n_timesteps, n_features, n_outputs = 50,1,1

        model = Sequential()
        model.add(layers.Embedding(input_dim=10, output_dim=10, input_length=50))  # For 10 unique characters

        model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(7, activation='relu'))
        model.add(Dense(n_outputs, activation='sigmoid'))  # sigmoid for binary classification
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def predict(self, X):# random predictions
        X = [[int(e) for e in f] for f in X] 
        X = np.array(X)
        #X = X.reshape(X.shape[0], -1)
        ret = self.model.predict(X)
        ret = (ret > 0.5).astype(int)
        return ret
    
class EmoticonModel(MLModel):
    def __init__(self, pct=100) -> None:
        pass

    def predict(self, X):# random predictions
        return np.random.randint(0,2,(len(X)))
    
class FeatureModel(MLModel):
    def __init__(self, pct=100) -> None:
        # read feature dataset
        train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        train_feat_X = train_feat['features']
        train_feat_Y = train_feat['label']

        n = train_feat_X.shape[0]
        n = int(n * pct / 100)

        train_feat_X = train_feat_X[0:n]
        train_feat_Y = train_feat_Y[0:n]

        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
        
        X_train = train_feat_X.reshape(train_feat_X.shape[0], -1)
        y_train = train_feat_Y

        self.model.fit(X_train, y_train)

    def predict(self, X): # random predictions
        X = X.reshape(X.shape[0], -1)
        ret = self.model.predict(X)
        return ret
    
class CombinedModel(MLModel):
    def __init__(self, pct=100) -> None:
        train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)
        train_feat_X = train_feat['features']
        train_feat_Y = train_feat['label']

        n = train_feat_X.shape[0]
        n = int(n * pct / 100)

        train_feat_X = train_feat_X[0:n]
        train_feat_Y = train_feat_Y[0:n]

        train_seq_df = pd.read_csv("datasets/train/train_emoticon.csv")

        train_emo_X = train_seq_df['input_emoticon'].tolist()
        train_emo_X = [[ord(x)  for x in e] for e in train_emo_X]
        train_emo_Y = train_seq_df['label'].tolist()

        X_train = np.array(train_emo_X, dtype='float64')
        y_train =np.array(train_emo_Y, dtype='float64')

        m = X_train.shape[0]
        m = int(m * pct / 100)

        X_train = X_train[0:m]
        y_train = y_train[0:m]

        m = train_feat_X.shape[0]
        m = int(m * pct / 100)
        train_feat_x = train_feat_X[0:m]

        X_train = X_train.reshape(X_train.shape[0], -1)

        X_train2 = train_feat_X.reshape(train_feat_X.shape[0], -1)

        X_train = np.concatenate((X_train, X_train2), axis=1)

        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))

        self.model.fit(X_train, y_train)

    def predict(self, X1, X2, X3): 
        X2 = [[ord(x) for x in e] for e in X2]
        X2 = np.array(X2, dtype='float64')

        X1 = X1.reshape(X1.shape[0], -1)
        X2 = X2.reshape(X2.shape[0], -1)
        
        X = np.concatenate((X2, X1), axis=1)
        return self.model.predict(X)
    
    
def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # your trained models 
    feature_model = FeatureModel()
    text_model = TextSeqModel()
    emoticon_model  = EmoticonModel()
    best_model = CombinedModel()
    
    # predictions from your trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
    
    
