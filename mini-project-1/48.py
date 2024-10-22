import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Conv1D, Dropout, MaxPooling1D, Flatten, Dense, Embedding
from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Base class
class MLModel():
    def __init__(self) -> None:
        pass
    
    def train(self, X, y):
        NotImplemented
    
    def predict(self, X):
        NotImplemented

# Task 1 Dataset 3: Our model is a Convolutional Neural Network 
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

    def predict(self, X):
        X = [[int(e) for e in f] for f in X] 
        X = np.array(X)
        ret = self.model.predict(X)
        ret = (ret > 0.5).astype(int)
        return ret

# Task 1 Dataset 1: Our model is a deep neural network
class EmoticonModel(MLModel):
    def __init__(self, pct=100):
        self.train_data = pd.read_csv("datasets/train/train_emoticon.csv")

        n = len(self.train_data)
        n = int(n * pct / 100)

        self.train_data = self.train_data[0:n]

        self.model = self.build_model()
        self.train()

    def create_emoji_dict(self):
        emoji_dict = {}
        for i in range(len(self.train_data['input_emoticon'])):
            for emoji in self.train_data['input_emoticon'][i]:
                if emoji not in emoji_dict:
                    emoji_dict[emoji] = len(emoji_dict)

        emoji_dict['<UNK>'] = len(emoji_dict)
        return emoji_dict

    def emoji_string_to_indices(self, emoji_string, emoji_dict):
        indices = []
        for emoji in emoji_string:
            if emoji in emoji_dict:
                indices.append(emoji_dict[emoji])
            else:
                indices.append(-1)  # Use -1 or any other placeholder for unknown emojis
        return indices

    def preprocess_data(self, data):
        emoji_dict = self.create_emoji_dict()
        numeric_data = [self.emoji_string_to_indices(sample, emoji_dict) for sample in data]
        numeric_data_arr = np.array(numeric_data)
        return numeric_data_arr

    def build_model(self):
        model = Sequential()
        # Embedding layer: convert 13 emoji Unicode integers into dense vectors (e.g., 50 dimensions)
        model.add(Embedding(input_dim=250, output_dim=10, input_length=13))  
        model.add(Flatten())

        # Dense layers for classification
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(16, activation='relu'))
        model.add(Dropout(0.5))

        # Output layer for binary classification
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.build(input_shape=(None, 13))
        return model

    def train(self):
        numeric_data_arr = self.preprocess_data(self.train_data['input_emoticon'])
        X_train = numeric_data_arr
        y_train = self.train_data['label']
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.model.fit(X_train, y_train, epochs=15, batch_size=32)

    def predict(self, X):
        data = self.preprocess_data(X)
        ret = self.model.predict(data)
        ret = (ret > 0.5).astype(int)

        return ret

# Task 1 Dataset 2: Our model is logistic regression
class FeatureModel(MLModel):
    def __init__(self, pct=100) -> None:
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

    def predict(self, X): 
        X = X.reshape(X.shape[0], -1)
        ret = self.model.predict(X)
        return ret

# Task 2: Our model is logistic regression on a combination of dataset 1 and 2
class CombinedModel(MLModel):
    def __init__(self, pct=100) -> None:
        # read emoticon dataset
        train_emoticon_df = pd.read_csv("datasets/train/train_emoticon.csv")

        n = len(train_emoticon_df)
        n = int(n * pct / 100)
        train_emoticon_df = train_emoticon_df[:n]

        train_emoticon_X = train_emoticon_df['input_emoticon'].tolist()
        train_emoticon_Y = train_emoticon_df['label'].tolist()

        # read text sequence dataset
        train_seq_df = pd.read_csv("datasets/train/train_text_seq.csv")

        n = len(train_seq_df)
        n = int(n * pct / 100)
        train_seq_df = train_seq_df[:n]

        train_seq_X = train_seq_df['input_str'].tolist()
        train_seq_Y = train_seq_df['label'].tolist()

        # read feature dataset
        train_feat = np.load("datasets/train/train_feature.npz", allow_pickle=True)

        train_feat_X = train_feat['features']
        train_feat_Y = train_feat['label']

        n = len(train_feat_X)
        n = int(n * pct / 100)
        train_feat_X = train_feat_X[:n]
        train_feat_Y = train_feat_Y[:n]

        self.n_components = 60  # Number of components you want to preserve
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)

        # Reshape the training data to (7080 * 13, 768)
        train_feat_X_reshaped = train_feat_X.reshape(-1, 768)  # Shape will be (7080*13, 768)


        # Standardize the reshaped training data
        train_feat_X_scaled = self.scaler.fit_transform(train_feat_X_reshaped)

        # Apply PCA
        train_X_pca = self.pca.fit_transform(train_feat_X_scaled)

        # Reshape back to (7080, 13, 50)
        X_pca = train_X_pca.reshape(n, 13, self.n_components)

        ds2 = X_pca.reshape(X_pca.shape[0], -1)

        train_emo_X = train_emoticon_df['input_emoticon'].tolist()
        train_emo_X = np.array([[ord(x) for x in e] for e in train_emo_X])
        train_emo_Y = train_emoticon_df['label'].tolist()

        self.emoji_encoder = LabelEncoder()

        # Flatten the list of emojis and fit the label encoder
        # Since train_emo_X is 2D (13, 786), we need to flatten it for unique emojis
        flat_train_emojis = [emoji for sublist in train_emo_X for emoji in sublist]
        self.emoji_encoder.fit(flat_train_emojis)
        self.emoji_set = set(flat_train_emojis)

        # Transform the training emojis to their coresponding numerical labels
        train_emo_X_encoded = np.array([[self.emoji_encoder.transform([x])[0] for x in e] for e in train_emo_X])

        X_train_seq = np.array([[int(char) for char in sequence] for sequence in train_seq_X])

        X_train_final = np.concatenate((np.array(train_emo_X_encoded), ds2 , np.array(X_train_seq)), axis=1)

        self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
        self.model.fit(X_train_final, train_seq_Y)


    def encode_with_unknown(self, emoji_list, encoder, known_set, unknown_label=-1):
        encoded_list = []
        for emoji_seq in emoji_list:
            encoded_seq = []
            for emoji in emoji_seq:
                if emoji in known_set:
                    # Encode with the existing LabelEncoder
                    encoded_seq.append(encoder.transform([emoji])[0])
                else:
                    # Assign the unknown label if the emoji is not known
                    encoded_seq.append(unknown_label)
            encoded_list.append(encoded_seq)
        return np.array(encoded_list)

    def predict(self, X1, X2, X3): 
        valid_feat_X = X1
        valid_seq_X = X3
        valid_emoticon_X = X2

        # Now handle the validation data similarly
        valid_feat_X_reshaped = valid_feat_X.reshape(-1, 768)  # Shape will be (len(valid_feat_X)*13, 768)
        valid_feat_X_scaled = self.scaler.transform(valid_feat_X_reshaped)

        # Apply PCA on the validation data
        valid_X_pca = self.pca.transform(valid_feat_X_scaled)

        # Reshape back to (len(valid_feat_X), 13, 50)
        valid_X_pca = valid_X_pca.reshape(len(valid_feat_X), 13, self.n_components)
   
        valid_ds2 = valid_X_pca.reshape(valid_X_pca.shape[0], -1)
    
        valid_emo_X = valid_emoticon_X
        valid_emo_X = np.array([[ord(x)  for x in e] for e in valid_emo_X])

        valid_emo_X_encoded = self.encode_with_unknown(valid_emo_X, self.emoji_encoder, self.emoji_set)
        #valid_emo_X_encoded = np.array([[self.emoji_encoder.transform([x])[0] for x in e] for e in valid_emo_X])

        X_valid_seq = np.array([[int(char) for char in sequence] for sequence in valid_seq_X])

        X_valid_final = np.concatenate((np.array(valid_emo_X_encoded), valid_ds2 , np.array(X_valid_seq)), axis=1)

        return self.model.predict(X_valid_final)

def save_predictions_to_file(predictions, filename):
    with open(filename, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

if __name__ == '__main__':
    # read datasets
    test_feat_X = np.load("datasets/test/test_feature.npz", allow_pickle=True)['features']
    test_emoticon_X = pd.read_csv("datasets/test/test_emoticon.csv")['input_emoticon'].tolist()
    test_seq_X = pd.read_csv("datasets/test/test_text_seq.csv")['input_str'].tolist()
    
    # the trained models 
    feature_model = FeatureModel()
    text_model = TextSeqModel()
    emoticon_model  = EmoticonModel()
    best_model = CombinedModel()
    
    # predictions from the trained models
    pred_feat = feature_model.predict(test_feat_X)
    pred_emoticons = emoticon_model.predict(test_emoticon_X)
    pred_text = text_model.predict(test_seq_X)
    pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)
    
    # saving prediction to text files
    save_predictions_to_file(pred_feat, "pred_feat.txt")
    save_predictions_to_file(pred_emoticons, "pred_emoticon.txt")
    save_predictions_to_file(pred_text, "pred_text.txt")
    save_predictions_to_file(pred_combined, "pred_combined.txt")
    
    
