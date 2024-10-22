ygn = __import__ ('48')

from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    stps = [20, 40, 60, 80, 100]

    x = []
    y = [[],[],[],[]]

    for stp in stps:
        print("\n----Training on " + str(stp) + "% of data----")
        x.append(stp)

        test_feat_X = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)['features']
        test_emoticon_X = pd.read_csv("datasets/valid/valid_emoticon.csv")['input_emoticon'].tolist()
        test_seq_X = pd.read_csv("datasets/valid/valid_text_seq.csv")['input_str'].tolist()
        
        feature_model = ygn.FeatureModel(stp)
        text_model = ygn.TextSeqModel(stp)
        emoticon_model  = ygn.EmoticonModel(stp)
        best_model = ygn.CombinedModel(stp)
        
        pred_feat = feature_model.predict(test_feat_X)
        pred_emoticons = emoticon_model.predict(test_emoticon_X)
        pred_text = text_model.predict(test_seq_X)
        pred_combined = best_model.predict(test_feat_X, test_emoticon_X, test_seq_X)

        test_emo_df = pd.read_csv("datasets/valid/valid_emoticon.csv")
        test_emo_Y = test_emo_df['label'].tolist()

        test_feat = np.load("datasets/valid/valid_feature.npz", allow_pickle=True)
        test_feat_Y = test_feat['label']

        test_seq_df = pd.read_csv("datasets/valid/valid_text_seq.csv")
        test_seq_Y = test_seq_df['label'].tolist()

        print("(1) Features dataset: ")
        accuracy = accuracy_score(pred_feat, test_feat_Y)
        y[0].append(accuracy * 100)
        print(f'Accuracy: {accuracy}')

        conf_matrix = confusion_matrix(pred_feat, test_feat_Y)
        print(f'Confusion Matrix:\n{conf_matrix}')

        print("(2) Emoticon dataset: ")
        accuracy = accuracy_score(pred_emoticons, test_emo_Y)
        y[1].append(accuracy * 100)
        print(f'Accuracy: {accuracy}')

        conf_matrix = confusion_matrix(pred_emoticons, test_emo_Y)
        print(f'Confusion Matrix:\n{conf_matrix}')

        print("(3) Sequences dataset: ")
        accuracy = accuracy_score(pred_text, test_seq_Y)
        y[2].append(accuracy * 100)
        print(f'Accuracy: {accuracy}')

        conf_matrix = confusion_matrix(pred_text, test_seq_Y)
        print(f'Confusion Matrix:\n{conf_matrix}')

        print("(4) Combined dataset: ")
        accuracy = accuracy_score(pred_combined, test_feat_Y)
        y[3].append(accuracy * 100)
        print(f'Accuracy: {accuracy}')

        conf_matrix = confusion_matrix(pred_combined, test_emo_Y)
        print(f'Confusion Matrix:\n{conf_matrix}')

    # Plot each line
    plt.plot(x, y[0], label='Features', color='blue')
    plt.plot(x, y[1], label='Emoticon', color='green')
    plt.plot(x, y[2], label='Sequence', color='red')
    plt.plot(x, y[3], label='Combined', color='orange')

    # Add labels and title
    plt.xlabel('Percentage of training data used')
    plt.ylabel('Accuracy')
    plt.title('Accuracy comparison of 4 models')

    # Add legend
    plt.legend()

    # Show plot
    plt.show()

    
