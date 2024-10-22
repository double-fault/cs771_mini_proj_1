# CS771 Mini-Project 1: Regraussaurs

The project requires Tensorflow, scikit-learn, matplotlib, numpy and pandas. Please ensure a tensorflow version of at least 2.17.0.

Please place the dataset in a `datasets` directory, as shown by the directory structure below. 

Run `48.py` to get the test set outputs. We have not used pre-frozen weights for any of our models, and as we have used a CNN for Task 1 Dataset 3, this file may take a few minutes to run.

Run `validation.py` to run the 4 final models with varying amount of training data and to get various statistics based on the validation dataset (confusion matrices etc.).

# Directory Structure

├── 48.py
├── datasets
│   ├── test
│   │   ├── test_emoticon.csv
│   │   ├── test_feature.npz
│   │   └── test_text_seq.csv
│   ├── train
│   │   ├── train_emoticon.csv
│   │   ├── train_feature.npz
│   │   └── train_text_seq.csv
│   └── valid
│       ├── valid_emoticon.csv
│       ├── valid_feature.npz
│       └── valid_text_seq.csv
├── read_data.py
├── README
└── validation.py

