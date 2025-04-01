import glob
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


# 1) Load Data
def load_data(data_path_pattern):
    all_files = glob.glob(data_path_pattern)
    df_list = []
    for file in all_files:
        temp_df = pd.read_csv(file)
        df_list.append(temp_df)
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    print("Loaded data from files:", all_files)
    print("Combined dataset shape:", combined_df.shape)
    return combined_df


# 2) Preprocess Data (Binary labeling)
def preprocess_data(df, label_col='Label', dropna=True):
    """
    1) Convert all non-Benign labels to 'Attack'.
    2) Optional drop rows with NaN or handle them differently.
    3) Separate features (X) and numeric labels (y).
    4) Scale numeric features (StandardScaler).
    """
    # Convert label to binary: 0 for benign, 1 for attack
    df[label_col] = df[label_col].apply(lambda x: 0 if str(x).lower().strip() == "benign" else 1)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if dropna:
        df = df.dropna()

    # Separate features and label
    # Example: remove 'Label' column to get features
    X = df.drop(columns=[label_col])
    y = df[label_col].values  # array of 0/1

    # Scale numeric features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# 3) Split Data
def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into train and test sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Helps preserve class distribution
    )
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Test set shape:    ", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test


# 4) Build Binary Model
def build_binary_model(input_dim, hidden_units=128):
    """
    Builds a simple 2-layer fully-connected Neural Network 
    for binary classification (Benign vs Attack).
    """
    model = Sequential()
    model.add(Dense(hidden_units, activation='relu', input_shape=(input_dim,)))
    model.add(Dense(hidden_units, activation='relu'))
    # Single output neuron with sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Use binary crossentropy for binary classification
    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=1e-3),
        metrics=['accuracy']
    )
    return model



# 5) Train Model
def train_model(model, X_train, y_train, val_split=0.1, epochs=20, batch_size=256):
    """
    Trains the neural network model for binary classification.
    """
    history = model.fit(
        X_train,
        y_train,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history



# 6) Evaluate Model
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set using Keras metrics (loss, accuracy).
    """
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    return test_loss, test_accuracy


# 7) Compute Classification Metrics
def compute_metrics(model, X_test, y_test):
    """
    Computes Accuracy, Precision, Recall, F1-score for binary classification.
    """
    # Predict probabilities
    y_probs = model.predict(X_test)
    # Convert probabilities to 0/1 predictions
    y_pred = (y_probs >= 0.5).astype(int).reshape(-1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='binary', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1   = f1_score(y_test, y_pred, average='binary', zero_division=0)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f} (binary)")
    print(f"Recall:    {rec:.4f} (binary)")
    print(f"F1-Score:  {f1:.4f} (binary)")

    results = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    return results


# 8) Save Model
def save_model(model, file_path='trained_model.h5'):
    model.save(file_path)
    print(f"Model saved to {file_path}")


# MAIN WORKFLOW
def main():
    """
    Main workflow for binary classification:
      1) Load data.
      2) Preprocess data (binary labels).
      3) Split into train/test.
      4) Build model.
      5) Train model.
      6) Evaluate model 
      7) Compute classification metrics.
      8) Save model.
    """
    data_path_pattern = "CICIoT2023/Merged*.csv"

    # 1) Load data
    df = load_data(data_path_pattern)

    # 2) Preprocess data (binary labeling)
    X_scaled, y, scaler = preprocess_data(df, label_col='Label', dropna=True)

    # 3) Split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # 4) Build model
    num_features = X_train.shape[1]
    model = build_binary_model(input_dim=num_features, hidden_units=128)

    # 5) Train model
    _ = train_model(model, X_train, y_train, val_split=0.1, epochs=20, batch_size=256)

    # 6) Evaluate model
    evaluate_model(model, X_test, y_test)

    # 7) Additional classification metrics
    compute_metrics(model, X_test, y_test)

    # 8) Save model
    save_model(model, "trained_model.h5")


if __name__ == "__main__":
    main()
