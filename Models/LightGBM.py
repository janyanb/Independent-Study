import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

 
# 1. Data Loading Function 
def load_csv_files(csv_dir, selected_columns):

    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("Merged") and f.endswith(".csv")]
    csv_files.sort()
    
    dfs = []
    for file in csv_files:
        file_path = os.path.join(csv_dir, file)
        df_temp = pd.read_csv(file_path, usecols=selected_columns)
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    return df

 
# 2. Data Preprocessing Function 
def label_to_num(label):
    """
    Maps a label to a numeric value:
    0 for 'benign', 1 for any attack.
    """
    label_str = str(label)
    return 0 if label_str.strip().lower() == 'benign' else 1

def preprocess_data(df):
    """
    Converts labels to numeric values and handles one-hot encoding of categorical features.
    Returns feature matrix X and target vector y.
    """
    # Convert textual labels to numeric
    df['numeric_label'] = df['Label'].apply(label_to_num)
    
    # One-hot encode the 'Protocol Type' categorical column
    df = pd.get_dummies(df, columns=['Protocol Type'], prefix='Protocol')
    
    # Separate features and labels
    X = df.drop(columns=['Label', 'numeric_label'])
    y = df['numeric_label']
    
    return X, y

def split_dataset(X, y, test_size=0.2, random_state=42):

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

 
# 3. Model Training Functions
 
def create_lgb_datasets(X_train, y_train, X_test, y_test):

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    return train_data, test_data

def train_lightgbm(train_data, test_data, params, num_rounds=100, early_stopping_rounds=10):

    bst = lgb.train(
        params,
        train_data,
        num_rounds,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
    )
    return bst

 
# 4. Evaluation Function
 
def evaluate_model(bst, X_test, y_test, threshold=0.5):

    # Predict probabilities on the test set
    y_pred_proba = bst.predict(X_test)
    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Print evaluation results
    print("\nEvaluation Metrics on Test Set:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    return y_pred, y_pred_proba

 
# 5. Main Function
 
def main():
    # Define CSV directory and columns to load
    csv_dir = "CICIoT2023/"
    selected_columns = [
        'Protocol Type', 'Header_Length', 'Time_To_Live', 'Rate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
        'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
        'TCP', 'UDP', 'ICMP', 'ARP', 'DNS', 'HTTP', 'HTTPS',
        'IAT', 'Tot size', 'AVG', 'Label'
    ]
    
    # Load and preprocess data
    df = load_csv_files(csv_dir, selected_columns)
    print("Data loaded successfully. Shape:", df.shape)
    
    X, y = preprocess_data(df)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # Create LightGBM datasets
    train_data, test_data = create_lgb_datasets(X_train, y_train, X_test, y_test)
    
    # Set LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }
    
    # Train the model
    bst = train_lightgbm(train_data, test_data, params, num_rounds=100, early_stopping_rounds=10)
    
    # Evaluate the model on the test set
    evaluate_model(bst, X_test, y_test)
    
    # Save the trained model to disk
    bst.save_model('lightgbm_model.txt')
    print("Model saved as 'lightgbm_model.txt'")

if __name__ == "__main__":
    main()

