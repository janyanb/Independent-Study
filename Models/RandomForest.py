import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

 
# 1. Data Loading 
def load_csv_data(csv_dir, selected_columns):

    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("Merged") and f.endswith(".csv")]
    csv_files.sort()
    
    dfs = []
    for file in csv_files:
        file_path = os.path.join(csv_dir, file)
        df_temp = pd.read_csv(file_path, usecols=selected_columns)
        dfs.append(df_temp)
    
    df = pd.concat(dfs, ignore_index=True)
    return df

 
# 2. Preprocessing 
def label_to_num(label):
    """
    Maps a textual label to a numeric value:
    0 for 'benign' and 1 for any attack.
    """
    label_str = str(label)
    return 0 if label_str.strip().lower() == 'benign' else 1

def preprocess_data(df):
   
    df['numeric_label'] = df['Label'].apply(label_to_num)
    
    # One-hot encode the 'Protocol Type' categorical column
    df = pd.get_dummies(df, columns=['Protocol Type'], prefix='Protocol')
    
   
    X = df.drop(columns=['Label', 'numeric_label'])
    y = df['numeric_label']
    
    return X, y

def clean_data(X, y):
   
    print("Checking for infinities or large values in X:")
    print(X.describe())
    print("Infinite values:\n", np.isinf(X).sum())
    
    # Replace infinities with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Cap extreme values using max float32 limits
    max_float32 = np.finfo(np.float32).max
    X = X.clip(lower=-max_float32, upper=max_float32)
    
    # Drop rows with NaN values and align y with filtered X
    X = X.dropna()
    y = y[X.index]
    
    return X, y

 
# 3. Data Splitting 
def split_dataset(X, y, test_size=0.2, random_state=42):

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

 
# 4. Model Training 
def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, random_state=42, n_jobs=-1):
    rf_classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

 
# 5. Evaluation 
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nEvaluation Metrics on Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    
    return y_pred

 # 6. Save Model 
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

 # 7. Main Function
 
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
    
    # Load data from CSV files
    df = load_csv_data(csv_dir, selected_columns)
    
    # Preprocess the data
    X, y = preprocess_data(df)
    
    # Clean data for infinities and extreme values
    X, y = clean_data(X, y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Train the Random Forest model
    rf_classifier = train_random_forest(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(rf_classifier, X_test, y_test)
    
    # Save the trained model
    save_model(rf_classifier, 'random_forest_model.pkl')

if __name__ == "__main__":
    main()
