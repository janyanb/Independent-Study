import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
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

    label_str = str(label)
    return 0 if label_str.strip().lower() == 'benign' else 1

def preprocess_data(df):
 
    # Convert labels to numeric values
    df['numeric_label'] = df['Label'].apply(label_to_num)
    
    # Select features (all columns except 'Label' and 'numeric_label')
    X = df.drop(columns=['Label', 'numeric_label'])
    y = df['numeric_label']
    
    return X, y

 
# 3. Splitting the Data
 
def split_data(X, y, test_size=0.2, random_state=42):

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

 
# 4. Model Training 
def train_logistic_regression(X_train, y_train, max_iter=1000):
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, y_train)
    return model

 
# 5. Evaluation
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    return y_pred

 
# 6. Save Model
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"Model saved as '{filename}'")

 
# 7. Main Function 
def main():
    # Define CSV directory and columns
    csv_dir = "CICIoT2023/"
    selected_columns = [
        'Header_Length', 'Time_To_Live', 'Rate', 'fin_flag_number', 'syn_flag_number',
        'rst_flag_number', 'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
        'cwr_flag_number', 'TCP', 'UDP', 'ICMP', 'ARP', 'DNS', 'HTTP', 'HTTPS',
        'IAT', 'Tot size', 'AVG', 'Label'
    ]
    
    # Load data from CSV files
    df = load_csv_data(csv_dir, selected_columns)
    print(f"Loaded data shape: {df.shape}")
    
    # Preprocess data to get features and labels
    X, y = preprocess_data(df)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")
    
    # Train the Logistic Regression model
    model = train_logistic_regression(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)
    
    # Save the trained model
    save_model(model, 'logistic_regression_model.pkl')

if __name__ == "__main__":
    main()
