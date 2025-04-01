import os
import numpy as np
import pandas as pd
import keras
import keras_hub
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Data loading
def load_merged_csv_data(csv_dir, selected_columns):

    # Find CSV files in the directory
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("Merged") and f.endswith(".csv")]
    csv_files.sort()

    dfs = []
    for file in csv_files:
        file_path = os.path.join(csv_dir, file)
        df_temp = pd.read_csv(file_path, usecols=selected_columns)
        dfs.append(df_temp)

    # Concatenate all DataFrames into one
    df = pd.concat(dfs, ignore_index=True)
    return df

 
# 2. Preprocessing 
def row_to_text(row):
    """
    Converts one row of the DataFrame into a single text string.
    """
    return (
        f"Protocol: {row['Protocol Type']}, "
        f"Header_Length: {row['Header_Length']}, "
        f"Time_To_Live: {row['Time_To_Live']}, "
        f"Rate: {row['Rate']}, "
        f"fin_flag_number: {row['fin_flag_number']}, "
        f"syn_flag_number: {row['syn_flag_number']}, "
        f"rst_flag_number: {row['rst_flag_number']}, "
        f"psh_flag_number: {row['psh_flag_number']}, "
        f"ack_flag_number: {row['ack_flag_number']}, "
        f"ece_flag_number: {row['ece_flag_number']}, "
        f"cwr_flag_number: {row['cwr_flag_number']}, "
        f"TCP: {row['TCP']}, "
        f"UDP: {row['UDP']}, "
        f"ICMP: {row['ICMP']}, "
        f"ARP: {row['ARP']}, "
        f"DNS: {row['DNS']}, "
        f"HTTP: {row['HTTP']}, "
        f"HTTPS: {row['HTTPS']}, "
        f"IAT: {row['IAT']}, "
        f"Tot size: {row['Tot size']}, "
        f"AVG: {row['AVG']}"
    )

def label_to_num(label):
    """
    Converts the 'Label' field to 0 (Benign) or 1 (Attack).
    """
    label_str = str(label).strip().lower()
    return 0 if label_str == 'benign' else 1

def create_features_and_labels(df):
    """
    Prepares the text data (features) and numeric labels from the DataFrame.
    """
    # Convert Label to numeric
    df['numeric_label'] = df['Label'].apply(label_to_num)

    # Create text features from each row
    features = df.apply(row_to_text, axis=1).tolist()
    labels = df['numeric_label'].tolist()
    return features, labels

 
# 3. Model building
 
def build_classifier(num_classes=2, learning_rate=5e-5):
    """
    Builds and compiles a BERT classifier using keras_hub.
    """
    classifier = keras_hub.models.BertClassifier.from_preset(
        "bert_base_en",  # Example BERT model
        num_classes=num_classes
    )
    
    # Compile the classifier
    classifier.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(learning_rate),
        metrics=['accuracy']
    )
    return classifier

 
# 4. Training and evaluation 
def train_and_evaluate(classifier, features_train, labels_train, features_test, labels_test, epochs=1, batch_size=32):

    # Train the model
    classifier.fit(
        x=features_train,
        y=labels_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(features_test, labels_test),
        verbose=1
    )

    # Prediction on test set
    test_predictions = classifier.predict(features_test, batch_size=batch_size)

    # Convert logits to class labels
    test_pred_labels = np.argmax(test_predictions, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(labels_test, test_pred_labels)
    precision = precision_score(labels_test, test_pred_labels)
    recall = recall_score(labels_test, test_pred_labels)
    f1 = f1_score(labels_test, test_pred_labels)

    # Print results
    print("\nEvaluation Metrics on Test Set:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    # Print classification report
    print("\nDetailed Classification Report:")
    print(classification_report(labels_test, test_pred_labels, target_names=['Benign', 'Attack']))

    return classifier

 
# 5. Main script
 
def main():
    """
    Main function that loads data, prepares features, builds, trains, and evaluates the classifier.
    """

    csv_dir = "CICIoT2023/"
    # Columns to load
    selected_columns = [
        'Protocol Type', 'Header_Length', 'Time_To_Live', 'Rate',
        'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number',
        'ack_flag_number', 'ece_flag_number', 'cwr_flag_number',
        'TCP', 'UDP', 'ICMP', 'ARP', 'DNS', 'HTTP', 'HTTPS',
        'IAT', 'Tot size', 'AVG', 'Label'
    ]

    df = load_merged_csv_data(csv_dir, selected_columns)
    print("Data loaded. Columns:", df.columns.tolist())

    # 2. Prepare features and labels
    features, labels = create_features_and_labels(df)

    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(features_train)} | Testing samples: {len(features_test)}")

    # 3. Build model
    classifier = build_classifier(num_classes=2, learning_rate=5e-5)

    # 4. Train and evaluate
    classifier = train_and_evaluate(
        classifier,
        features_train, labels_train,
        features_test, labels_test,
        epochs=1,        
        batch_size=32    
    )

    classifier.save("bert_classifier_model.keras")
    print("Model saved as 'bert_classifier_model.keras'")

    # 6. Example predictions on the first few instances
    predictions = classifier.predict(features[:5], batch_size=32)
    print("\nPredictions for the first 5 instances:")
    print(predictions)

if __name__ == "__main__":
    main()
