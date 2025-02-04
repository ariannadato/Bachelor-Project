import pandas as pd
import glob
import re
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#----------------------------------------------------------------------------------------------
# Organize AlphaFold3 data from summary confidences to train ML model

file_path = "/content/drive/MyDrive/1.Thesis/1_AF3_structures/*/fold_*/*confidences*.json"

data_frame= pd.DataFrame()
files = glob.glob(file_path)
data = {}


for file in files:
    # Extract nanobody + target pair from the file name
    filename = os.path.basename(file)
    name_parts = filename.split('_')
    nanobody_target = f"{name_parts[1]}_{name_parts[2]}"  
    is_true_pair = 'yes' if 'true' in filename.lower() else 'no'  # Check if the file name contains 'true'
    index = int(name_parts[-1].split('.')[0])  # Extract the index number from file name (e.g. "confidences_0.json" -> 0)

    with open(file, 'r') as f:
        json_data = json.load(f)

    if nanobody_target not in data: #Initialize nanobody entry
        data[nanobody_target] = {"True Pair?": is_true_pair}

    for key, value in json_data.items(): #Add values to data dictionary
        if not isinstance(value, list):  # Only handle single (non-list) values
            column_name = f"{key}_{index}"
            data[nanobody_target][column_name] = value

data_frame = pd.DataFrame.from_dict(data, orient='index')

# Reset the index to have nanobody_target as a column
data_frame.reset_index(inplace=True)
data_frame.rename(columns={'index': 'Nanobody_Target'}, inplace=True)
print(data_frame)

#-------------------------------------------------------------------------------------------------------------------------------
# Add HADDOCK scoring to the dataframe

tsv_file_path = "/content/drive/MyDrive/1.Thesis/1_AF3_structures/1_emscoring/emscoring.tsv"
haddock_df = pd.read_csv(tsv_file_path, sep="\t")


haddock_scores = []
for _, row in haddock_df.iterrows():
    pdb_file = row['original_name']
    score = row['score']

    # Extract nanobody_target and model index
    filename_parts = pdb_file.split("_")
    nanobody_target = f"{filename_parts[1]}_{filename_parts[2]}"  # e.g., "6rqm_p08962"
    match = re.search(r'(?:model(?:_from)?_)([0-9]+)', pdb_file)
    if match:
        model_index = int(match.group(1))
    else:
        model_index = None  # Handle cases where the model index can't be extracted
    haddock_scores.append((nanobody_target, model_index, score))

haddock_df = pd.DataFrame(haddock_scores, columns=["Nanobody_Target", "Model_Index", "HADDOCK_Score"])


for _, row in haddock_df.iterrows():  # Add HADDOCK score columns to the AlphaFold3 DataFrame
    nanobody_target = row["Nanobody_Target"]
    model_index = row["Model_Index"]
    haddock_score = row["HADDOCK_Score"]

    # Create a column for the model's HADDOCK score
    if model_index is not None: # Only create column name if the index is not None
        column_name = f"HADDOCK_score_model_{int(model_index)}"
        if column_name not in data_frame.columns:
            data_frame[column_name] = None 

        # Update the value for the corresponding nanobody_target
        data_frame.loc[data_frame["Nanobody_Target"] == nanobody_target, column_name] = haddock_score

# Calculate the mean HADDOCK score for each nanobody_target
haddock_mean = haddock_df.groupby("Nanobody_Target")["HADDOCK_Score"].mean().to_dict()
data_frame["HADDOCK_score_mean"] = data_frame["Nanobody_Target"].map(haddock_mean)


# Impute any remaining NaN values (to debug)
imputer = SimpleImputer(strategy='mean')
numeric_cols = df.select_dtypes(include=np.number).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

#---------------------------------------------------------------------------------------------------------------------------------
# Train Machine Learning Model

# Split data in train and test set and standardize
def load_and_preprocess_data(df, target_column, test_size=0.2, random_state=42):
    X = df.drop(columns=[target_column, "Nanobody_Target"]) #feature columns
    y = df[target_column] #target

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

# Function to train and evaluate a model

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
   
    model.fit(X_train, y_train) # Train the model
    y_pred = model.predict(X_test)    # Make predictions

    accuracy = accuracy_score(y_test, y_pred) # Evaluate model
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return accuracy, cm, report


# Function to run machine learning

def run_machine_learning(dataframe, target_column, model_choice='random_forest', test_size=0.2, random_state=42):
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataframe, target_column, test_size, random_state)

    # Initialize the model based on the choice
    if model_choice == 'random_forest':
        model = RandomForestClassifier(random_state=random_state)
    elif model_choice == 'svm':
        model = SVC(random_state=random_state)
    elif model_choice == 'logistic_regression':
        model = LogisticRegression(random_state=random_state)

    # Train and evaluate the model
    accuracy, cm, report = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)

    # Print evaluation results
    print(f"Model: {model_choice}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(report)


run_machine_learning(df, "True Pair?", model_choice='svm', random_state=1)
