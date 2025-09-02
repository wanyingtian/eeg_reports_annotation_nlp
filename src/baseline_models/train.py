# Copyright (c) 2025 Wanying Tian
# Licensed under the Apache-2.0 License (see LICENSE file in the project root for details).
import pandas as pd
import sqlite3
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import os
import argparse
import joblib
from pathlib import Path
# -------------Constants-------------------
# Resolve paths relative to the repo root (two levels up from this script)
BASE_DIR = Path(__file__).resolve().parent      # e.g., src/baseline_models
REPO_ROOT = BASE_DIR.parents[1]                 # repo root

REPORT_DB_PATH = REPO_ROOT / "data/zoe_reports_sample.db" # This will not work unless replaced with path to ANNOTATED DB
# The db should have the following columns: 'Hashed_ID', 'Report', 'Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi', 'Abnormality'
# if the db format is different, please modify the fetch_reports function accordingly.
MODEL_DIR = REPO_ROOT / 'outputs/baseline_results/trained_models'
OUTPUT_DIR = REPO_ROOT / 'outputs/baseline_results/training_results'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result_columns = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]

# Configurable Indices
START_INDEX = 0
END_INDEX = 100

# Predefined Models
MODEL_MAPPING = {

    "bert_base": "bert-base-uncased",
    "bag_of_words": "BagOfWords"  # Placeholder for Bag of Words
}
# -------------Functions-------------------

# Fetch Reports
def fetch_reports(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT `Hashed ID`, `Report`, `Focal Epi`, `Gen Epi`, `Focal Non-epi`, `Gen Non-epi`, `Abnormality` FROM reports"
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Prepare Labels
def transform_labels(y):
    return y.map({1: 0, 2: 0, 3: 1, 4: 1})

# Load Subset of Reports with Missing Value Handling
def load_reports(df, start, end):
    if start < 0 or end > len(df) or start >= end:
        raise ValueError("Invalid start or end indices.")
    subset_df = df.iloc[start:end]
    if subset_df[result_columns].isnull().any().any():
        print("Removing rows with NaN values in result columns.")
        subset_df = subset_df.dropna(subset=result_columns)
    return subset_df

# Load Model for Embedding
def load_embedding_model(model_name):
    if model_name in ["clinical-bert", "bert_base"]:
        model = AutoModel.from_pretrained(MODEL_MAPPING[model_name]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[model_name])
        return model, tokenizer
    elif model_name == "bag_of_words":
        print("Initializing Bag of Words embedding...")
        vectorizer = CountVectorizer(
            max_features=10000,  # Limit the number of features
            stop_words="english",  # Exclude stop words
            ngram_range=(1, 5),  # Include unigrams, bigrams, and trigrams
            token_pattern=r'\b[a-zA-Z]{3,}\b'  # Exclude short words & numbers
        )
        return vectorizer
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Extract Embeddings
def extract_embeddings(model_name, reports, embedding_model):
    if model_name in ["clinical-bert", "bert_base"]:
        model, tokenizer = embedding_model
        embeddings = []
        token_lengths = [len(tokenizer(report, truncation=False, padding=False)["input_ids"]) for report in reports]

        if not token_lengths:
            return embeddings, 0, 0, 0

        average_length = sum(token_lengths) / len(token_lengths)
        max_length = max(token_lengths)
        min_length = min(token_lengths)

        print(f"Average report length: {average_length:.2f} tokens")
        print(f"Maximum report length: {max_length} tokens")
        print(f"Minimum report length: {min_length} tokens")

        with torch.no_grad():
            for report in reports:
                inputs = tokenizer(report, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                outputs = model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embedding.flatten())

        return embeddings, average_length, max_length, min_length

    elif model_name == "bag_of_words":
        vectorizer = embedding_model
        print("Fitting and transforming Bag of Words...")
        bow_matrix = vectorizer.fit_transform(reports)  # Sparse matrix
        joblib.dump(vectorizer, os.path.join(MODEL_DIR, "bag_of_words_vectorizer.pkl"))
        with open(os.path.join(MODEL_DIR, "feature_names.txt"), 'w') as f:
            f.write("\n".join(vectorizer.get_feature_names_out()))

        return bow_matrix.toarray(), 0, 0, 0  # Return dense matrix, no length metrics for BoW

# Save Training Configuration and Results
def save_training_config(num_reports, hyperparameters, training_results, model_name):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    config_filename = os.path.join(OUTPUT_DIR, f"training_config_{model_name}.txt")
    with open(config_filename, 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Number of reports: {num_reports}\n")
        f.write("Model Hyperparameters:\n")
        for key, value in hyperparameters.items():
            f.write(f" - {key}: {value}\n")
        f.write("\nTraining Results:\n")
        for column, results in training_results.items():
            f.write(f"{column}:\n")
            f.write(f" - Average Report Tokens: {results['average_report_tokens']}\n")
            f.write(f" - Max Report Tokens: {results['max token']}\n")
            f.write(f" - Min Report Tokens: {results['min token']}\n")
            f.write(f" - Cross-Validation Scores: {results['cross_val_scores']}\n")
            f.write(f" - Mean Cross-Validation Score: {results['mean_cross_val_score']:.4f}\n")
            f.write(f" - Precision: {results['precision']:.4f}\n")
            f.write(f" - Recall: {results['recall']:.4f}\n")
            f.write(f" - F1-Score: {results['f1_score']:.4f}\n")

    print(f"Training config and results saved to {config_filename}")

# Train and Save Models
def train_and_save_models(df, model_name):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    print(f"Loading embedding model: {model_name}")
    embedding_model = load_embedding_model(model_name)

    print("Extracting embeddings...")
    embeddings, average, max_tokens, min_tokens = extract_embeddings(model_name, df['Report'].tolist(), embedding_model)

    training_results = {}
    hyperparameters = {"max_iter": 1000}

    for column in result_columns:
        print(f"\nTraining model for: {column}")
        y = transform_labels(df[column])

        # Initialize Logistic Regression model
        model = LogisticRegression(max_iter=hyperparameters["max_iter"], verbose=1)
        skf = StratifiedKFold(n_splits=5)
        precision_list, recall_list, f1_list, accuracy_list = [], [], [], []

        for train_index, test_index in skf.split(embeddings, y):
            X_train, X_test = [embeddings[i] for i in train_index], [embeddings[i] for i in test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            accuracy_list.append(accuracy_score(y_test, y_pred))
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        mean_cv_score = sum(accuracy_list) / len(accuracy_list)
        print(f"Cross-validation accuracy for {column}: {accuracy_list}")
        print(f"Mean CV accuracy: {mean_cv_score:.4f}")

        # Save the trained model
        model_path = os.path.join(MODEL_DIR, f"{column}_{model_name}_model.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved: {model_path}")

        # Save model coefficients for BoW
        if model_name == "bag_of_words":
            coeff_path = os.path.join(MODEL_DIR, f"{column}_coefficients.pkl")
            joblib.dump(model.coef_, coeff_path)
            print(f"Coefficients saved to {coeff_path}")

        training_results[column] = {
            "average_report_tokens": average,
            "max token": max_tokens,
            "min token": min_tokens,
            "cross_val_scores": accuracy_list,
            "mean_cross_val_score": mean_cv_score,
            "precision": sum(precision_list) / len(precision_list),
            "recall": sum(recall_list) / len(recall_list),
            "f1_score": sum(f1_list) / len(f1_list),
        }



    save_training_config(len(df), hyperparameters, training_results, model_name)
    print(f"All models trained and saved for {model_name}.")

# Main Function
def main():

    parser = argparse.ArgumentParser(description="Run inference on EEG reports.")
    parser.add_argument("--model", required=True, choices=MODEL_MAPPING.keys(),
                        help="Choose the model for inference: bert_base, bag_of_words")
    # NEW: dataset controls (replace --author)
    parser.add_argument("--dataset-path", type=Path, default=REPORT_DB_PATH,
                        help='''
                        Path to the ANNOTATED dataset SQLite file. \n
                        The db should have the following columns: 'Hashed_ID', 'Report', 'Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi', 'Abnormality';" \n
                        "if the db format is different, please modify the fetch_reports function accordingly.''')
    parser.add_argument("--start", type=int, default=START_INDEX, help="Start index for reports (default: 0)")
    parser.add_argument("--end", type=int, default=END_INDEX, help="End index for reports (default: 100)")

    args = parser.parse_args()
    model_name = args.model
    dataset_path: Path = args.dataset_path
    start = max(args.start, 0)
    end = args.end
 

    print("Fetching reports from database...")
    df = fetch_reports(dataset_path)
    if df.empty:
        print("No reports loaded from the database.")
        return

    print("Loading subset of reports...")
    training_df = load_reports(df, start, end)  # Load reports for training
    print(f"Training models using {model_name}...")
    train_and_save_models(training_df, model_name)

if __name__ == "__main__":
    main()
