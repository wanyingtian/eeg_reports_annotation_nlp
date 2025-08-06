import pandas as pd
import sqlite3
import torch
from transformers import AutoModel, AutoTokenizer
import os
import joblib
# import gensim
# import gensim.downloader as api
import argparse
from tqdm import tqdm  # Progress bar

# Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = '../../outputs/baseline_results/trained_models'
OUTPUT_DIR = '../../outputs/baseline_results/inference_results'
result_columns = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]

# Predefined Models
MODEL_MAPPING = {

    "bert_base": "bert-base-uncased",
    "bag_of_words": "BagOfWords" 
}

# Function to load reports from the SQLite database
def fetch_reports(db_path):
    try:
        conn = sqlite3.connect(db_path)
        query = "SELECT `Hashed ID`, `Report` FROM reports"
        df = pd.read_sql_query(query, conn)
        return df
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

# Function to load the embedding model
def load_embedding_model(model_name):
    if model_name == "bert_base":
        model = AutoModel.from_pretrained(MODEL_MAPPING[model_name]).to(device)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_MAPPING[model_name])
        return model, tokenizer
    elif model_name == "bag_of_words":
        print("Loading Bag of Words vectorizer...")
        vectorizer_path = os.path.join(MODEL_DIR, "bag_of_words_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}. Ensure it was saved during training.")
        return joblib.load(vectorizer_path)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

# Function to extract embeddings based on the selected model
def extract_embeddings(model_name, reports, embedding_model):
    embeddings = []
    if model_name == "bert_base":
        model, tokenizer = embedding_model
        batch_size = 16  # Set batch size for processing
        with torch.no_grad():
            for i in range(0, len(reports), batch_size):
                batch_reports = reports[i:i + batch_size]
                inputs = tokenizer(batch_reports, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
                outputs = model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(cls_embeddings)
        return embeddings
    elif model_name == "bag_of_words":  
        print("Transforming Bag of Words embeddings...")
        return embedding_model.transform(reports).toarray()
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    

# Predict with Threshold Logic
def predict_with_confidence(model, embedding, epsilon):
    probability = model.predict_proba([embedding])[0, 1]  # Get probability for class 1
    if probability < 0.5 - epsilon:
        return 1  # Confident No
    elif probability < 0.5:
        return 2  # Low confidence No
    elif probability < 0.5 + epsilon:
        return 3  # Low confidence Yes
    else:
        return 4  # Confident Yes

def run_inference(df, model_name, epsilon, author):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading embedding model: {model_name}")
    embedding_model = load_embedding_model(model_name)

    # Load feature names for BoW
    feature_names = None
    if model_name == "bag_of_words":
        feature_names_path = os.path.join(MODEL_DIR, "feature_names.txt")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found at {feature_names_path}.")
        with open(feature_names_path, 'r') as f:
            feature_names = f.read().splitlines()

    print("Extracting embeddings for inference...")
    embeddings = extract_embeddings(model_name, df['Report'].tolist(), embedding_model)

    results = []
    missing_models = []

    for index, (hashed_id, report) in tqdm(enumerate(zip(df['Hashed ID'], df['Report'])), total=len(df), desc="Processing reports"):
        classifications = {}
        probabilities = {}
        top_words = {}  # Store top contributing words for each column

        print(f"\nProcessing report ID: {hashed_id}")
        # print(f"Report text: {report}")

        for column in result_columns:
            model_path = os.path.join(MODEL_DIR, f"{column}_{model_name}_model.pkl")
            if not os.path.exists(model_path):
                if column not in missing_models:
                    missing_models.append(column)
                continue

            # Load the trained model
            model = joblib.load(model_path)
            prediction = predict_with_confidence(model, embeddings[index], epsilon)
            classifications[column] = prediction
            probabilities[f"Prob_{column}"] = model.predict_proba([embeddings[index]])[0, 1]

            # Compute top contributing words for BoW
            if model_name == "bag_of_words" and feature_names is not None:
                coefficients = model.coef_.flatten()
                word_contributions = [(feature_names[i], embeddings[index][i] * coefficients[i]) for i in range(len(coefficients))]
                word_contributions = sorted(word_contributions, key=lambda x: abs(x[1]), reverse=True)[:30]

                # Print the top words and their contributions
                # print(f"Top contributing words for {column}:")
                # for word, weight in word_contributions:
                #     print(f"  - {word}: {weight:.4f}")

                # Save to results
                top_words[f"TopWords_{column}"] = ", ".join([f"{word} ({weight:.4f})" for word, weight in word_contributions])

        # Create a temporary DataFrame for the current report
        temp_df = pd.DataFrame([{
            'Hashed ID': hashed_id,
            'Report': report,
            **classifications,
            **probabilities,
            **top_words  # Add top contributing words to the output
        }])
        results.append(temp_df)

    # Concatenate all results into a single DataFrame
    results_df = pd.concat(results, ignore_index=True)

    # Generate versioned output filenames
    version_index = 1
    output_filename = os.path.join(OUTPUT_DIR, f"{author}_inference_results_{model_name}_ep={epsilon}_v{version_index}.csv")
    while os.path.exists(output_filename):
        version_index += 1
        output_filename = os.path.join(OUTPUT_DIR, f"{author}_inference_results_{model_name}_ep={epsilon}_v{version_index}.csv")

    # Save results to CSV
    results_df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")

    if missing_models:
        print(f"Skipped inference for the following columns due to missing models: {', '.join(missing_models)}")

# Main function
def main():
    parser = argparse.ArgumentParser(description="Run inference on EEG reports.")
    parser.add_argument("--model", required=True, choices=MODEL_MAPPING.keys(),
                        help="Choose the model for inference: bert_base, bag_of_words")
    parser.add_argument('--author', type=str, choices=['zoe', 'maria'], default="zoe",
                        help='Report author: choose either "zoe" or "maria"')
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Set the epsilon value for confidence thresholds (default: 0.1)")
    parser.add_argument("--start", type=int, default=0, help="Start index for reports (default: 100)")
    parser.add_argument("--end", type=int, default=10, help="End index for reports (default: 2000)")

    args = parser.parse_args()
    model_name = args.model
    author = args.author
    epsilon = args.epsilon
    start = args.start
    end = args.end

    # Load all reports from the database
    print("Fetching all reports from the database...")
    if author == "zoe":
        REPORT_DB_PATH = '../../data/zoe_reports_10.db' # only 10 reports as sample for Zoe
    elif author == "maria":
        REPORT_DB_PATH = '../../data/zoe_reports_10.db' # placeholder for Maria's database
    else:
        raise ValueError("Invalid author. Choose either 'zoe' or 'maria'.")
    df = fetch_reports(REPORT_DB_PATH)
    if df.empty:
        print("No reports loaded from the database.")
        return

    # Load reports for inference based on start and end indices
    print("Loading reports for inference...")
    inference_df = df.iloc[start:end]

    print("Running inference...")
    run_inference(inference_df, model_name, epsilon,author)

if __name__ == "__main__":
    main()
