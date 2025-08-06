import os
import json
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import json
import re
import logging

def sanitize_json(json_str):
    """Ensure JSON strings have correct formatting before parsing."""
    try:
        # Remove leading/trailing spaces
        json_str = json_str.strip()

        #  Find the last occurrence of a valid JSON closing character (`}` or `]`)
        last_brace = max(json_str.rfind("}"), json_str.rfind("]"))

        if last_brace != -1:
            json_str = json_str[: last_brace + 1]  # Keep only valid JSON portion

        # Replace single quotes (used incorrectly in JSON) with double quotes
        json_str = json_str.replace("'", '\'')

        # Ensure inner double quotes inside string values are escaped
        json_str = json_str.replace('"', '\"')  # Fix duplicate quotes
        json_str = json_str.replace(',"}', '}')  # Remove trailing commas
        json_str = json_str.replace(',]', ']')  # Remove trailing commas from arrays
        # json_str = json_str.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        #  Detect and fix missing closing quotes inside `reasons` lists
        json_str = re.sub(r'(\["[^"\]]+)(?=\])', r'\1"', json_str)  # Add `"` if missing at the end of a list element

        #  Detect and fix missing closing quotes in key-value pairs
        json_str = re.sub(r'(:\s*")([^"]+)(?=[,\}])', r'\1\2"', json_str)

        #  Remove trailing commas before closing brackets
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        return json_str

    except Exception as e:
        logging.error("Error fixing JSON: %s", e)
        return "{}"

def clean_problematic_json(json_str, hashed_id):
    """Final failsafe JSON cleaning to extract valid information."""
    try:
        # Remove excessive commas inside lists
        json_str = re.sub(r',\s*,+', ',', json_str)  # Remove consecutive commas
        json_str = re.sub(r',\s*\]', ']', json_str)  # Remove trailing commas in lists
        json_str = re.sub(r',\s*\}', '}', json_str)  # Remove trailing commas before closing bracket
        json_str = json_str.replace("\n", "").replace("\r", "").replace("\t", "")

        # Convert string to JSON object and sanitize nested values
        json_data = json.loads(json_str)

        def clean_dict(data):
            """Recursively clean JSON and remove empty/malformed values."""
            if isinstance(data, dict):
                return {k: clean_dict(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [clean_dict(v) for v in data if isinstance(v, (str, int, float)) and v.strip() != ","]
            return data  # Keep valid values

        return clean_dict(json_data)

    except json.JSONDecodeError as e:
        print(hashed_id)
        logging.error("Still failed to decode JSON after cleanup: %s", e)
        logging.error("Problematic JSON: %s", repr(json_str))
        return {}


def process_original_json(json_str, hashed_id):
    """Process JSON for classifications, extracting only the decision values."""
    try:
        if not json_str or json_str.strip() == "":
            return {}

        json_data = json.loads(json_str.replace("'", '"'))  # Fix JSON format

        standardized_keys = {
            "focal_epileptiform_activity": "Focal Epi",
            "generalized_epileptiform_activity": "Gen Epi",
            "focal_non_epileptiform_activity": "Focal Non-epi",
            "generalized_non_epileptiform_activity": "Gen Non-epi",
            "abnormality": "Abnormality"
        }

        return {standardized_keys.get(key, key): json_data.get(key, None) for key in standardized_keys}

    except json.JSONDecodeError as e:
        print(hashed_id)  # Print the Hashed ID for debugging
        logging.error("Failed to decode JSON for Hashed ID %s: %s", hashed_id, e)
        logging.error("Problematic JSON: %s", repr(json_str))
        return {}



def process_ngram_json(json_str, hashed_id):
    """Extract both 'decision' values and 'reasons' from JSON format."""
    try:
        if not json_str or json_str.strip() == "":
            return {}

        json_str_sanitized = sanitize_json(json_str)  # First-pass sanitization
        try:
            json_data = json.loads(json_str_sanitized)  # Try parsing
        except json.JSONDecodeError:
            # print(f"First sanitization did not work on {hashed_id}\n")
            print(f"original text: {json_str}\n")
            print(f"sanitized json str: {json_str_sanitized}\n")
            json_data = clean_problematic_json(json_str_sanitized, hashed_id)  # Failsafe cleaning

        standardized_keys = {
            "focal_epileptiform_activity": "Focal Epi",
            "generalized_epileptiform_activity": "Gen Epi",
            "focal_non_epileptiform_activity": "Focal Non-epi",
            "generalized_non_epileptiform_activity": "Gen Non-epi",
            "abnormality": "Abnormality"
        }

        processed_data = {}
        for key, label in standardized_keys.items():
            if key in json_data:
                decision = json_data[key].get("decision")
                reasons = json_data[key].get("reasons", [])

                # Ensure reasons list is valid
                if not isinstance(reasons, list):
                    reasons = []

                processed_data[label] = decision
                processed_data[f"{label} Reasons"] = "; ".join(reasons) if reasons else "No Explanation Given"

        return processed_data

    except json.JSONDecodeError as e:
        print(hashed_id)  # Print the Hashed ID for debugging
        logging.error("Failed to decode JSON for Hashed ID %s: %s", hashed_id, e)
        logging.error("Problematic JSON: %s", repr(json_data))
        return {}


def main(input_filename, output_filename=None, folder_path="../../outputs/pipeline_output", num_reports=None):
    input_filepath = os.path.join(folder_path, input_filename)

    try:
        df = pd.read_csv(input_filepath, nrows=num_reports)
        logging.info("Processing %d reports from %s", len(df), input_filepath)
    except FileNotFoundError:
        logging.error("File not found: %s", input_filepath)
        return

    # Process classification results (decisions only)
    if "classifications" in df and df["classifications"].notna().any():
        df["classifications_dict"] = df.apply(
            lambda row: process_original_json(row["classifications"], row["Hashed ID"]), axis=1
        )
    else:
        logging.warning("Column 'classifications' is missing or empty!")

    # Process explanations (decisions + reasons)
    if "explanations" in df and df["explanations"].notna().any():
        df["explanations_dict"] = df.apply(
            lambda row: process_ngram_json(row["explanations"], row["Hashed ID"]), axis=1
        )
    else:
        logging.warning("Column 'explanations' is missing or empty!")

    # Convert processed data to DataFrames
    json_df1 = pd.DataFrame(df.get("classifications_dict", {}).tolist())  # Classifications
    json_df2 = pd.DataFrame(df.get("explanations_dict", {}).tolist())  # Explanations (Decisions + Reasons)

    # Ensure expected columns exist
    expected_columns = ["Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi", "Abnormality"]
    for col in expected_columns:
        json_df1[col] = json_df1.get(col, pd.NA)
        json_df2[col] = json_df2.get(col, pd.NA)
        json_df2[f"{col} Reasons"] = json_df2.get(f"{col} Reasons", pd.NA)

    # Preserve Hashed ID and Report text
    json_df1[["Hashed ID", "Report"]] = df[["Hashed ID", "Report"]]
    json_df2[["Hashed ID", "Report"]] = df[["Hashed ID", "Report"]]

    logging.info("Preview of processed DataFrame (Classifications):")
    logging.info("\n%s", json_df1.head())

    logging.info("Preview of processed DataFrame (Explanations):")
    logging.info("\n%s", json_df2.head())

    # Create output directory if it doesn't exist
    output_folder = "../../outputs/processed_output"
    os.makedirs(output_folder, exist_ok=True)
    logging.info("Ensured output directory exists: %s", output_folder)

    # Determine output file name
    if output_filename is None:
        base_filename = os.path.splitext(input_filename)[0]
        output_filename = f"processed_{base_filename}.xlsx"

    output_filepath = os.path.join(output_folder, output_filename)

    # Save to Excel
    try:
        with pd.ExcelWriter(output_filepath, engine="xlsxwriter") as writer:
            json_df1.to_excel(writer, sheet_name="classifications", index=False)
            json_df2.to_excel(writer, sheet_name="explanations", index=False)
        logging.info("Data saved successfully to %s", output_filepath)
    except Exception as e:
        logging.error("Failed to save data: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process EEG classification and explanation results.")
    parser.add_argument("input_filename", help="Input CSV file name (e.g., 'mistral_zoe_first_10_results_v1.csv')")
    parser.add_argument("--output_filename", help="Output Excel file name (default: 'processed_<input_filename>.xlsx')")
    parser.add_argument("--folder_path", default="../../outputs/pipeline_output", help="Path to the folder containing the input CSV")
    parser.add_argument("--num_reports", type=int, help="Number of reports to process (default: all)")

    args = parser.parse_args()
    main(args.input_filename, args.output_filename, args.folder_path, args.num_reports)
