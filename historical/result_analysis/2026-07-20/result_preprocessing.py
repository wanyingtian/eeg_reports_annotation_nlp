import sqlite3
import pandas as pd

def process_db(file_location):
    """Load data from a SQLite database table 'reports'."""
    conn = sqlite3.connect(file_location)
    query = "SELECT * FROM reports"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def load_excel_to_dfs(file_path, enforced=False, explanations=True):
    """Load data from an Excel file and return the two relevant DataFrames."""
    xls = pd.ExcelFile(file_path)
    if enforced:
        df_a1 = pd.read_excel(xls, sheet_name='A1_Enforced')
        df_a2 = pd.read_excel(xls, sheet_name='A2_Enforced')
    elif explanations:
        df_a1 = pd.read_excel(xls, sheet_name='explanations')
        df_a2 = pd.read_excel(xls, sheet_name='classifications')
    else:
        df_a1 = pd.read_excel(xls, sheet_name='A1_standardized_result')
        df_a2 = pd.read_excel(xls, sheet_name='A2_raw_result')
    return df_a1, df_a2

# Function to clean ground truth DataFrame by removing rows with NaN values and extracting relevant columns
def clean_ground_truth_by_index_range(df_ground_truth, index_ranges):
    columns_of_interest = ["Hashed ID", "Report", "Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]

    extracted_dfs = [df_ground_truth[columns_of_interest].iloc[start:end] for start, end in index_ranges]
    relevant_df_ground_truth = pd.concat(extracted_dfs, ignore_index=True)
    relevant_df_ground_truth = relevant_df_ground_truth.dropna().reset_index(drop=True)
    print(f"Length of LD df: {len(relevant_df_ground_truth)}")
    
    return relevant_df_ground_truth

# Function to align model DataFrame with the cleaned ground truth DataFrame
def align_model_with_ground_truth(cleaned_ground_truth_df, model_df):
    columns_of_interest = ["Hashed ID", "Report", "Abnormality", "Focal Epi", "Gen Epi", "Focal Non-epi", "Gen Non-epi"]
    
    # Extract the relevant columns from the model DataFrame
    relevant_model_df = model_df[columns_of_interest]
    
    # Ensure the model DataFrame matches the ground truth using "Hashed ID"
    hashed_ids = set(cleaned_ground_truth_df["Hashed ID"])
    
    # Filter out extra rows in the model DataFrame based on "Hashed ID"
    relevant_model_df = relevant_model_df[relevant_model_df["Hashed ID"].isin(hashed_ids)]
    
    # Re-index the model DataFrame to match the ground truth order and reset the index
    relevant_model_df = relevant_model_df.set_index("Hashed ID").reindex(cleaned_ground_truth_df.set_index("Hashed ID").index).reset_index()

    # Reset the index to start from 0
    relevant_model_df = relevant_model_df.reset_index(drop=True)
    
    print(f"Length of model df: {len(relevant_model_df)}")
    
    return relevant_model_df

# Convert all prediction values in the DataFrame to binary classes.
def get_core_predictions(df):
    df.iloc[:, 2:] = df.iloc[:, 2:].applymap(lambda x: 1 if x in [3, 4] else 0)
    return df


def process_all_files(LD_file, SG_file, baseline_files, excel_files, index_ranges, core=False):
    df_LD = process_db(LD_file)
    df_LD = clean_ground_truth_by_index_range(df_LD, index_ranges)
    if core:
        df_LD = get_core_predictions(df_LD)

    df_SG = process_db(SG_file)
    df_SG = clean_ground_truth_by_index_range(df_SG, index_ranges)
    df_SG = align_model_with_ground_truth(df_LD, df_SG)
    if core:
        df_SG = get_core_predictions(df_SG)

    models = {"LD": df_LD, "SG": df_SG}

    for model_name, file_path in excel_files.items():
        try:
            _, df_a2 = load_excel_to_dfs(file_path)
            df_a2 = align_model_with_ground_truth(df_LD, df_a2)
            if core:
                df_a2 = get_core_predictions(df_a2)
            models[model_name] = df_a2
            print(f"Loaded proposed model: {model_name}")
        except Exception as e:
            print(f"Failed to load Excel model {model_name}: {e}")
    

    for model_name, file_path in baseline_files.items():
        try:
            df = pd.read_csv(file_path)
            df = align_model_with_ground_truth(df_LD, df)
            if core:
                df = get_core_predictions(df)
            models[model_name] = df
            print(f"Loaded baseline: {model_name}")
        except Exception as e:
            print(f"Failed to load baseline {model_name}: {e}")


    return models