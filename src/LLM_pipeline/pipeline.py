import pandas as pd
import sqlite3
import httpx
import json
import os
import argparse
from llama_cpp.llama import Llama, LlamaGrammar
from huggingface_hub import hf_hub_download
import time
import multiprocessing
import re
# Config
OUTPUT_FOLDER_PATH = '../../outputs/pipeline_output'
TEMPERATURE = 0  # default 0.8
MAX_TOKENS = 3000
STOP_SEQUENCES = None
COMMENT = """
testing the pipeline with 10 reports in organized repo
"""
REPORT_DB_PATH = '../../data/zoe_reports_10.db'

REPORT_DB_PATH_MARIA = '../../data/zoe_reports_10.db' # this is a placeholder, change it to the actual path for Maria's reports


# Function to fetch reports from SQLite database
def fetch_reports(db_path):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT `Hashed ID`, `Report` FROM reports"
        cursor.execute(query)
        
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        conn.close()

# Function to generate results as JSON format
def convert_to_json(model, prompt1, report, optional_context=None, grammar=None, 
                    temperature=TEMPERATURE, stop=STOP_SEQUENCES):
    full_prompt = ""
    full_prompt += prompt1 + report
    if optional_context:
        full_prompt += optional_context + "\n\n"
    
    try:
        response = model(
            full_prompt,
            grammar=grammar,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            stop=stop
        )
        result = response['choices'][0]['text']
        return result
    except json.JSONDecodeError as e:
        print("Failed to decode JSON from the following response:")
        print(response['choices'][0]['text'])
        raise e  # Re-raise the exception after logging

# Function to load grammar file    
def load_gbnf_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()

        if not content.strip():
            raise ValueError("The .gbnf file is empty.")
        
        grammar = LlamaGrammar.from_string(content)
        return grammar

    except FileNotFoundError:
        print(f"Error: The file at {file_path} does not exist.")
        return None
    except ValueError as e:
        print(f"ValueError: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def download_model(model_name="mistral"):
    # download the selected models
    model_configs = {
        "mistral": {
            "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
            "filename": "mistral-7b-instruct-v0.2.Q5_K_M.gguf"
        },
        "deepseek": {
            "repo_id": "TheBloke/deepseek-llm-7b-base-GGUF",
            "filename": "deepseek-llm-7b-base.Q5_K_M.gguf"
        },
        "deepseek-coder": {
            "repo_id": "TheBloke/deepseek-coder-6.7B-instruct-GGUF",
            "filename": "deepseek-coder-6.7b-instruct.Q5_K_M.gguf"
        },
        "deepseek-chat": {
            "repo_id": "TheBloke/deepseek-llm-7B-chat-GGUF",
            "filename": "deepseek-llm-7b-chat.Q5_K_M.gguf"
        },
        "hermes-mistral": {
            "repo_id": "NousResearch/Nous-Hermes-2-Mistral-7B-DPO-GGUF",
            "filename": "Nous-Hermes-2-Mistral-7B-DPO.Q5_K_M.gguf"
        },
        "hermes-llama2": {
            "repo_id": "TheBloke/Nous-Hermes-Llama-2-7B-GGUF",
            "filename": "nous-hermes-llama-2-7b.Q5_K_M.gguf"
        },

    }

    if model_name not in model_configs:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from {list(model_configs.keys())}.")

    config = model_configs[model_name]

    try:
        model_path = hf_hub_download(repo_id=config["repo_id"], filename=config["filename"])
        llm_model = Llama(model_path=model_path, n_ctx=4096, n_gpu_layers=30)
        print(f"{model_name.capitalize()} model loaded successfully")
        return llm_model
    except Exception as e:
        print(f"Failed to load the {model_name} model: {e}")
        raise

def determine_filenames(num_reports, author = 'zoe',model = 'mistral'):
    """ Determine the output filenames based on the number of reports and existing files. """
    # Define the folder path
    folder_path = OUTPUT_FOLDER_PATH 

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Determine the version index based on existing files in the folder
    version_index = 1
    while os.path.exists(os.path.join(folder_path, f'{model}_{author}_first_{num_reports}_results_v{version_index}.csv')):
        version_index += 1

    output_filename = os.path.join(folder_path, f'{model}_{author}_first_{num_reports}_results_v{version_index}.csv')
    config_filename = os.path.join(folder_path, f'{model}_{author}_config_first_{num_reports}_v{version_index}.txt')

    return output_filename, config_filename



# Function to load and process the completed CSV
def process_completed_csv(completed_csv_path):
    """ Process and load completed reports from CSV """
    results_df = pd.DataFrame(columns=[
        'Hashed ID', 'Report', 'Label', 'classifications', 
        'explanations'
    ])
    
    if completed_csv_path and os.path.exists(completed_csv_path):
        print("Completed file exists")
        try:
            completed_df = pd.read_csv(completed_csv_path)
            results_df = pd.concat([results_df, completed_df], ignore_index=True)
            completed_hashes = set(completed_df['Hashed ID'])
            print(f"Completed reports CSV file loaded successfully: {completed_csv_path}")
            return results_df, completed_hashes
        except Exception as e:
            print(f"Error reading completed reports CSV file: {e}")
            return results_df, set()
    else:
        print("File Don't Exist, proceed with empty df")
        return results_df, set()
    
def load_reports(num_reports, completed_hashes, author = 'zoe'):
    if author == 'zoe':
        file_name = REPORT_DB_PATH
    elif author == 'maria':
        file_name = REPORT_DB_PATH_MARIA
    # Load reports to df
    data = [row for _, row in zip(range(num_reports), fetch_reports(file_name))]
    df = pd.DataFrame(data, columns=['Hashed ID', 'Report'])

    # Filter out already processed reports
    df = df[~df['Hashed ID'].isin(completed_hashes)]
    return df

def run_pipeline(model, df, results_df, result_grammar, result_grammar_exp, output_filename, config_filename, start_time, n=1):
    results_df = results_df
    print(f"completed report in pipeline: {len(results_df)}\n")
    #output_files(results_df, output_filename, config_filename, start_time)
    # Process each report and save the results incrementally
    for index, row in df.iterrows():
        hashed_id = row['Hashed ID']
        report = row['Report']
        #label = row.get('Label', 'N/A')  # Handle missing labels by setting a default value
        
        classifications = convert_to_json(model, prompt1, report, grammar=result_grammar)
        explanations = convert_to_json(model, prompt2, report, optional_context = classifications, grammar=result_grammar_exp)
    
        
        #a1_a2_agree = normalize_string(a1_qa_results) == normalize_string(a2_qa_results_report_direct)
        
        # Create a temporary DataFrame for the current result
        temp_df = pd.DataFrame([{
            'Hashed ID': hashed_id,
            'Report': report,
            #'Label': label,
            'classifications': classifications,
            'explanations': explanations
            #'A1_A2_agree': a1_a2_agree
        }])
        
        # Append the temporary DataFrame to results_df
        results_df = pd.concat([results_df, temp_df], ignore_index=True)
        
        # save the results incrementally to the CSV file
        if (index + 1) % n == 0:  # Adjust the frequency as needed
            print(f"Reports completed: {len(results_df)}")
            with open(config_filename, 'w') as f:
                f.write(f'Temperature: {TEMPERATURE}\n')
                f.write(f"Reports completed: {len(results_df)}")
            results_df.to_csv(output_filename, index=False)
    
    return results_df

def output_files(results_df, output_filename, config_filename, start_time, author = 'zoe', model = 'mistral'):

    # Calculate the percentage of absolute agreement
    total_rows = len(results_df)
    
    # Save the results to the output file
    results_df.to_csv(output_filename, index=False)
    
    # End the timer and calculate elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    
    # Save the settings used for that run in a separate file
    with open(config_filename, 'w') as f:
        f.write(f'Author: {author}\n')
        f.write(f'Temperature: {TEMPERATURE}\n')
        f.write(f'Stop Sequences: {STOP_SEQUENCES}\n')
        f.write(f'Model: {model}\n')
        f.write(f'Comment: {COMMENT}\n')
        f.write(f'Reports completed: {total_rows}\n')
        f.write(f'Elapsed Time: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)\n')
        f.write(f'Prompt1: {prompt1}\n')
        f.write(f'Prompt2: {prompt2}\n')


    print(f"Saved results to {output_filename} and settings to {config_filename}")
    print(f"Time taken to run the script: {elapsed_time:.2f} seconds ({elapsed_minutes:.2f} minutes)")
# Prompts
prompt1 = """
Read the following EEG Report data carefully, then answer the following questions about the report. Use the provided definitions and examples as a guide for interpretation.
Remember to follow constraints.

Definitions and Examples:
1. Focal Epileptiform Activity:
Definition: Epileptiform discharges limited to a specific area, suggesting focal seizure activity.
Example: "Sharp waves in the right temporal region" or "focal spike activity seen in the left frontal lobe."

2. Generalized Epileptiform Activity:
Definition: Epileptiform discharges occurring simultaneously across both hemispheres, indicating generalized epilepsy.
Example: "Generalized spike-and-wave discharges at 3 Hz" or "bilateral synchronous polyspike bursts."

3. Focal Non-Epileptiform Activity:
Definition: Non-epileptic activity confined to a specific area, possibly indicating localized brain dysfunction.
Example: "Regional slowing in the left posterior quadrant" or "focal attenuation in the right frontal area."

4. Generalized Non-Epileptiform Activity:
Definition: Non-epileptic abnormalities broadly distributed over both hemispheres, suggesting systemic dysfunction.
Example: "Diffuse background slowing" or "generalized low-amplitude theta activity."

5. Abnormality:
Definition: Any deviation from normal EEG patterns, which could be epileptiform or non-epileptiform.
Example: "Abnormal interictal spikes in the temporal lobe," "persistent delta slowing in the frontal regions," or "asymmetric voltage attenuation."

Questions:
Is there Focal Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is there Generalized Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is there Focal Non-Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is there Generalized Non-Epileptiform Activity present?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)
Is the EEG abnormal?
(options: 1 = Confident no, 2 = Low confidence no, 3 = Low confidence yes, 4 = Confident yes)

Constraints:
Err on the side of confident decisions. Use 1 or 4 whenever possible. Only use 2 or 3 if there is strong, unavoidable ambiguity.
Choose 2 or 3 sparingly, only when absolutely necessary.
If all of "Focal Epileptiform Activity," "Generalized Epileptiform Activity," "Focal Non-Epileptiform Activity," and "Generalized Non-Epileptiform Activity" are marked as normal (1 or 2), then "Abnormality" must also be marked as normal (1 or 2).
If any of "Focal Epileptiform Activity," "Generalized Epileptiform Activity," "Focal Non-Epileptiform Activity," or "Generalized Non-Epileptiform Activity" is marked as abnormal (3 or 4), then "Abnormality" must also be marked as abnormal (3 or 4).

Please provide the answers ONLY in the following JSON format:

{
  "focal_epileptiform_activity": "integer",
  "generalized_epileptiform_activity": "integer",
  "focal_non_epileptiform_activity": "integer",
  "generalized_non_epileptiform_activity": "integer",
  "abnormality": "integer"
}
Do not include any additional explanations or comments in the output.

"""

prompt2 = """ 
Read the following EEG Report and the corresponding classification output carefully. Your task is to generate a machine-readable JSON output that provides explanations for each classification by identifying and extracting verbatim phrases from the EEG report that contributed to each decision.

***Guidelines***
1. The output must strictly follow the JSON format provided below with no extra text.
2. Each category (Focal Epileptiform Activity, Generalized Epileptiform Activity, Focal Non-Epileptiform Activity, Generalized Non-Epileptiform Activity, and Abnormality) should include:
    - "decision": An integer taken from the classification output.
    - "reasons": A list of verbatim phrases from the EEG report that support the classification.
3.  Handle all quotation marks properly:
4.  Escape double quotes inside text (" → \") to prevent JSON parsing errors.
5. Preserve single quotes (') inside text as-is unless they cause formatting issues.
6. If the classification is:
    - 1 (Confident No) or 2 (Low Confidence No) → Extract phrases that indicate an absence of relevant findings.
    - 3 (Low Confidence Yes) or 4 (Confident Yes) → Extract phrases that explicitly support the presence of relevant findings.
7. DO NOT paraphrase or summarize. Only extract exact text from the EEG report.
8. If no relevant phrase is found in the report, return "No specific mention in the report."
9. Ensure the output is valid JSON with no extra text.

**Output Format**
json
{
  "focal_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "generalized_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "focal_non_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "generalized_non_epileptiform_activity": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  },
  "abnormality": {
    "decision": <integer>,
    "reasons": ["<escaped verbatim text>", "<escaped verbatim text>", ...]
  }
}

Do not include any additional explanations, comments, or extraneous text outside of the required JSON format.

---

**Input Format:**
- This prompt will be followed by:
  1. The original EEG report.
  2. The classification output from the previous LLM response.

Process the input accordingly and generate the required structured JSON output.

"""
# Helper function to find the latest versioned CSV file in the folder
def get_latest_version_csv(num_reports, folder_path='pipeline_results',file_type = 'csv'):
    """ Get the latest version of the results CSV file for a given number of reports. """

    version_files = []
    if file_type == 'csv':
        for file in os.listdir(folder_path):
            match = re.match(f'first_{num_reports}_results_v(\\d+).csv', file)
            if match:
                version_files.append((int(match.group(1)), file))
    elif file_type == 'config':
        for file in os.listdir(folder_path):
            match = re.match(f'config_first_{num_reports}_v(\\d+).csv', file)
            if match:
                version_files.append((int(match.group(1)), file))

    if version_files:
        # Sort by version number and return the latest one
        version_files.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(folder_path, version_files[0][1])
    else:
        return None

def manager(num_reports, initial_csv_path, author, model):
    
    current_csv_path = initial_csv_path  # Start with the CSV file passed as an argument (if any)
    
    completed_df, completed_hashes = process_completed_csv(initial_csv_path)
    current_doc_index = len(completed_df) if current_csv_path else 0
    print(current_doc_index)
    print(current_csv_path)

    def worker_function(completed_csv_path):
        """ Worker runs the pipeline """
        main(num_reports, completed_csv_path, author, model)

    while current_doc_index < num_reports:


        # Start worker process with the current CSV path
        worker_process = multiprocessing.Process(target=worker_function, args=(current_csv_path,))
        worker_process.start()
        crash_filename = "crash_report.txt"
        # Monitor the worker process
        while worker_process.is_alive():
            worker_process.join(timeout=1)

        # If worker finished or crashed
        if worker_process.exitcode == 0:
            print("Pipeline completed successfully.")
            break
        else:
            # Log crash info to the config file
            crash_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            crash_message = f"Worker crashed at {crash_time}. Restarting...\n"
            
            with open(crash_filename, 'a') as f:  # Append crash info to the config file
                f.write(crash_message)
                f.write(f"Current document index: {current_doc_index}\n")
                f.write(f"Current CSV file: {current_csv_path}\n")
                f.write(f"Worker exit code: {worker_process.exitcode}\n")

            print("Worker crashed. Restarting...")

            # On crash, find the latest version of the CSV file if it exists
            latest_csv_path = get_latest_version_csv(num_reports) 
            if latest_csv_path:
                print(f"Switching to latest CSV file: {latest_csv_path}")
                current_csv_path = latest_csv_path  # Use the latest versioned file after the crash
            else:
                current_csv_path = None  # Continue with no completed CSV file

            # Reload the completed reports from the latest file (if any)
            completed_df, completed_hashes = process_completed_csv(current_csv_path)
            current_doc_index = len(completed_hashes)
            print(f"Restarting from document {current_doc_index}...")




def main(num_reports, completed_csv_path, author,model_name):
    # Start the timer
    start_time = time.time()
    # Prepare the grammars
    result_grammar = load_gbnf_file('result_grammar.gbnf')
    result_grammar_exp = load_gbnf_file('result_grammar_exp.gbnf')
    # Download the mistral model
    model = download_model(model_name)
    # output filenames based on version
    output_filename, config_filename = determine_filenames(num_reports, author,model_name)
    # Load completed reports to DataFrame (if any)
    completed_df, completed_hashes = process_completed_csv(completed_csv_path)
    # load reports from database and filter out completed ones
    df = load_reports(num_reports, completed_hashes, author)
    # Run the pipeline
    
    results_df = run_pipeline(model, df, completed_df, result_grammar, result_grammar_exp, output_filename,config_filename,start_time, n=1)
    # Output files
    output_files(results_df, output_filename, config_filename, start_time, author, model_name)

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process EEG reports with LLM.')
    parser.add_argument('--num_reports', type=int, required=True, help='Number of reports to process')
    parser.add_argument('--completed_csv_path', type=str, default=None, help='Optional path to CSV file of completed reports')
    parser.add_argument('--author', type=str, choices=['zoe', 'maria'], default="zoe",
                        help='Report author: choose either "zoe" or "maria"')
    parser.add_argument('--model', type=str, choices=['mistral', 'deepseek','deepseek-coder','deepseek-chat','hermes-mistral','hermes-llama2',], default="mistral",
                    help='Model to use: choose "mistral" or "deepseek" or "hermes-mistral" or "hermes-llama2" or "deepseek-coder" or "deepseek-chat"')

    args = parser.parse_args()

    # Start the manager process
    manager(args.num_reports, args.completed_csv_path, args.author, args.model)
