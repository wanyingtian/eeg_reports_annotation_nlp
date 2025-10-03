# LLM-Based Clinical Report Analysis

This project analyzes clinical EEG reports using a combination of traditional machine learning models and LLMs like Mistral via `llama.cpp`. The repository contains the full LLM pipeline and baseline training scripts.

## Project Structure

```
.
├── data/                       # Raw sample data (e.g., SQLite .db file)
│   └── zoe_reports_sample.db

├── requirements.txt           # environment file (Python + NLP tools)

├── outputs/                   # Output artifacts and intermediate results
│   ├── baseline_results/      # ML model outputs
│   │   ├── inference_results/
│   │   ├── trained_models/
│   │   └── training_results/
│   ├── pipeline_output/             # Output from LLM pipeline
│   │   ├── mistral_zoe_config_*.txt
│   │   └── mistral_zoe_first_10_results_*.csv
│   └── processed_output/            # Final processed results

└── src/                      # Source code
    ├── baseline_models/      # BoW + LR and SHAP explanation scripts
    │   ├── train.py
    │   ├── inference.py
    │   └── shap_explanations.ipynb
    ├── evidence_analysis     # Evidence evaluation
    │   ├── evidence_factuality.py
    │   └── evidence_alignment.py
    └── LLM_pipeline/         # LLM-based classification and grammar-based parsing
        ├── pipeline.py
        ├── process_output.py
        ├── result_grammar.gbnf
        └── result_grammar_exp.gbnf

```

## Installation

1. Ensure Python 3.10 is installed (tested on 3.10.12)
2. Create virtual environment: 
```bash
python -m venv venv
```
3. Activate: 
```bash
source venv/bin/activate # (Linux/Mac) 
# or 
source venv\Scripts\activate # (Windows)
```
4. Install: 
```bash
pip install -r requirements.txt
```
### Optional: Deployment on FIR Cluster

FIR provides some dependencies (e.g., Rust, PyArrow) via environment modules.  
If you are deploying on Fir, make sure to load these before creating or activating your virtual environment.

```bash
# example setup for FIR

# 1.load modules required for building/installing
module load gcc
module load rust          # needed for HuggingFace `tokenizers`
module load arrow         # provides pyarrow 
# 2.activate venv
source venv/bin/activate
# 3. install 
pip install -r requirements.txt
```
## Usage

### 1. Run the LLM pipeline

#### GPU build
If your machine has an **NVIDIA GPU**, install llama-cpp-python with CUDA/cuBLAS support:
```bash
CUDACXX=$(which nvcc) \
CMAKE_ARGS="-DGGML_CUDA=on" \
FORCE_CMAKE=1 \
pip install --no-binary=:all: llama-cpp-python
```
To confirm the GPU build is working, run your pipeline as usual and, in a separate terminal, check GPU activity:

```bash
nvidia-smi
```
If you see Python using GPU memory, the GPU build is successful.
If not, you probably have a CPU-only build.
#### CPU build
If your machine has **CPU** only, ignore the previous step, proceed.

#### Running pipeline.py
By default, outputs go to: outputs/pipeline_output/ (from the repo root).

To run the pipeline (basic), with default sample data (10 reports) and mistral model:
```bash
python src/LLM_pipeline/pipeline.py --num-reports 10 --model mistral 
```

##### Customizations: 
* If you want to resume from an existing CSV that is partially complete from previous runs:
```bash
python src/LLM_pipeline/pipeline.py --num-reports 10 --model mistral \
  --completed-csv path/to/previous.csv
```
* Process 10 reports with custom dataset identifier, custom datapath, and output directory:
```bash
python src/LLM_pipeline/pipeline.py --num-reports 10 --model mistral --dataset-id "john_data"  --dataset-path /path/to/john_data.db --outdir /custom/output/dir
```
* The default temperature is 0, making results definitive, to explore other configurations:
```bash
python src/LLM_pipeline/pipeline.py --num-reports 50 --model mistral --temperature 0.7 --top-k 40 --top-p 0.95
```


#### How output files are named

Results are saved as CSVs with this pattern:
```bash
raw_{dataset_id}_{model}_{num_reports}_v{version}_run{run}.csv
```
- **dataset_id** → from `--dataset-id` (or dataset filename if not provided)  
- **model** → the model you selected (e.g., `mistral`)  
- **num_reports** → the number of reports you requested (`--num-reports`)  
- **v{version}** → version number, increments if you change configuration  
- **run{run}** → run number, increments if you keep the same config but don’t overwrite  

When you run the pipeline with the same dataset and model that previous runs have used, users are given options to: 
1. extend from previous results and only run additional reports that do not exist in previous results
2. rerun using previous config (run +1)
3. rerun using new config (version +1)

Each version also saves a config file:
```bash
config_{dataset_id}_{model}_v{version}.json
```

To see all other potential usage of pipeline.py, you can run:
```bash
python src/LLM_pipeline/pipeline.py --help
```
You will see:
```bash
usage: pipeline.py [-h] --num-reports NUM_REPORTS [--completed-csv COMPLETED_CSV] [--dataset-id DATASET_ID] [--dataset-path DATASET_PATH]
                   [--model {mistral,deepseek,deepseek-chat,hermes-mistral,hermes-llama2}] [--outdir OUTDIR] [--comment COMMENT] [--temperature TEMPERATURE] [--top-k TOP_K] [--top-p TOP_P]
                   [--max-tokens MAX_TOKENS] [-v]

Process EEG reports with an LLM.

options:
  -h, --help            show this help message and exit
  --num-reports NUM_REPORTS
                        Required: Number of reports to run.
  --completed-csv COMPLETED_CSV
                        Optional: Path to an existing results CSV to resume from.
  --dataset-id DATASET_ID
                        Optional: Dataset identifier (e.g., "zoe", "johns_data"). If not provided, uses dataset filename.
  --dataset-path DATASET_PATH
                        Optional: Path to the dataset SQLite file. If not provided, uses default sample dataset.
  --model {mistral,deepseek,deepseek-chat,hermes-mistral,hermes-llama2}
                        Model to use (GGUF). If not provided, defaults to 'mistral'.
  --outdir OUTDIR       Optional: Directory to write outputs. Defaults to ./outputs/pipeline_output
  --comment COMMENT     Optional: comment to save in config.
  --temperature TEMPERATURE
                        Optional: Sampling temperature (0 for greedy). Defaults to 0.
  --top-k TOP_K         Optional: Top-k sampling cutoff. Defaults to 40.
  --top-p TOP_P         Optional: Top-p (nucleus) sampling threshold. Defaults to 0.95.
  --max-tokens MAX_TOKENS
                        Optional: Max new tokens to generate. Defaults to 3000.
  -v, --verbose         Increase verbosity (-v, -vv).
```

### 2. Process LLM output
By default, the script reads from outputs/pipeline_output/ and writes to outputs/processed_output/.
```bash
python src/LLM_pipeline/process_output.py raw_zoe_reports_sample_mistral_5_v1_run1.csv
```
This produces:

* Excel: outputs/processed_output/processed_raw_zoe_reports_sample_mistral_5_v1_run1.xlsx

(sheets: classifications, explanations)

* SQLite: outputs/processed_output/processed_raw_zoe_reports_sample_mistral_5_v1_run1.db

(tables: classifications, explanations)

* Errors (if any): outputs/processed_output/errors_log.csv


### 3. Train and evaluate baseline models
Note: train.py will not work on the default sample db, you need to replace or pass in the path to an annotated database:
```bash
python src/baseline_models/train.py --model bert_base --dataset-path /path/to/annotated/db
python src/baseline_models/train.py --model bag_of_words --dataset-path /path/to/annotated/db
```
The db should have the following columns: 'Hashed_ID', 'Report', 'Focal Epi', 'Gen Epi', 'Focal Non-epi', 'Gen Non-epi', 'Abnormality'
If the db format is different, please modify the fetch_reports function accordingly.

```bash
python src/baseline_models/inference.py --model bert_base
python src/baseline_models/inference.py --model bag_of_words
```

### 4. LLM Evidence Analysis

#### 4.1 Evidence Factuality
```bash
python src/evidence_analysis/evidence_factuality.py /path/to/processed_llm_output_file
```
This will produce: evidence_factuality_stats.txt, which reports percentage of explanation factuality (traceability) to original reports.

#### 4.2 Evidence Alignment
There are two methods to assign polarity label to explnations:
1. Rule-based (regex matching)
2. Clinical BERT + LR (requires some labelled data for training - not supplied in repo)

To try the rule-based method, no need for additional training data:
```bash
python src/evidence_analysis/evidence_alignment.py /path/to/processed_llm_output_file
```

To try Clinical BERT + LR, you should have a train file that has labelled columns such as "Abnormality Reasons Polarity" in the explanation sheet. The labels should be based on the semantic of the provided reasons (-1 as normal, 1 as abnormal)

Then,
```bash
python src/evidence_analysis/evidence_alignment.py /path/to/processed_llm_output_file --method clinicalbert --train-data /path/to/train_data
```

<!-- 
### 5. Generate plots and evaluation results
```bash
python src/performance_analysis/main.py
``` -->

## Features
- [x] LLM-based pipeline for standardized report annotation
- [x] LLM grammar-based parsing via `llama-cpp-python`
- [x] BoW + Logistic Regression baseline
- [x] BERT_base + Logistic Regression baseline
- [x] SHAP explanation visualization for baselines
- [x] evidence polarity classification 


## Requirements

- Python 3.10
- `llama-cpp-python` (for running GGUF models)

See `requirements.txt` for full list.


## License
This repository is licensed under the Apache License 2.0.

Copyright 2025 Wanying Tian

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note: Models downloaded via hf_hub_download have their own licenses. Users are responsible for reviewing and complying with the license terms of any models they download.

## Contact

If you use or modify this code, feel free to reach out or submit improvements via pull request.
