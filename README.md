# LLM-Based Clinical Report Analysis

This project analyzes clinical EEG reports using a combination of traditional machine learning models and LLMs like Mistral via `llama.cpp`. The pipeline processes text data, runs multiple classification models, evaluates performance, and analyzes supporting evidence.

## Project Structure

```
.
├── data/                       # Raw data (e.g., SQLite .db file)
│   └── zoe_reports_10.db

├── environment.yml            # Conda environment file (Python + NLP tools)

├── outputs/                   # Output artifacts and intermediate results
│   ├── baseline_results/      # ML model outputs
│   │   ├── inference_results/
│   │   ├── trained_models/
│   │   └── training_results/
│   ├── evidence_analysis_results/   # Polarity classification output
│   ├── performance_plots/           # Visualization results (bar charts, etc.)
│   ├── pipeline_output/             # Output from LLM pipeline
│   │   ├── mistral_zoe_config_*.txt
│   │   └── mistral_zoe_first_10_results_*.csv
│   └── processed_output/            # Final processed results

└── src/                      # Source code
    ├── baseline_models/      # BoW + LR and SHAP explanation scripts
    │   ├── train.py
    │   ├── inference.py
    │   └── shap_explanations.ipynb
    ├── evidence_analysis/    # Rule-based and model-based polarity classification
    │   └── evidence_polarity_classification.py
    ├── LLM_pipeline/         # LLM-based classification and grammar-based parsing
    │   ├── pipeline.py
    │   ├── process_output.py
    │   ├── result_grammar.gbnf
    │   └── result_grammar_test.gbnf
    └── performance_analysis/ # Accuracy, agreement metrics, visualization
        ├── main.py
        ├── analysis_functions.py
        ├── result_preprocessing.py
        └── plotting.py
```

## Installation

Ensure you have [conda](https://docs.conda.io/en/latest/) installed. Then create the environment:

```bash
conda env create -f environment.yml
conda activate llama_grammar
```

If you encounter dependency resolution issues (slow install or solver errors), consider installing [**Mamba**](https://mamba.readthedocs.io/en/latest/installation.html) for faster and more robust environment solving:

```bash
conda install -n base -c conda-forge mamba
```

Then use:

```bash
mamba env create -f environment.yml
```

## Usage

### 1. Run the LLM pipeline

If your machine has GPU, make sure to install llama_cpp with GPU support:
```bash
CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```
Then check the current GPU status using NVIDIA’s System Management Interface
```bash
nvidia-smi
```

If your machine has CPU only, ignore the previous step.
To run the pipeline:
```bash
python src/LLM_pipeline/pipeline.py --num_reports 10 --author zoe --model mistral 
```

### 2. Process LLM output
```bash
python src/LLM_pipeline/process_output.py mistral_zoe_first_10_results_v1.csv
```

### 3. Train and evaluate baseline models
```bash
python src/baseline_models/train.py
```
(input baseline model when prompted: bag_of_words or bert_base)

```bash
python src/baseline_models/inference.py --model bert_base
python src/baseline_models/inference.py --model bag_of_words
```

### 4. Analyze evidence polarity
```bash
python src/evidence_analysis/evidence_polarity_classification.py
```

### 5. Generate plots and evaluation results
```bash
python src/performance_analysis/main.py
```

## Features

- BoW + Logistic Regression baseline
- Sentence-transformer and transformer-based pipelines
- LLM grammar-based parsing via `llama-cpp-python`
- SHAP explanation integration
- Rule-based and model-based evidence polarity classification
- Performance visualization

## Requirements

- Python 3.12
- `llama-cpp-python` (for running GGUF models)
- `sentence-transformers`, `transformers`, `torch`
- `pandas`, `scikit-learn`, `matplotlib`, etc.

See `environment.yml` for full list.


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
