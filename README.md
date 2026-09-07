# Automated Hypothesis Generation, Validation, and Summarization Pipeline

This project implements an end-to-end pipeline for automated hypothesis discovery from tabular datasets. The system combines statistical analysis with Large Language Models (LLMs) to generate, validate, and summarize statistically supported hypotheses.

## Pipeline Overview

The pipeline consists of the following stages:

1. Dataset preprocessing and profiling
2. Association discovery
3. Multiple testing correction (Benjamini-Hochberg FDR)
4. Effect size filtering
5. Evidence pack construction
6. LLM-based hypothesis generation
7. Statistical hypothesis validation
8. Bootstrap stability filtering
9. LLM-based hypothesis summarization

The application provides an interactive web interface built with Streamlit.

---

# Requirements

Before running the project, make sure the following software is installed.

## Python

- Python 3.10 or later

## Ollama

Install Ollama from:

https://ollama.com/

After installation, ensure that the following model is available:

```
gpt-oss:120b-cloud
```

You can verify that the model is installed by running:

```bash
ollama list
```

The output should include:

```
gpt-oss:120b-cloud
```

If the model is not available, download it using Ollama before running the application.

---

# Python Dependencies

Install the required Python packages:
```bash
pip install \
streamlit \
pandas \
numpy \
scipy \
statsmodels \
scikit-learn \
ollama \
matplotlib \
networkx
```

or, if a `requirements.txt` file is included:

```bash
pip install -r requirements.txt
```

The project uses the following libraries:

| Package | Purpose |
|----------|---------|
| streamlit | Dashboard interface |
| pandas | Data manipulation |
| numpy | Numerical computations |
| scipy | Statistical tests |
| statsmodels | Regression models and statistical analysis |
| scikit-learn | Data preprocessing and train/test splitting |
| ollama | Communication with the local Ollama LLM |

The project also uses several Python standard libraries that do not require installation:

- argparse
- itertools
- pathlib
- json
- dataclasses
- typing
- subprocess
- sys
- os
- math
- warnings

---


# Running the Application

Open a terminal in the project directory and run:

```bash
python -m streamlit run dashboard.py
```

After a few seconds, Streamlit will open automatically in your web browser.

If it does not open automatically, copy and open the URL displayed in the terminal (usually http://localhost:8501).

---

# How to Use

1. Launch the dashboard.
2. Upload a supported tabular dataset or view created results.
3. Configure any required parameters.
4. Run the pipeline.
5. The system will automatically execute:
   - Data preprocessing
   - Association discovery
   - FDR correction
   - Effect size filtering
   - Evidence pack construction
   - Hypothesis generation
   - Statistical validation
   - Bootstrap stability filtering
   - Hypothesis summarization
6. View and download the generated results.

---

# Output Files

Depending on the selected options, the pipeline produces files such as:

- Association discovery results
- Corrected association tables
- Filtered associations
- Evidence pack (JSON)
- LLM prompt
- Generated hypotheses
- Validation results
- Bootstrap stability results
- Final summarized insights

---

# Troubleshooting

## Streamlit command not found

Run Streamlit using:

```bash
python -m streamlit run dashboard.py
```

instead of:

```bash
streamlit run dashboard.py
```

## Ollama model not found

Ensure:

- Ollama is installed.
- The `gpt-oss:120b-cloud` model has been downloaded.


