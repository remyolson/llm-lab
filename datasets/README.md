# LLM Datasets Collection

This repository contains curated samples of popular LLM benchmarking and fine-tuning datasets from Hugging Face.

## Dataset Structure

```
datasets/
├── benchmarking/          # Evaluation datasets
│   └── raw/              # Original format samples
│       ├── mmlu/         # Multi-task language understanding
│       ├── hellaswag/    # Common sense reasoning
│       ├── arc/          # AI2 Reasoning Challenge
│       ├── truthfulqa/   # Truthfulness evaluation
│       └── gsm8k/        # Grade school math problems
├── fine-tuning/          # Training datasets
│   └── raw/              # Original format samples
│       ├── alpaca/       # Instruction-following dataset
│       ├── openorca/     # OpenOrca conversations
│       ├── dolly-15k/    # Databricks Dolly dataset
│       └── wizardlm/     # WizardLM evolved instructions
└── manifest.json         # Complete dataset metadata
```

## Benchmarking Datasets

### MMLU (Massive Multitask Language Understanding)
- **Size**: 1,000 samples
- **Format**: Multiple choice questions across various subjects
- **Fields**: question, subject, choices, answer

### HellaSwag
- **Size**: 1,000 samples
- **Format**: Commonsense reasoning with context completion
- **Fields**: context, endings, label

### ARC (AI2 Reasoning Challenge)
- **Size**: 1,000 samples
- **Format**: Science questions with multiple choices
- **Fields**: question, choices, answerKey

### TruthfulQA
- **Size**: 500 samples
- **Format**: Questions testing truthfulness
- **Fields**: question, best_answer, correct_answers, incorrect_answers

### GSM8K (Grade School Math)
- **Size**: 1,000 samples
- **Format**: Math word problems with solutions
- **Fields**: question, answer

## Fine-tuning Datasets

### Alpaca
- **Size**: 1,000 samples
- **Format**: Instruction-input-output pairs
- **Fields**: instruction, input, output, text

### OpenOrca
- **Size**: 1,000 samples
- **Format**: System prompt with Q&A pairs
- **Fields**: system_prompt, question, response

### Dolly-15k
- **Size**: 2,000 samples (from 15,011 total)
- **Format**: Instructions with optional context
- **Fields**: instruction, context, response, category

### WizardLM Evol Instruct
- **Size**: 1,000 samples
- **Format**: Multi-turn conversations
- **Fields**: idx, conversations

## Usage

Each dataset is available in both JSON and CSV formats in their respective directories. The `metadata.json` file in each dataset folder contains additional information about the dataset.

### Loading Examples

```python
import pandas as pd
import json

# Load CSV
df = pd.read_csv('datasets/benchmarking/raw/mmlu/sample.csv')

# Load JSON
with open('datasets/benchmarking/raw/mmlu/sample.json', 'r') as f:
    data = json.load(f)
```

## Download Script

To refresh or update the datasets, use the provided download script:

```bash
source venv/bin/activate
python datasets/download_datasets.py
```

Note: Some datasets may require authentication or have changed locations on Hugging Face.
