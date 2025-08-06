# Fine-Tuning Datasets Documentation

This directory contains documentation and utilities for working with datasets for fine-tuning large language models.

## 📚 Available Datasets

### Featured Dataset: NVIDIA Nemotron Post-Training Dataset v1

The **Nemotron Post-Training Dataset** is a comprehensive collection of 25.7 million high-quality examples designed for post-training and fine-tuning large language models. It's specifically optimized for improving model capabilities across multiple domains.

#### Key Features:
- **Size**: 25.7M rows across 5 specialized splits
- **License**: CC BY 4.0 (Creative Commons Attribution)
- **Format**: Parquet files for efficient loading
- **Domains**: Math, Code, STEM, Chat, and Tool Calling

#### Dataset Splits:
1. **chat** - General conversation and instruction-following
2. **code** - Programming and code generation tasks
3. **math** - Mathematical reasoning and problem-solving
4. **stem** - Science, Technology, Engineering, Mathematics questions
5. **tool_calling** - Function calling and tool use examples

#### Quick Start:
```python
from src.use_cases.fine_tuning.datasets.dataset_processor import DatasetProcessor

# Initialize processor
processor = DatasetProcessor()

# Load specific split
chat_data = processor.load_nemotron_split(split="chat", streaming=True)

# Or load entire dataset
nemotron = processor.load_dataset_from_registry("nemotron-post-training")
```

## 🗂️ Dataset Registry

The project includes a comprehensive dataset registry with popular fine-tuning datasets:

### Instruction-Following Datasets
- **Stanford Alpaca** - 52K instruction-following examples
- **Databricks Dolly 15K** - Human-generated instruction-response pairs
- **FLAN v2** - 15M multi-task instruction examples

### Conversation Datasets
- **OpenAssistant (OASST1)** - Human-annotated conversations
- **Nemotron Chat Split** - High-quality conversational data

### Mathematical Reasoning
- **GSM8K** - Grade school math word problems
- **MATH** - Competition mathematics problems
- **Nemotron Math Split** - Advanced mathematical reasoning

### Code Generation
- **CodeParrot Clean** - Cleaned Python code from GitHub
- **HumanEval** - Hand-written programming problems
- **Nemotron Code Split** - Code generation and completion

### Tool Calling & Function Use
- **ToolBench** - External tool and API usage
- **Nemotron Tool Calling Split** - Function calling examples

### Question Answering
- **SQuAD 2.0** - Reading comprehension with unanswerable questions

## 🚀 Usage Examples

### Loading from Registry

```python
from src.use_cases.fine_tuning.datasets.dataset_registry import DatasetRegistry

# Get dataset information
dataset_info = DatasetRegistry.get_dataset("nemotron-post-training")
print(f"Dataset: {dataset_info.name}")
print(f"Size: {dataset_info.size}")
print(f"Splits: {dataset_info.splits}")

# List all available datasets
all_datasets = DatasetRegistry.list_datasets()
for dataset in all_datasets:
    print(f"- {dataset.name}: {dataset.description}")

# Search datasets by task
math_datasets = DatasetRegistry.get_datasets_by_task("mathematical-reasoning")
```

### Processing Datasets

```python
from src.use_cases.fine_tuning.datasets.dataset_processor import DatasetProcessor

processor = DatasetProcessor(
    max_length=2048,
    validation_split=0.1,
    test_split=0.05
)

# Load from registry
dataset = processor.load_dataset_from_registry(
    "nemotron-post-training",
    split="math",
    streaming=True,
    max_samples=10000
)

# Process for training
formatted = processor.format_for_instruction_tuning(dataset)

# Split into train/val/test
splits = processor.split_dataset(formatted)
```

### Streaming Large Datasets

For large datasets like Nemotron (25.7M rows), use streaming mode:

```python
# Stream dataset to avoid loading everything into memory
processor = DatasetProcessor()

# Load with streaming
dataset = processor.load_nemotron_split(
    split="code",
    streaming=True
)

# Process in batches
batch_size = 1000
for i, batch in enumerate(dataset.take(10000).batch(batch_size)):
    # Process batch
    print(f"Processing batch {i}: {len(batch['text'])} examples")
    # Your processing logic here
```

## 📊 Dataset Statistics

### Nemotron Dataset Size Breakdown:
| Split | Approximate Rows | Primary Use Case |
|-------|-----------------|------------------|
| chat | ~5M | General conversation, instruction-following |
| code | ~5M | Code generation, completion, debugging |
| math | ~5M | Mathematical reasoning, problem-solving |
| stem | ~5M | Scientific and technical Q&A |
| tool_calling | ~5M | Function calling, API usage |

## 🔧 Configuration

### Dataset Processor Options

```python
processor = DatasetProcessor(
    tokenizer=tokenizer,           # Optional tokenizer for encoding
    max_length=2048,               # Maximum sequence length
    validation_split=0.1,          # 10% for validation
    test_split=0.05,              # 5% for testing
    seed=42                       # Random seed for reproducibility
)
```

### Loading Arguments

The Nemotron dataset supports various loading configurations:

```python
# Full dataset (not recommended due to size)
full_dataset = processor.load_dataset_from_registry(
    "nemotron-post-training",
    streaming=False  # Will load entire dataset into memory
)

# Streaming mode (recommended)
stream_dataset = processor.load_dataset_from_registry(
    "nemotron-post-training",
    streaming=True  # Efficient memory usage
)

# Limited samples for testing
test_dataset = processor.load_dataset_from_registry(
    "nemotron-post-training",
    split="chat",
    max_samples=1000,
    streaming=False
)
```

## 📝 Adding New Datasets

To add a new dataset to the registry:

1. Edit `src/use_cases/fine_tuning/datasets/dataset_registry.py`
2. Add your dataset to the `DATASETS` dictionary:

```python
"your-dataset": DatasetInfo(
    name="Your Dataset Name",
    hf_path="organization/dataset-name",
    description="Description of your dataset",
    dataset_type=DatasetType.INSTRUCTION,
    size="100K rows",
    splits=["train", "test"],
    license="Apache 2.0",
    tasks=["task1", "task2"],
    format="json"
)
```

## 🧪 Testing Dataset Integration

Run the demo script to test dataset loading:

```bash
# Show Nemotron dataset info
python examples/use_cases/nemotron_dataset_demo.py

# Preview specific split
python examples/use_cases/nemotron_dataset_demo.py --split math --max-samples 10

# List all available datasets
python examples/use_cases/nemotron_dataset_demo.py --list-datasets

# Show loading code examples
python examples/use_cases/nemotron_dataset_demo.py --show-code

# Run processing demonstration
python examples/use_cases/nemotron_dataset_demo.py --process-demo
```

## 📚 References

- [Nemotron Dataset on HuggingFace](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1)
- [Dataset Card](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v1/blob/main/README.md)
- [Llama-3.3-Nemotron Model](https://huggingface.co/nvidia/Llama-3.3-Nemotron-Super-49B-v1.5)

## 🔒 License Information

The Nemotron Post-Training Dataset is licensed under **CC BY 4.0**, which allows:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

With the requirement of:
- ⚠️ Attribution to NVIDIA

Always check individual dataset licenses before use in production.
