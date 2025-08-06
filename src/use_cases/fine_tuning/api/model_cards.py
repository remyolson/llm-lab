"""
Model card generation with training details, performance metrics, and usage guidelines
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelCardData:
    """Data structure for model card information"""

    # Model details
    model_name: str
    model_version: str
    model_type: str
    base_model: str
    fine_tuned_by: str
    fine_tuned_date: datetime

    # Training details
    training_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    training_metrics: Dict[str, Any]
    hardware_used: Dict[str, Any]
    carbon_footprint: Optional[Dict[str, Any]]

    # Performance metrics
    evaluation_results: Dict[str, Any]
    benchmark_scores: List[Dict[str, Any]]
    comparison_baseline: Optional[Dict[str, Any]]

    # Model characteristics
    model_size: Dict[str, Any]
    inference_speed: Dict[str, Any]
    context_length: int
    vocabulary_size: int

    # Usage information
    intended_use: str
    limitations: List[str]
    ethical_considerations: List[str]
    recommendations: List[str]

    # Metadata
    license: str
    citation: str
    contact_info: str
    tags: List[str]
    language: List[str]


class ModelCardGenerator:
    """Generate comprehensive model cards"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Load model card templates"""
        return {
            "huggingface": self._get_huggingface_template(),
            "standard": self._get_standard_template(),
            "academic": self._get_academic_template(),
        }

    def generate_model_card(
        self,
        model_data: ModelCardData,
        format: str = "standard",
        include_code_examples: bool = True,
        include_visualizations: bool = True,
    ) -> str:
        """Generate a model card in the specified format"""

        if format == "huggingface":
            return self._generate_huggingface_card(model_data, include_code_examples)
        elif format == "academic":
            return self._generate_academic_card(model_data, include_visualizations)
        else:
            return self._generate_standard_card(
                model_data, include_code_examples, include_visualizations
            )

    def _generate_standard_card(
        self, data: ModelCardData, include_code: bool, include_viz: bool
    ) -> str:
        """Generate standard format model card"""

        card = f"""# {data.model_name} - Model Card

## Model Details

### Overview
- **Model Name**: {data.model_name}
- **Version**: {data.model_version}
- **Type**: {data.model_type}
- **Base Model**: {data.base_model}
- **Fine-tuned By**: {data.fine_tuned_by}
- **Date**: {data.fine_tuned_date.strftime("%Y-%m-%d")}
- **License**: {data.license}

### Model Architecture
- **Parameters**: {data.model_size.get("parameters", "N/A")}
- **Context Length**: {data.context_length} tokens
- **Vocabulary Size**: {data.vocabulary_size:,} tokens
- **Model Size on Disk**: {data.model_size.get("disk_size", "N/A")}

## Training Details

### Configuration
```json
{json.dumps(data.training_config, indent=2)}
```

### Dataset
- **Name**: {data.dataset_info.get("name", "N/A")}
- **Size**: {data.dataset_info.get("size", "N/A")} samples
- **Format**: {data.dataset_info.get("format", "N/A")}
- **Languages**: {", ".join(data.language)}

### Training Process
- **Hardware**: {data.hardware_used.get("gpu_model", "N/A")} x {data.hardware_used.get("gpu_count", 1)}
- **Training Time**: {data.hardware_used.get("training_hours", "N/A")} hours
- **Optimization**: {data.training_config.get("optimizer", "AdamW")}
- **Learning Rate**: {data.training_config.get("learning_rate", "N/A")}
- **Batch Size**: {data.training_config.get("batch_size", "N/A")}
- **Epochs**: {data.training_config.get("num_epochs", "N/A")}

### Training Metrics
{self._format_metrics_table(data.training_metrics)}

## Performance

### Evaluation Results
{self._format_evaluation_results(data.evaluation_results)}

### Benchmark Scores
{self._format_benchmark_table(data.benchmark_scores)}

"""

        if data.comparison_baseline:
            card += f"""### Comparison with Baseline
{self._format_comparison(data.comparison_baseline)}

"""

        card += f"""### Inference Performance
- **Average Latency**: {data.inference_speed.get("avg_latency_ms", "N/A")}ms
- **Throughput**: {data.inference_speed.get("tokens_per_second", "N/A")} tokens/sec
- **Memory Required**: {data.inference_speed.get("memory_gb", "N/A")}GB

## Usage

### Intended Use
{data.intended_use}

### How to Use
"""

        if include_code:
            card += self._generate_usage_examples(data)

        card += f"""
### Limitations
{self._format_list(data.limitations)}

### Ethical Considerations
{self._format_list(data.ethical_considerations)}

### Recommendations
{self._format_list(data.recommendations)}

## Environmental Impact
"""

        if data.carbon_footprint:
            card += f"""- **Training Emissions**: {data.carbon_footprint.get("co2_kg", "N/A")} kg CO2
- **Energy Consumption**: {data.carbon_footprint.get("energy_kwh", "N/A")} kWh
- **Cloud Provider**: {data.carbon_footprint.get("provider", "N/A")}
- **Region**: {data.carbon_footprint.get("region", "N/A")}
"""
        else:
            card += "Carbon footprint data not available.\n"

        card += f"""
## Citation

```bibtex
{data.citation}
```

## Contact
{data.contact_info}

## Tags
{", ".join([f"`{tag}`" for tag in data.tags])}

---
*Generated on {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}*
"""

        return card

    def _generate_huggingface_card(self, data: ModelCardData, include_code: bool) -> str:
        """Generate HuggingFace format model card"""

        # YAML frontmatter
        frontmatter = {
            "language": data.language,
            "license": data.license,
            "tags": data.tags,
            "datasets": [data.dataset_info.get("name", "custom")],
            "metrics": list(data.evaluation_results.keys()),
            "model-index": [
                {"name": data.model_name, "results": self._format_hf_results(data.benchmark_scores)}
            ],
        }

        card = f"""---
{yaml.dump(frontmatter, default_flow_style=False)}---

# {data.model_name}

## Model description
{data.intended_use}

## Training data
{data.dataset_info.get("description", "Custom dataset")}

## Training procedure

### Training hyperparameters
The following hyperparameters were used during training:
- learning_rate: {data.training_config.get("learning_rate")}
- train_batch_size: {data.training_config.get("batch_size")}
- eval_batch_size: {data.training_config.get("eval_batch_size", data.training_config.get("batch_size"))}
- seed: {data.training_config.get("seed", 42)}
- optimizer: {data.training_config.get("optimizer", "AdamW")}
- lr_scheduler_type: {data.training_config.get("scheduler", "linear")}
- num_epochs: {data.training_config.get("num_epochs")}

### Framework versions
- Transformers: {data.training_config.get("transformers_version", "4.30.0")}
- PyTorch: {data.training_config.get("pytorch_version", "2.0.0")}
- Datasets: {data.training_config.get("datasets_version", "2.0.0")}
"""

        if include_code:
            card += f"""
## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{data.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{data.model_name}")

inputs = tokenizer("Hello, I'm a language model,", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""

        card += f"""
## Limitations and bias
{self._format_list(data.limitations)}

## Citation
```bibtex
{data.citation}
```
"""

        return card

    def _generate_academic_card(self, data: ModelCardData, include_viz: bool) -> str:
        """Generate academic paper style model card"""

        card = f"""# {data.model_name}: Technical Report

## Abstract
This model card presents {data.model_name}, a fine-tuned language model based on {data.base_model}.
The model was trained on {data.dataset_info.get("size", "N/A")} samples and achieves state-of-the-art
performance on several benchmarks.

## 1. Introduction
{data.intended_use}

## 2. Model Architecture
The model is based on {data.base_model} with the following specifications:
- Parameters: {data.model_size.get("parameters", "N/A")}
- Context Length: {data.context_length} tokens
- Architecture: {data.model_type}

## 3. Training Methodology

### 3.1 Dataset
{data.dataset_info.get("description", "Dataset description not available.")}

**Statistics:**
- Total Samples: {data.dataset_info.get("size", "N/A")}
- Training/Validation Split: {data.dataset_info.get("split_ratio", "80/20")}
- Average Sequence Length: {data.dataset_info.get("avg_length", "N/A")} tokens

### 3.2 Training Procedure
Training was conducted using the following configuration:
{self._format_training_details(data.training_config)}

### 3.3 Hardware and Computational Requirements
- Hardware: {data.hardware_used.get("gpu_model", "N/A")} × {data.hardware_used.get("gpu_count", 1)}
- Training Duration: {data.hardware_used.get("training_hours", "N/A")} hours
- Peak Memory Usage: {data.hardware_used.get("peak_memory_gb", "N/A")} GB

## 4. Experimental Results

### 4.1 Performance Metrics
{self._format_academic_results(data.evaluation_results)}

### 4.2 Benchmark Comparison
{self._format_academic_benchmarks(data.benchmark_scores)}

## 5. Analysis

### 5.1 Ablation Studies
{data.comparison_baseline.get("ablation_results", "Ablation studies pending.") if data.comparison_baseline else "N/A"}

### 5.2 Error Analysis
Analysis of model errors reveals the following patterns:
{self._format_list(data.limitations)}

## 6. Ethical Considerations
{self._format_list(data.ethical_considerations)}

## 7. Conclusions and Future Work
{self._format_list(data.recommendations)}

## References
{data.citation}

## Appendix A: Hyperparameter Settings
```json
{json.dumps(data.training_config, indent=2)}
```

## Appendix B: Evaluation Details
Full evaluation results available at: [Link to results]

---
**Correspondence:** {data.contact_info}
**Date:** {datetime.utcnow().strftime("%Y-%m-%d")}
"""

        return card

    def _format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as a markdown table"""
        if not metrics:
            return "No metrics available."

        table = "| Metric | Value |\n|--------|-------|\n"
        for key, value in metrics.items():
            if isinstance(value, float):
                table += f"| {key.replace('_', ' ').title()} | {value:.4f} |\n"
            else:
                table += f"| {key.replace('_', ' ').title()} | {value} |\n"

        return table

    def _format_evaluation_results(self, results: Dict[str, Any]) -> str:
        """Format evaluation results"""
        if not results:
            return "No evaluation results available."

        output = ""
        for metric, value in results.items():
            if isinstance(value, dict):
                output += f"\n**{metric.replace('_', ' ').title()}**\n"
                for k, v in value.items():
                    output += f"- {k}: {v}\n"
            else:
                output += f"- **{metric.replace('_', ' ').title()}**: {value}\n"

        return output

    def _format_benchmark_table(self, benchmarks: List[Dict[str, Any]]) -> str:
        """Format benchmark scores as a table"""
        if not benchmarks:
            return "No benchmark scores available."

        table = "| Benchmark | Score | Baseline | Improvement |\n"
        table += "|-----------|-------|----------|-------------|\n"

        for bench in benchmarks:
            score = bench.get("score", 0)
            baseline = bench.get("baseline", 0)
            improvement = ((score - baseline) / baseline * 100) if baseline else 0

            table += f"| {bench.get('name', 'N/A')} | "
            table += f"{score:.3f} | "
            table += f"{baseline:.3f} | "
            table += f"{improvement:+.1f}% |\n"

        return table

    def _format_comparison(self, comparison: Dict[str, Any]) -> str:
        """Format model comparison"""
        output = ""
        for key, value in comparison.items():
            output += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        return output

    def _format_list(self, items: List[str]) -> str:
        """Format a list of items"""
        if not items:
            return "None specified."
        return "\n".join([f"- {item}" for item in items])

    def _generate_usage_examples(self, data: ModelCardData) -> str:
        """Generate usage code examples"""
        return f"""
```python
# Using Transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("{data.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{data.model_name}")

# Generate text
prompt = "Your prompt here"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(
    **inputs,
    max_length=100,
    temperature=0.7,
    do_sample=True
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

# Using with streaming
from transformers import TextStreamer

streamer = TextStreamer(tokenizer, skip_special_tokens=True)
outputs = model.generate(
    **inputs,
    max_length=100,
    streamer=streamer
)
```

```python
# API Usage
import requests

response = requests.post(
    "https://api.example.com/v1/completions",
    json={{
        "model": "{data.model_name}",
        "prompt": "Your prompt here",
        "max_tokens": 100,
        "temperature": 0.7
    }},
    headers={{"Authorization": "Bearer YOUR_API_KEY"}}
)
print(response.json())
```
"""

    def _format_hf_results(self, benchmarks: List[Dict[str, Any]]) -> List[Dict]:
        """Format results for HuggingFace model index"""
        results = []
        for bench in benchmarks:
            results.append(
                {
                    "task": {
                        "type": bench.get("task_type", "text-generation"),
                        "name": bench.get("name"),
                    },
                    "metrics": [
                        {
                            "type": bench.get("metric_type", "accuracy"),
                            "value": bench.get("score"),
                            "name": bench.get("metric_name", "Accuracy"),
                        }
                    ],
                }
            )
        return results

    def _format_training_details(self, config: Dict[str, Any]) -> str:
        """Format training configuration details"""
        details = ""
        for key, value in config.items():
            details += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        return details

    def _format_academic_results(self, results: Dict[str, Any]) -> str:
        """Format results in academic style"""
        output = "Our model achieves the following results:\n\n"
        for metric, value in results.items():
            output += f"- {metric.replace('_', ' ').title()}: {value}\n"
        return output

    def _format_academic_benchmarks(self, benchmarks: List[Dict[str, Any]]) -> str:
        """Format benchmarks in academic style"""
        output = "Table 1: Benchmark Performance Comparison\n\n"
        output += self._format_benchmark_table(benchmarks)
        output += "\nOur model demonstrates consistent improvements across all benchmarks."
        return output

    def _get_standard_template(self) -> str:
        """Get standard model card template"""
        return "standard"

    def _get_huggingface_template(self) -> str:
        """Get HuggingFace model card template"""
        return "huggingface"

    def _get_academic_template(self) -> str:
        """Get academic model card template"""
        return "academic"

    def save_model_card(self, card_content: str, output_path: str, format: str = "md"):
        """Save model card to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "md":
            file_path = output_path.with_suffix(".md")
        elif format == "html":
            # Convert markdown to HTML
            import markdown

            html_content = markdown.markdown(card_content, extensions=["tables", "fenced_code"])
            card_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Model Card</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        code {{ background-color: #f4f4f4; padding: 2px 4px; }}
        pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""
            file_path = output_path.with_suffix(".html")
        else:
            file_path = output_path

        file_path.write_text(card_content)
        logger.info(f"Model card saved to {file_path}")


# Example usage
if __name__ == "__main__":
    # Create sample model data
    model_data = ModelCardData(
        model_name="my-fine-tuned-model",
        model_version="1.0.0",
        model_type="Causal Language Model",
        base_model="llama-2-7b",
        fine_tuned_by="ML Team",
        fine_tuned_date=datetime.utcnow(),
        training_config={
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 3,
            "optimizer": "AdamW",
        },
        dataset_info={"name": "custom-dataset", "size": 10000, "format": "jsonl"},
        training_metrics={"final_loss": 1.234, "final_accuracy": 0.876},
        hardware_used={"gpu_model": "A100", "gpu_count": 4, "training_hours": 24},
        carbon_footprint={"co2_kg": 50, "energy_kwh": 200},
        evaluation_results={"perplexity": 12.5, "bleu_score": 0.72},
        benchmark_scores=[
            {"name": "MMLU", "score": 0.75, "baseline": 0.65},
            {"name": "HellaSwag", "score": 0.82, "baseline": 0.75},
        ],
        comparison_baseline=None,
        model_size={"parameters": "7B", "disk_size": "13GB"},
        inference_speed={"avg_latency_ms": 450, "tokens_per_second": 30},
        context_length=4096,
        vocabulary_size=32000,
        intended_use="This model is intended for customer support automation.",
        limitations=["May generate incorrect information", "Limited to English"],
        ethical_considerations=["Ensure human oversight", "Regular bias audits needed"],
        recommendations=["Use with temperature < 0.8", "Implement safety filters"],
        license="Apache 2.0",
        citation="@misc{mymodel2024, title={My Fine-Tuned Model}, author={ML Team}, year={2024}}",
        contact_info="mlteam@example.com",
        tags=["fine-tuned", "llama", "customer-support"],
        language=["en"],
    )

    # Generate model card
    generator = ModelCardGenerator()
    card = generator.generate_model_card(model_data, format="standard")
    print(card[:1000])  # Print first 1000 characters

    # Save model card
    generator.save_model_card(card, "model_card.md")
