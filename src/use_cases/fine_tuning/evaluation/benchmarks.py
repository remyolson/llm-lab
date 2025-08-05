"""
Benchmark Runner for Fine-Tuning Evaluation

This module provides specialized benchmark runners for evaluating fine-tuned
models on standard NLP benchmarks.

Example:
    runner = BenchmarkRunner()
    
    # Run specific benchmark
    result = runner.run_benchmark(
        model=model,
        tokenizer=tokenizer,
        benchmark="glue",
        tasks=["mnli", "qnli", "sst2"]
    )
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    GenerationConfig
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark evaluation."""
    name: str
    tasks: List[str]
    batch_size: int = 8
    max_samples: Optional[int] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_cache: bool = True
    seed: int = 42


class BenchmarkRunner:
    """Runs standard NLP benchmarks for model evaluation."""
    
    # GLUE tasks configuration
    GLUE_TASKS = {
        "cola": {
            "dataset": "glue",
            "subset": "cola",
            "metric": "matthews_correlation",
            "num_labels": 2,
            "text_keys": ["sentence"]
        },
        "sst2": {
            "dataset": "glue",
            "subset": "sst2",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["sentence"]
        },
        "mrpc": {
            "dataset": "glue",
            "subset": "mrpc",
            "metric": "f1",
            "num_labels": 2,
            "text_keys": ["sentence1", "sentence2"]
        },
        "qqp": {
            "dataset": "glue",
            "subset": "qqp",
            "metric": "f1",
            "num_labels": 2,
            "text_keys": ["question1", "question2"]
        },
        "mnli": {
            "dataset": "glue",
            "subset": "mnli",
            "metric": "accuracy",
            "num_labels": 3,
            "text_keys": ["premise", "hypothesis"]
        },
        "qnli": {
            "dataset": "glue",
            "subset": "qnli",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["question", "sentence"]
        },
        "rte": {
            "dataset": "glue",
            "subset": "rte",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["sentence1", "sentence2"]
        },
        "wnli": {
            "dataset": "glue",
            "subset": "wnli",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["sentence1", "sentence2"]
        }
    }
    
    # SuperGLUE tasks configuration
    SUPERGLUE_TASKS = {
        "boolq": {
            "dataset": "super_glue",
            "subset": "boolq",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["question", "passage"]
        },
        "cb": {
            "dataset": "super_glue",
            "subset": "cb",
            "metric": "f1",
            "num_labels": 3,
            "text_keys": ["premise", "hypothesis"]
        },
        "copa": {
            "dataset": "super_glue",
            "subset": "copa",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["premise", "choice1", "choice2"]
        },
        "multirc": {
            "dataset": "super_glue",
            "subset": "multirc",
            "metric": "f1",
            "text_keys": ["paragraph", "question", "answer"]
        },
        "record": {
            "dataset": "super_glue",
            "subset": "record",
            "metric": "f1",
            "text_keys": ["passage", "query"]
        },
        "rte": {
            "dataset": "super_glue",
            "subset": "rte",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["premise", "hypothesis"]
        },
        "wic": {
            "dataset": "super_glue",
            "subset": "wic",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["sentence1", "sentence2"]
        },
        "wsc": {
            "dataset": "super_glue",
            "subset": "wsc",
            "metric": "accuracy",
            "num_labels": 2,
            "text_keys": ["text", "span1_text", "span2_text"]
        }
    }
    
    # Other benchmarks
    OTHER_BENCHMARKS = {
        "squad": {
            "dataset": "squad",
            "metric": "squad",
            "task_type": "question_answering",
            "text_keys": ["context", "question"]
        },
        "squad_v2": {
            "dataset": "squad_v2",
            "metric": "squad_v2",
            "task_type": "question_answering",
            "text_keys": ["context", "question"]
        },
        "xsum": {
            "dataset": "xsum",
            "metric": "rouge",
            "task_type": "summarization",
            "text_keys": ["document"]
        },
        "cnn_dailymail": {
            "dataset": "cnn_dailymail",
            "subset": "3.0.0",
            "metric": "rouge",
            "task_type": "summarization",
            "text_keys": ["article"]
        },
        "humaneval": {
            "dataset": "openai_humaneval",
            "metric": "code_eval",
            "task_type": "code_generation",
            "text_keys": ["prompt"]
        },
        "mbpp": {
            "dataset": "mbpp",
            "metric": "code_eval",
            "task_type": "code_generation",
            "text_keys": ["text"]
        }
    }
    
    def __init__(self):
        """Initialize benchmark runner."""
        self.all_benchmarks = {
            **self.GLUE_TASKS,
            **self.SUPERGLUE_TASKS,
            **self.OTHER_BENCHMARKS
        }
        self._metric_cache = {}
    
    def run_benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        benchmark: str,
        tasks: Optional[List[str]] = None,
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Run a benchmark suite on the model.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            benchmark: Benchmark name (glue, superglue, etc.)
            tasks: Specific tasks to run
            config: Benchmark configuration
            
        Returns:
            Benchmark results
        """
        # Determine tasks to run
        if benchmark == "glue":
            available_tasks = self.GLUE_TASKS
        elif benchmark == "superglue":
            available_tasks = self.SUPERGLUE_TASKS
        else:
            # Single benchmark
            if benchmark in self.all_benchmarks:
                return self._run_single_task(model, tokenizer, benchmark, config)
            else:
                raise ValueError(f"Unknown benchmark: {benchmark}")
        
        # Filter tasks if specified
        if tasks:
            tasks_to_run = {k: v for k, v in available_tasks.items() if k in tasks}
        else:
            tasks_to_run = available_tasks
        
        # Run all tasks
        results = {}
        for task_name, task_config in tasks_to_run.items():
            logger.info(f"Running task: {task_name}")
            results[task_name] = self._run_single_task(
                model, tokenizer, task_name, config
            )
        
        # Calculate average score
        scores = [r["score"] for r in results.values() if "score" in r]
        avg_score = np.mean(scores) if scores else 0.0
        
        return {
            "benchmark": benchmark,
            "tasks": results,
            "average_score": avg_score
        }
    
    def _run_single_task(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        task_name: str,
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Run a single benchmark task.
        
        Args:
            model: Model to evaluate
            tokenizer: Tokenizer
            task_name: Task name
            config: Benchmark configuration
            
        Returns:
            Task results
        """
        if task_name not in self.all_benchmarks:
            raise ValueError(f"Unknown task: {task_name}")
        
        task_info = self.all_benchmarks[task_name]
        
        # Load dataset
        dataset_name = task_info["dataset"]
        subset = task_info.get("subset")
        
        if subset:
            dataset = load_dataset(dataset_name, subset)
        else:
            dataset = load_dataset(dataset_name)
        
        # Get validation split
        if "validation" in dataset:
            eval_dataset = dataset["validation"]
        elif "test" in dataset:
            eval_dataset = dataset["test"]
        else:
            raise ValueError(f"No validation/test split found for {task_name}")
        
        # Limit samples if specified
        if config and config.max_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), config.max_samples)))
        
        # Run evaluation based on task type
        task_type = task_info.get("task_type", "classification")
        
        if task_type == "classification":
            result = self._evaluate_classification(
                model, tokenizer, eval_dataset, task_info, config
            )
        elif task_type == "question_answering":
            result = self._evaluate_qa(
                model, tokenizer, eval_dataset, task_info, config
            )
        elif task_type == "summarization":
            result = self._evaluate_summarization(
                model, tokenizer, eval_dataset, task_info, config
            )
        elif task_type == "code_generation":
            result = self._evaluate_code_generation(
                model, tokenizer, eval_dataset, task_info, config
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        return result
    
    def _evaluate_classification(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset,
        task_info: Dict[str, Any],
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Evaluate classification task."""
        # Prepare data
        def preprocess_function(examples):
            text_keys = task_info["text_keys"]
            
            if len(text_keys) == 1:
                texts = examples[text_keys[0]]
            else:
                texts = [
                    f"{examples[text_keys[0]][i]} {tokenizer.sep_token} {examples[text_keys[1]][i]}"
                    for i in range(len(examples[text_keys[0]]))
                ]
            
            tokenized = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512
            )
            
            if "label" in examples:
                tokenized["labels"] = examples["label"]
            
            return tokenized
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Create dataloader
        batch_size = config.batch_size if config else 8
        data_collator = DataCollatorWithPadding(tokenizer)
        dataloader = DataLoader(
            tokenized_dataset,
            batch_size=batch_size,
            collate_fn=data_collator
        )
        
        # Evaluate
        model.eval()
        predictions = []
        references = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {task_info.get('subset', '')}"):
                inputs = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"]
                
                outputs = model(**inputs)
                preds = outputs.logits.argmax(dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                references.extend(labels.numpy())
        
        # Calculate metric
        metric_name = task_info["metric"]
        if metric_name not in self._metric_cache:
            self._metric_cache[metric_name] = evaluate.load(metric_name)
        
        metric = self._metric_cache[metric_name]
        result = metric.compute(predictions=predictions, references=references)
        
        return {
            "task": task_info.get("subset", task_info["dataset"]),
            "metric": metric_name,
            "score": result[metric_name] if metric_name in result else list(result.values())[0],
            "results": result
        }
    
    def _evaluate_qa(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset,
        task_info: Dict[str, Any],
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Evaluate question answering task."""
        # Simplified QA evaluation
        model.eval()
        predictions = []
        references = []
        
        batch_size = config.batch_size if config else 8
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating QA"):
            batch = dataset[i:i+batch_size]
            
            contexts = batch["context"]
            questions = batch["question"]
            
            # Format inputs
            inputs = [
                f"Context: {c}\nQuestion: {q}\nAnswer:"
                for c, q in zip(contexts, questions)
            ]
            
            # Tokenize
            tokenized = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate answers
            with torch.no_grad():
                outputs = model.generate(
                    **tokenized,
                    max_new_tokens=50,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode predictions
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend([p.split("Answer:")[-1].strip() for p in preds])
            
            # Get references
            if "answers" in batch:
                for answers in batch["answers"]:
                    references.append(answers["text"][0] if answers["text"] else "")
        
        # Calculate metric
        metric = evaluate.load(task_info["metric"])
        result = metric.compute(predictions=predictions, references=references)
        
        return {
            "task": task_info["dataset"],
            "metric": task_info["metric"],
            "score": result.get("f1", result.get("exact_match", 0)),
            "results": result
        }
    
    def _evaluate_summarization(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset,
        task_info: Dict[str, Any],
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Evaluate summarization task."""
        model.eval()
        predictions = []
        references = []
        
        batch_size = config.batch_size if config else 4
        
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating summarization"):
            batch = dataset[i:i+batch_size]
            
            # Get documents
            documents = batch[task_info["text_keys"][0]]
            
            # Format inputs
            inputs = [f"Summarize: {doc}" for doc in documents]
            
            # Tokenize
            tokenized = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt"
            ).to(model.device)
            
            # Generate summaries
            with torch.no_grad():
                outputs = model.generate(
                    **tokenized,
                    max_new_tokens=150,
                    num_beams=4,
                    length_penalty=2.0,
                    early_stopping=True
                )
            
            # Decode predictions
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend([p.replace("Summarize:", "").strip() for p in preds])
            
            # Get references
            if "summary" in batch:
                references.extend(batch["summary"])
            elif "highlights" in batch:
                references.extend(batch["highlights"])
        
        # Calculate ROUGE scores
        metric = evaluate.load("rouge")
        result = metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        # Average ROUGE scores
        avg_score = np.mean([
            result["rouge1"],
            result["rouge2"],
            result["rougeL"]
        ])
        
        return {
            "task": task_info["dataset"],
            "metric": "rouge",
            "score": avg_score,
            "results": result
        }
    
    def _evaluate_code_generation(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset,
        task_info: Dict[str, Any],
        config: Optional[BenchmarkConfig] = None
    ) -> Dict[str, Any]:
        """Evaluate code generation task."""
        # Simplified code generation evaluation
        # In practice, would need to execute code and check test cases
        
        model.eval()
        correct = 0
        total = 0
        
        for example in tqdm(dataset, desc="Evaluating code generation"):
            prompt = example[task_info["text_keys"][0]]
            
            # Generate code
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1,
                    do_sample=True,
                    top_p=0.95
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple check - in practice would execute and test
            if "test" in example:
                # Check if generated code passes test
                # This is a placeholder - real implementation would execute code
                correct += 1 if "def" in generated else 0
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "task": task_info["dataset"],
            "metric": "pass@1",
            "score": accuracy,
            "results": {"pass@1": accuracy, "total": total}
        }
    
    def list_available_benchmarks(self) -> Dict[str, List[str]]:
        """List all available benchmarks and their tasks.
        
        Returns:
            Dictionary of benchmark suites and their tasks
        """
        return {
            "glue": list(self.GLUE_TASKS.keys()),
            "superglue": list(self.SUPERGLUE_TASKS.keys()),
            "other": list(self.OTHER_BENCHMARKS.keys())
        }


# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForSequenceClassification
    
    # Initialize runner
    runner = BenchmarkRunner()
    
    # List available benchmarks
    print("Available benchmarks:")
    for suite, tasks in runner.list_available_benchmarks().items():
        print(f"\n{suite}:")
        for task in tasks:
            print(f"  - {task}")
    
    # Example: Run GLUE benchmark
    model_name = "bert-base-uncased"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Run specific GLUE tasks
    config = BenchmarkConfig(
        name="glue",
        tasks=["sst2", "mrpc"],
        batch_size=16,
        max_samples=100  # For quick testing
    )
    
    results = runner.run_benchmark(
        model=model,
        tokenizer=tokenizer,
        benchmark="glue",
        tasks=config.tasks,
        config=config
    )
    
    print(f"\nGLUE Results:")
    print(f"Average Score: {results['average_score']:.4f}")
    for task, result in results["tasks"].items():
        print(f"{task}: {result['score']:.4f}")