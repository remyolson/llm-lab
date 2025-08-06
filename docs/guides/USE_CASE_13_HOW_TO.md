# Use Case 13: Benchmark Creation Tool

*Comprehensive platform for creating, validating, and managing custom benchmarks and evaluation datasets for Large Language Models across diverse domains and tasks.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Create custom benchmarks** tailored to your specific domain and use cases
- **Generate diverse test cases** using advanced template-based generation
- **Validate benchmark quality** with comprehensive statistical and semantic analysis
- **Support multiple task types** (classification, generation, Q&A, reasoning, coding)
- **Scale benchmark creation** to thousands of high-quality test cases
- **Export in standard formats** compatible with popular evaluation frameworks
- **Collaborate on benchmark development** with version control and team features
- **Integrate with existing pipelines** for automated benchmark generation and updates

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Define your evaluation objectives and target capabilities
- Have sample data or domain expertise for benchmark creation
- Time required: ~30-120 minutes (depending on benchmark complexity)
- Estimated cost: $0.20-$5.00 per 1,000 benchmark questions/tasks

### 💰 Cost Breakdown

Creating benchmarks with different complexity and domains:

**💡 Pro Tip:** Use `--sample-size 50` for testing to reduce costs by 95% (approximately $0.01-$0.25 per test run)

- **Simple Question-Answer Benchmarks:**
  - Basic factual Q&A (1,000 questions): ~$0.20
  - Multiple choice questions: ~$0.30
  - Reading comprehension: ~$0.50

- **Complex Domain-Specific Benchmarks:**
  - Technical reasoning tasks: ~$1.50
  - Code generation benchmarks: ~$2.00
  - Mathematical problem solving: ~$2.50
  - Creative writing evaluation: ~$3.00

- **Comprehensive Benchmark Suites:**
  - Multi-domain evaluation suite: ~$5.00
  - Adaptive difficulty progression: ~$7.50
  - Cross-lingual benchmark creation: ~$10.00

*Note: Costs are estimates based on January 2025 pricing. Complex domains requiring expert knowledge cost more.*

## 🔧 Setup and Installation

Navigate to the benchmark creation module:
```bash
cd src/use_cases/benchmark_creation
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start - Create Your First Benchmark

### Step 1: Simple Question-Answer Benchmark

Create a basic Q&A benchmark for your domain:
```bash
# Generate a simple factual Q&A benchmark
python -m benchmark_builder.cli create-benchmark \
  --type "question_answer" \
  --domain "technology" \
  --count 100 \
  --difficulty "mixed" \
  --output tech_qa_benchmark.json

# Create multiple choice questions
python -m benchmark_builder.cli create-benchmark \
  --type "multiple_choice" \
  --domain "science" \
  --options 4 \
  --count 200 \
  --include-explanations \
  --output science_mcq_benchmark.json
```

### Step 2: Review Generated Benchmark

Example benchmark structure:
```json
{
  "benchmark_metadata": {
    "name": "Technology Q&A Benchmark",
    "version": "1.0",
    "created_date": "2025-01-06",
    "total_questions": 100,
    "domains": ["artificial_intelligence", "software_engineering", "cybersecurity"],
    "difficulty_distribution": {
      "easy": 30,
      "medium": 50,
      "hard": 20
    }
  },
  "questions": [
    {
      "id": "tech_qa_001",
      "question": "What is the primary advantage of transformer architecture over RNN-based models in natural language processing?",
      "answer": "Transformers can process sequences in parallel rather than sequentially, leading to faster training and better handling of long-range dependencies.",
      "category": "artificial_intelligence",
      "difficulty": "medium",
      "topics": ["transformers", "neural_networks", "nlp"],
      "evaluation_criteria": {
        "key_concepts": ["parallelization", "attention_mechanism", "long_range_dependencies"],
        "expected_length": "1-3 sentences",
        "technical_accuracy_required": true
      },
      "metadata": {
        "source": "generated",
        "validation_status": "reviewed",
        "last_updated": "2025-01-06"
      }
    }
  ]
}
```

Quality indicators included:
- **Domain relevance:** Questions align with specified technical domains
- **Difficulty progression:** Balanced distribution across skill levels
- **Answer quality:** Comprehensive, accurate responses with proper technical detail
- **Evaluation criteria:** Clear guidelines for automated and human evaluation

## 📊 Benchmark Types and Templates

### 🤖 **Question-Answer Benchmarks** (`--type question_answer`)
Create comprehensive Q&A datasets for knowledge evaluation:
```bash
# Domain-specific factual Q&A
python -m benchmark_builder.cli create-benchmark \
  --type "question_answer" \
  --domain "medicine" \
  --question-types "factual,diagnostic,treatment" \
  --difficulty-levels "medical_student,resident,attending" \
  --count 500 \
  --include-medical-disclaimers \
  --output medical_qa_benchmark.json

# Multi-hop reasoning questions
python -m benchmark_builder.cli create-benchmark \
  --type "question_answer" \
  --reasoning-type "multi_hop" \
  --knowledge-base "custom_kb.json" \
  --reasoning-steps "2,3,4" \
  --count 300 \
  --output reasoning_benchmark.json
```

**Features:**
- Single and multi-hop reasoning questions
- Domain-specific terminology and concepts
- Graduated difficulty levels
- Comprehensive answer explanations

---

### ✅ **Multiple Choice Questions** (`--type multiple_choice`)
Generate MCQs with distractors and explanations:
```bash
# Technical multiple choice with detailed explanations
python -m benchmark_builder.cli create-benchmark \
  --type "multiple_choice" \
  --domain "computer_science" \
  --subtopics "algorithms,data_structures,complexity_analysis" \
  --options 4 \
  --distractor-quality "high" \
  --explanation-detail "comprehensive" \
  --count 250 \
  --output cs_mcq_benchmark.json

# Adaptive difficulty MCQ generation
python -m benchmark_builder.cli create-benchmark \
  --type "multiple_choice" \
  --adaptive-difficulty \
  --starting-difficulty "medium" \
  --difficulty-adjustment-factor 0.1 \
  --target-accuracy 0.75 \
  --count 400 \
  --output adaptive_mcq_benchmark.json
```

**Distractor Generation:**
- Semantically plausible wrong answers
- Common misconception-based distractors
- Graduated difficulty in option selection
- Explanation of why each option is correct/incorrect

---

### 💻 **Code Generation Benchmarks** (`--type code_generation`)
Create programming and code evaluation tasks:
```bash
# Programming challenges with test cases
python -m benchmark_builder.cli create-benchmark \
  --type "code_generation" \
  --programming-languages "python,java,javascript" \
  --difficulty-range "easy,medium,hard" \
  --include-test-cases \
  --include-edge-cases \
  --count 150 \
  --output programming_benchmark.json

# Algorithm implementation tasks
python -m benchmark_builder.cli create-benchmark \
  --type "code_generation" \
  --focus "algorithms" \
  --algorithm-categories "sorting,searching,graph,dynamic_programming" \
  --complexity-requirements "time,space" \
  --include-optimization-variants \
  --count 100 \
  --output algorithm_benchmark.json
```

**Code Quality Features:**
- Comprehensive test case generation
- Multiple solution approaches
- Performance and complexity analysis
- Code style and best practices evaluation

---

### 📚 **Reading Comprehension** (`--type reading_comprehension`)
Generate text comprehension and analysis tasks:
```bash
# Multi-document comprehension tasks
python -m benchmark_builder.cli create-benchmark \
  --type "reading_comprehension" \
  --text-sources "scientific_papers,news_articles,literature" \
  --comprehension-types "factual,inferential,critical" \
  --passage-length "short,medium,long" \
  --multi-document-tasks \
  --count 200 \
  --output comprehension_benchmark.json

# Domain-specific reading comprehension
python -m benchmark_builder.cli create-benchmark \
  --type "reading_comprehension" \
  --domain "legal" \
  --document-types "contracts,case_law,regulations" \
  --legal-reasoning-required \
  --citation-accuracy-evaluation \
  --count 150 \
  --output legal_comprehension_benchmark.json
```

**Comprehension Features:**
- Multiple question types per passage
- Inference and critical thinking evaluation
- Multi-document synthesis tasks
- Domain-specific comprehension challenges

---

### 🧮 **Mathematical Reasoning** (`--type mathematical_reasoning`)
Create math problem-solving and reasoning benchmarks:
```bash
# Mathematical problem solving across skill levels
python -m benchmark_builder.cli create-benchmark \
  --type "mathematical_reasoning" \
  --math-domains "algebra,geometry,calculus,statistics" \
  --skill-levels "high_school,undergraduate,graduate" \
  --problem-types "computational,proof_based,application" \
  --step-by-step-solutions \
  --count 300 \
  --output math_reasoning_benchmark.json

# Word problem generation with real-world context
python -m benchmark_builder.cli create-benchmark \
  --type "mathematical_reasoning" \
  --context-domains "finance,physics,engineering" \
  --word-problem-complexity "multi_step" \
  --real-world-applications \
  --count 200 \
  --output applied_math_benchmark.json
```

**Mathematical Features:**
- LaTeX-formatted mathematical expressions
- Step-by-step solution verification
- Multiple solution path validation
- Real-world application contexts

## 🔄 Advanced Benchmark Creation

### Template-Based Generation
```yaml
# custom_template.yaml
benchmark_template:
  name: "Customer Service Evaluation"
  description: "Assess LLM performance in customer service scenarios"

  scenarios:
    - type: "complaint_resolution"
      variables: ["product_type", "issue_severity", "customer_tone"]
      templates:
        - "A customer is {customer_tone} about their {product_type} because {issue_description}"

    - type: "product_inquiry"
      variables: ["product_category", "inquiry_type", "customer_expertise"]
      templates:
        - "A {customer_expertise} customer asks about {inquiry_type} for {product_category}"

  evaluation_criteria:
    - empathy_score: "Rate empathy level (1-5)"
    - solution_quality: "Evaluate solution effectiveness"
    - communication_clarity: "Assess response clarity"
```

```bash
# Use custom templates
python -m benchmark_builder.cli create-from-template \
  --template custom_template.yaml \
  --generation-count 500 \
  --validation-rules validation_config.json \
  --output custom_benchmark.json
```

### Multi-Modal Benchmark Creation
```bash
# Create benchmarks with images, text, and audio
python -m benchmark_builder.cli create-multimodal-benchmark \
  --modalities "text,image,audio" \
  --task-types "image_description,audio_transcription,multimodal_qa" \
  --media-sources "./media_assets/" \
  --count 200 \
  --output multimodal_benchmark.json
```

### Collaborative Benchmark Development
```python
from src.use_cases.benchmark_creation import CollaborativeBenchmark

# Set up collaborative benchmark creation
collaborative = CollaborativeBenchmark()

# Create benchmark project
project = collaborative.create_project(
    name="Medical AI Evaluation Suite",
    collaborators=["domain_expert@hospital.com", "ml_engineer@company.com"],
    review_requirements={"min_reviews": 2, "expert_approval": True}
)

# Distributed question generation
questions = collaborative.distributed_generation(
    project_id=project.id,
    assignments={
        "cardiology": "cardiologist@hospital.com",
        "radiology": "radiologist@hospital.com",
        "general_medicine": "ml_engineer@company.com"
    }
)
```

## 📊 Quality Validation and Assessment

### Automated Quality Checks
```bash
# Comprehensive quality assessment
python -m benchmark_builder.cli validate-benchmark \
  --benchmark-file custom_benchmark.json \
  --validation-types "statistical,semantic,difficulty,bias" \
  --quality-threshold 0.8 \
  --output quality_report.html

# Cross-validation with existing benchmarks
python -m benchmark_builder.cli cross-validate \
  --new-benchmark custom_benchmark.json \
  --reference-benchmarks "mmlu.json,arc.json" \
  --similarity-threshold 0.3 \
  --output validation_results.json
```

### Human Review Integration
```python
from src.use_cases.benchmark_creation import HumanReviewSystem

review_system = HumanReviewSystem()

# Set up human review workflow
review_config = {
    "review_stages": ["content_accuracy", "difficulty_assessment", "bias_check"],
    "reviewers_per_question": 3,
    "inter_reviewer_agreement_threshold": 0.7,
    "expert_review_required": True
}

# Submit benchmark for review
review_session = review_system.submit_for_review(
    benchmark="medical_qa_benchmark.json",
    config=review_config,
    deadline="2025-02-01"
)
```

### Statistical Analysis
```bash
# Comprehensive statistical analysis of benchmark properties
python -m benchmark_builder.cli analyze-statistics \
  --benchmark-file large_benchmark.json \
  --analysis-types "difficulty_distribution,topic_coverage,length_statistics" \
  --generate-plots \
  --output statistical_analysis/
```

## 🔧 Customization and Configuration

### Domain-Specific Generators
```python
# Create specialized generator for your domain
from src.use_cases.benchmark_creation import DomainSpecificGenerator

class FinancialBenchmarkGenerator(DomainSpecificGenerator):
    def __init__(self):
        super().__init__(domain="finance")
        self.load_financial_ontology()
        self.load_regulatory_requirements()

    def generate_risk_assessment_questions(self, count=100):
        return self.generate_questions(
            question_type="risk_assessment",
            complexity_levels=["basic", "intermediate", "advanced"],
            regulatory_frameworks=["basel_iii", "mifid_ii", "dodd_frank"],
            count=count
        )

    def generate_trading_scenarios(self, count=50):
        return self.generate_scenarios(
            scenario_type="trading_decision",
            market_conditions=["bull", "bear", "volatile", "stable"],
            asset_classes=["equity", "fixed_income", "derivatives"],
            count=count
        )

# Use specialized generator
fin_generator = FinancialBenchmarkGenerator()
risk_questions = fin_generator.generate_risk_assessment_questions(200)
trading_scenarios = fin_generator.generate_trading_scenarios(100)
```

### Custom Evaluation Metrics
```yaml
# evaluation_metrics.yaml
custom_metrics:
  domain_expertise:
    weight: 0.3
    evaluation_method: "expert_rating"
    scale: [1, 5]
    description: "Assess domain-specific knowledge accuracy"

  practical_applicability:
    weight: 0.25
    evaluation_method: "real_world_validation"
    criteria: ["feasibility", "relevance", "usefulness"]

  communication_clarity:
    weight: 0.2
    evaluation_method: "automated_readability"
    metrics: ["flesch_kincaid", "gunning_fog", "coleman_liau"]

  bias_detection:
    weight: 0.15
    evaluation_method: "bias_analysis"
    bias_types: ["gender", "racial", "cultural", "socioeconomic"]

  creativity_originality:
    weight: 0.1
    evaluation_method: "novelty_scoring"
    comparison_corpus: "existing_benchmarks"
```

## 📈 Integration with Evaluation Frameworks

### Hugging Face Datasets Integration
```python
from src.use_cases.benchmark_creation import HuggingFaceIntegration

# Export to Hugging Face Datasets format
hf_integration = HuggingFaceIntegration()

dataset = hf_integration.convert_to_hf_dataset(
    benchmark_file="custom_benchmark.json",
    dataset_name="custom_evaluation_suite",
    description="Domain-specific evaluation benchmark",
    tags=["evaluation", "custom", "domain_specific"]
)

# Upload to Hugging Face Hub
dataset.push_to_hub(
    repo_id="your_org/custom_evaluation_suite",
    token="your_hf_token"
)
```

### OpenAI Evals Framework Integration
```bash
# Export to OpenAI Evals format
python -m benchmark_builder.cli export-openai-evals \
  --benchmark-file custom_benchmark.json \
  --eval-name "custom_domain_eval" \
  --eval-description "Domain-specific capability assessment" \
  --output openai_evals/
```

### Integration with Popular Benchmarking Libraries
```python
# Integration with lm-evaluation-harness
from src.use_cases.benchmark_creation import LMEvalHarnessIntegration

lm_eval = LMEvalHarnessIntegration()

# Convert benchmark for lm-evaluation-harness
lm_eval.convert_benchmark(
    input_benchmark="custom_benchmark.json",
    task_name="custom_domain_task",
    output_directory="./lm_eval_tasks/",
    include_few_shot_examples=True
)
```

## 🚀 Advanced Features

### Adaptive Benchmark Generation
```python
from src.use_cases.benchmark_creation import AdaptiveBenchmarkGenerator

adaptive_gen = AdaptiveBenchmarkGenerator()

# Generate benchmark that adapts to model performance
adaptive_benchmark = adaptive_gen.create_adaptive_benchmark(
    initial_questions=100,
    difficulty_adaptation_rate=0.1,
    target_accuracy_range=(0.6, 0.8),
    max_total_questions=500,
    adaptation_criteria=["accuracy", "response_time", "confidence"]
)
```

### Cross-Lingual Benchmark Creation
```bash
# Generate multilingual benchmarks
python -m benchmark_builder.cli create-multilingual-benchmark \
  --base-language "english" \
  --target-languages "spanish,french,german,mandarin" \
  --translation-quality-check \
  --cultural-adaptation \
  --count 300 \
  --output multilingual_benchmark/
```

### Benchmark Versioning and Evolution
```python
from src.use_cases.benchmark_creation import BenchmarkVersionControl

version_control = BenchmarkVersionControl()

# Track benchmark evolution
version_control.create_version(
    benchmark="medical_benchmark.json",
    version="2.0",
    changes=["added_pediatric_questions", "updated_diagnostic_criteria"],
    compatibility_matrix={"1.0": "backward_compatible", "1.5": "partial"}
)

# Generate benchmark diffs
diff_report = version_control.generate_diff(
    old_version="medical_benchmark_v1.json",
    new_version="medical_benchmark_v2.json",
    output_format="html"
)
```

## 🔗 Integration with Other Use Cases

- **Use Cases 1-4:** Use custom benchmarks in standard evaluation pipelines
- **Use Case 6:** Create benchmarks for fine-tuned model evaluation
- **Use Case 8:** Include custom benchmarks in continuous monitoring
- **Use Case 9:** Create security-focused evaluation benchmarks

## 🚀 Next Steps

1. **Define Evaluation Goals:** Start by clearly defining what capabilities you want to assess
2. **Start Simple:** Begin with basic question-answer benchmarks before moving to complex tasks
3. **Validate Quality:** Always validate benchmark quality with both automated and human review
4. **Iterate and Improve:** Continuously refine benchmarks based on evaluation results
5. **Share and Collaborate:** Consider sharing high-quality benchmarks with the research community
6. **Monitor Performance:** Track how models perform on your custom benchmarks over time

---

*This guide provides comprehensive coverage of benchmark creation capabilities for evaluating LLM performance across diverse domains and tasks. The benchmark creation platform supports enterprise-grade quality validation, collaboration features, and integration with popular evaluation frameworks. For additional support, refer to the [Troubleshooting Guide](./TROUBLESHOOTING.md) or reach out via GitHub issues.*
