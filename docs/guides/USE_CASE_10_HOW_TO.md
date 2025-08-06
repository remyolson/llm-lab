# Use Case 10: Synthetic Data Generation Platform

*Generate high-quality, privacy-preserving synthetic data across multiple domains using advanced LLM-powered generation techniques.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Generate synthetic datasets** for medical, legal, financial, educational, and e-commerce domains
- **Preserve privacy** using differential privacy and anonymization techniques
- **Validate data quality** with comprehensive assessment metrics
- **Scale generation** to millions of synthetic records efficiently
- **Customize generation** with domain-specific templates and constraints
- **Export in multiple formats** (JSON, CSV, Parquet, SQL) for ML workflows
- **Integrate with ML pipelines** for training data augmentation
- **Comply with regulations** (GDPR, HIPAA, CCPA) through privacy preservation

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have at least one API key configured (Google, OpenAI, or Anthropic)
- Time required: ~20-60 minutes (depending on dataset size)
- Estimated cost: $0.50-$10.00 per 10,000 synthetic records

### 💰 Cost Breakdown

Generating synthetic data with different complexity levels:

**💡 Pro Tip:** Use `--sample-size 100` for testing to reduce costs by 99% (approximately $0.005-$0.10 per test run)

- **Simple Data Generation** (10,000 records):
  - Basic e-commerce: ~$0.50 (product names, descriptions)
  - Educational content: ~$0.75 (Q&A pairs, lessons)
  - Code snippets: ~$1.00 (functions, documentation)

- **Complex Data Generation** (10,000 records):
  - Medical records: ~$3.00 (symptoms, diagnoses, treatments)
  - Legal documents: ~$5.00 (contracts, case summaries)
  - Financial reports: ~$4.00 (transactions, risk assessments)

*Note: Costs are estimates based on January 2025 pricing. Complex domains require more detailed prompting.*

## 🔧 Setup and Installation

Navigate to the synthetic data generation module:
```bash
cd src/use_cases/synthetic_data
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start - Generate Your First Dataset

### Step 1: Basic E-commerce Data Generation

Start with a simple e-commerce dataset:
```bash
# Generate 500 product records
python -m synthetic_data.cli generate \
  --domain ecommerce \
  --count 500 \
  --output products_synthetic.json

# Generate with specific categories
python -m synthetic_data.cli generate \
  --domain ecommerce \
  --categories "electronics,clothing,books" \
  --count 1000 \
  --format csv \
  --output ecommerce_data.csv
```

### Step 2: Review Generated Data

Example output structure:
```json
{
  "products": [
    {
      "id": "prod_001",
      "name": "Wireless Bluetooth Earbuds Pro",
      "category": "electronics",
      "price": 89.99,
      "description": "Premium noise-canceling earbuds with 24-hour battery life",
      "rating": 4.3,
      "reviews_count": 847,
      "in_stock": true,
      "brand": "TechSound",
      "specifications": {
        "battery_life": "24 hours",
        "connectivity": "Bluetooth 5.2",
        "water_resistance": "IPX7"
      }
    }
  ]
}
```

The generation will create:
- **Realistic product names** with appropriate technical details
- **Consistent pricing** based on category and features
- **Authentic descriptions** with marketing language
- **Correlated attributes** (high-end products have higher ratings)
- **Variety and diversity** across different product categories

## 📊 Available Data Domains

Choose the right domain based on your use case requirements:

### 🏥 **Medical Data** (`--domain medical`)
Generate realistic medical records, symptoms, and treatment data:
- Patient demographics (anonymized)
- Medical conditions and symptoms
- Treatment plans and medications
- Lab results and vital signs
- Clinical notes and observations

**Example Command:**
```bash
python -m synthetic_data.cli generate \
  --domain medical \
  --privacy-level high \
  --count 1000 \
  --include-demographics \
  --output medical_records.json
```

**Privacy Features:**
- HIPAA-compliant anonymization
- Differential privacy for sensitive attributes
- Realistic but non-identifiable patient data

---

### ⚖️ **Legal Documents** (`--domain legal`)
Generate contracts, case summaries, and legal documentation:
- Contract templates and clauses
- Case law summaries
- Legal briefs and motions
- Compliance documentation
- Terms of service and privacy policies

**Example Command:**
```bash
python -m synthetic_data.cli generate \
  --domain legal \
  --document-types "contracts,briefs,policies" \
  --count 500 \
  --complexity detailed \
  --output legal_documents.json
```

**Use Cases:**
- Legal AI training data
- Contract analysis systems
- Compliance testing datasets

---

### 💰 **Financial Data** (`--domain financial`)
Generate transaction records, market data, and financial reports:
- Transaction histories
- Account information (anonymized)
- Market data and prices
- Financial statements
- Risk assessment reports

**Example Command:**
```bash
python -m synthetic_data.cli generate \
  --domain financial \
  --privacy-level maximum \
  --include-transactions \
  --date-range "2023-01-01,2024-12-31" \
  --count 10000 \
  --output financial_data.parquet
```

**Compliance Features:**
- PCI DSS compliant transaction generation
- GDPR privacy preservation
- Realistic financial patterns without real data exposure

---

### 📚 **Educational Content** (`--domain educational`)
Generate learning materials, Q&A pairs, and curriculum content:
- Question-answer pairs
- Lesson plans and curricula
- Student performance data (anonymized)
- Educational assessments
- Learning resource metadata

**Example Command:**
```bash
python -m synthetic_data.cli generate \
  --domain educational \
  --subjects "mathematics,science,history" \
  --grade-levels "6,7,8,9" \
  --count 2000 \
  --include-assessments \
  --output educational_content.json
```

**Features:**
- Age-appropriate content generation
- Curriculum alignment
- Diverse difficulty levels and learning styles

---

### 💻 **Code Generation** (`--domain code`)
Generate code samples, documentation, and development artifacts:
- Function implementations
- Code documentation
- API specifications
- Bug reports and issues
- Code review comments

**Example Command:**
```bash
python -m synthetic_data.cli generate \
  --domain code \
  --languages "python,javascript,java" \
  --complexity "beginner,intermediate,advanced" \
  --count 1500 \
  --include-tests \
  --output code_samples.json
```

**Applications:**
- Code completion model training
- Programming tutorial generation
- Software testing datasets

## 🔄 Advanced Generation Patterns

### Multi-Domain Data Generation
```bash
# Generate related datasets across domains
python -m synthetic_data.cli generate-suite \
  --domains "ecommerce,financial" \
  --relationships "customer-transaction" \
  --count 5000 \
  --output integrated_dataset/
```

### Time-Series Data Generation
```bash
# Generate temporal data with realistic patterns
python -m synthetic_data.cli generate \
  --domain financial \
  --type timeseries \
  --start-date "2023-01-01" \
  --end-date "2024-12-31" \
  --frequency daily \
  --seasonality \
  --output financial_timeseries.csv
```

### Custom Template Generation
```bash
# Use custom templates for specific formats
python -m synthetic_data.cli generate \
  --template ./custom_templates/healthcare_form.yaml \
  --count 1000 \
  --privacy-config ./privacy_rules.json \
  --output custom_healthcare.json
```

### Batch Generation with Quality Control
```bash
# Generate large datasets with quality validation
python -m synthetic_data.cli batch-generate \
  --domain medical \
  --batch-size 1000 \
  --total-count 50000 \
  --quality-threshold 0.95 \
  --output-dir ./large_medical_dataset/
```

## 🔒 Privacy and Compliance Features

### Differential Privacy Configuration
```python
# Configure differential privacy parameters
from src.use_cases.synthetic_data import PrivacyEngine, SyntheticDataGenerator

privacy_config = {
    "epsilon": 1.0,  # Privacy budget
    "delta": 1e-5,   # Privacy parameter
    "sensitivity": 1.0,  # Data sensitivity
    "noise_multiplier": 1.1
}

generator = SyntheticDataGenerator(
    privacy_engine=PrivacyEngine(privacy_config)
)
```

### GDPR Compliance Mode
```bash
# Generate GDPR-compliant synthetic data
python -m synthetic_data.cli generate \
  --domain medical \
  --gdpr-compliant \
  --anonymization-level high \
  --no-quasi-identifiers \
  --output gdpr_medical_data.json
```

### Custom Privacy Rules
```yaml
# privacy_rules.yaml
privacy_rules:
  pii_removal:
    - names
    - addresses
    - phone_numbers
    - email_addresses

  anonymization:
    method: k_anonymity
    k_value: 5

  sensitive_attributes:
    - medical_conditions
    - financial_status
    - personal_beliefs

  retention_policy:
    max_age_days: 365
    auto_delete: true
```

## 📊 Data Quality Assessment

### Built-in Quality Metrics
```bash
# Assess generated data quality
python -m synthetic_data.cli assess-quality \
  --dataset generated_data.json \
  --reference-schema schema.json \
  --metrics "completeness,accuracy,consistency,diversity" \
  --output quality_report.html
```

### Custom Quality Validation
```python
from src.use_cases.synthetic_data import DataValidator

validator = DataValidator()
quality_report = validator.assess_dataset(
    dataset_path="medical_synthetic.json",
    validation_rules="medical_validation_rules.yaml",
    statistical_tests=True
)

print(f"Overall Quality Score: {quality_report.overall_score}")
print(f"Completeness: {quality_report.completeness}")
print(f"Diversity Index: {quality_report.diversity}")
```

## 🔧 Customization and Configuration

### Create Custom Generation Configuration
```python
# custom_generator.py
from src.use_cases.synthetic_data import SyntheticDataGenerator, GeneratorConfig

config = GeneratorConfig(
    domain="custom_healthcare",
    output_format="json",
    privacy_level="maximum",
    quality_threshold=0.9,
    batch_size=100,
    generation_model="gpt-4o-mini",
    templates_path="./custom_templates/",
    schema_path="./schemas/healthcare_schema.json"
)

generator = SyntheticDataGenerator(config)
dataset = await generator.generate_dataset(count=5000)
```

### Domain-Specific Templates
```yaml
# templates/medical_patient.yaml
patient_template:
  demographics:
    age_range: [18, 85]
    gender_distribution: ["male", "female", "other"]
    ethnicity_diversity: true

  medical_history:
    conditions_count: [0, 5]
    medications_count: [0, 10]
    allergies_probability: 0.3

  visit_patterns:
    frequency: "realistic_distribution"
    seasonal_variations: true
    emergency_probability: 0.05

privacy_settings:
  anonymize_identifiers: true
  add_noise_to_dates: true
  generalize_locations: true
```

## 📈 Scaling and Performance

### Distributed Generation
```bash
# Scale generation across multiple processes
python -m synthetic_data.cli generate \
  --domain financial \
  --count 1000000 \
  --parallel-workers 8 \
  --chunk-size 10000 \
  --output-format parquet \
  --output financial_large_dataset.parquet
```

### Cloud Integration
```python
# Integration with cloud storage and computing
from src.use_cases.synthetic_data import CloudGenerator

cloud_generator = CloudGenerator(
    provider="aws",
    compute_resources={"instance_type": "m5.2xlarge", "worker_count": 10},
    storage_config={"bucket": "synthetic-data-bucket", "region": "us-east-1"}
)

dataset = await cloud_generator.generate_large_dataset(
    domain="medical",
    count=10000000,
    output_location="s3://synthetic-data-bucket/medical_10m/"
)
```

## 🔗 Integration with ML Workflows

### Data Augmentation Pipeline
```python
# augment_training_data.py
from src.use_cases.synthetic_data import DataAugmenter

augmenter = DataAugmenter()

# Augment existing dataset with synthetic data
augmented_data = augmenter.augment_dataset(
    original_data="real_training_data.json",
    augmentation_ratio=2.0,  # 2x synthetic data
    domain="medical",
    preserve_distribution=True
)

# Save augmented dataset for training
augmented_data.to_parquet("augmented_training_data.parquet")
```

### MLflow Integration
```python
import mlflow
from src.use_cases.synthetic_data import SyntheticDataGenerator

with mlflow.start_run():
    generator = SyntheticDataGenerator()

    # Track generation parameters
    mlflow.log_params({
        "domain": "financial",
        "count": 10000,
        "privacy_level": "high"
    })

    # Generate dataset
    dataset = generator.generate_dataset(domain="financial", count=10000)

    # Log dataset as artifact
    mlflow.log_artifact("synthetic_financial_data.json", "datasets")

    # Log quality metrics
    quality_metrics = dataset.assess_quality()
    mlflow.log_metrics(quality_metrics)
```

## 📊 Export and Integration Formats

### Multiple Export Formats
```bash
# Export to different formats for various tools
python -m synthetic_data.cli convert \
  --input synthetic_data.json \
  --output-formats "csv,parquet,sql,xlsx" \
  --output-dir ./exports/

# Generate SQL insert statements
python -m synthetic_data.cli export-sql \
  --input medical_data.json \
  --table-name patients \
  --database-type postgresql \
  --output patients_insert.sql
```

### API Integration
```python
from src.use_cases.synthetic_data import SyntheticDataAPI

# RESTful API for on-demand generation
api = SyntheticDataAPI()

@api.route('/generate/<domain>')
async def generate_data(domain, count=100, format='json'):
    generator = SyntheticDataGenerator()
    dataset = await generator.generate_dataset(
        domain=domain,
        count=count
    )
    return dataset.export(format=format)
```

## 🚀 Next Steps

1. **Start Small:** Begin with simple domains like e-commerce to understand the generation process
2. **Validate Quality:** Always assess generated data quality before using in production
3. **Implement Privacy:** Configure appropriate privacy settings for your compliance requirements
4. **Scale Gradually:** Start with small datasets and scale up as you validate results
5. **Customize Templates:** Create domain-specific templates for your unique use cases
6. **Monitor Performance:** Track generation costs and quality metrics over time

---

*This guide covers the complete synthetic data generation capabilities available in the LLM Lab platform. The synthetic data platform provides enterprise-grade privacy preservation, quality validation, and scalable generation across multiple domains. For additional support, refer to the [Troubleshooting Guide](./TROUBLESHOOTING.md) or reach out via GitHub issues.*
