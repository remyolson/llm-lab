# LLM Lab Documentation

Welcome to the comprehensive documentation for LLM Lab - a professional-grade framework for benchmarking, fine-tuning, monitoring, and aligning Large Language Models.

## 📚 Documentation Index

### 🚀 Getting Started
- [**Prerequisites**](guides/PREREQUISITES.md) - System requirements and initial setup
- [**Quick Start Guide**](../README.md#-installation) - Get up and running in minutes
- [**Use Cases Overview**](guides/USE_CASES_OVERVIEW.md) - What you can do with LLM Lab

### 📖 API Reference
- [**API Overview**](api/README.md) - Complete API documentation index
- [**Providers API**](api/providers.md) - LLM provider interfaces
- [**Configuration API**](api/configuration.md) - Configuration management
- [**Evaluation API**](api/evaluation.md) - Evaluation metrics and methods
- [**Analysis API**](api/analysis.md) - Results analysis and comparison *(coming soon)*

### 🏗️ Architecture
- [**System Architecture**](architecture/README.md) - High-level design and patterns
- [**Component Architecture**](architecture/README.md#component-architecture) - Detailed component design
- [**Data Flow**](architecture/README.md#data-flow) - Request and data flow diagrams
- [**Extension Points**](architecture/README.md#extension-points) - How to extend the system

### 📋 User Guides

#### Core Use Cases
1. [**Run Standard Benchmarks**](guides/USE_CASE_1_HOW_TO.md) - Evaluate models on standard datasets
2. [**Compare Cost vs Performance**](guides/USE_CASE_2_HOW_TO.md) - Optimize for budget constraints
3. [**Test Custom Prompts**](guides/USE_CASE_3_HOW_TO.md) - Evaluate domain-specific prompts
4. [**Run Tests Across LLMs**](guides/USE_CASE_4_HOW_TO.md) - Cross-model testing

#### Advanced Use Cases
5. [**Local LLM Testing**](guides/USE_CASE_5_HOW_TO.md) - Work with self-hosted models
6. [**Fine-tune Local LLMs**](guides/USE_CASE_6_HOW_TO.md) - Customize models with LoRA/QLoRA
7. [**Alignment Research**](guides/USE_CASE_7_HOW_TO.md) - Implement safety measures
8. [**Continuous Monitoring**](guides/USE_CASE_8_HOW_TO.md) - Production monitoring

#### Specialized Guides
- [**Custom Evaluation Metrics**](guides/CUSTOM_EVALUATION_METRICS.md) - Create domain-specific metrics
- [**Cross-LLM Testing Setup**](guides/CROSS_LLM_TESTING_SETUP.md) - Configure multi-provider tests
- [**Custom Prompt CLI**](guides/CUSTOM_PROMPT_CLI.md) - Command-line prompt testing
- [**Cost Estimates**](guides/COST_ESTIMATES.md) - Understanding and managing costs
- [**Troubleshooting**](guides/TROUBLESHOOTING.md) - Common issues and solutions

### 🔧 Provider Documentation
- [**OpenAI Provider**](providers/openai.md) - GPT models integration
- [**Anthropic Provider**](providers/anthropic.md) - Claude models integration
- [**Google Provider**](providers/google.md) - Gemini models integration
- [**Adding New Providers**](api/providers.md#extension-points) - Implement custom providers

### 🛠️ Development
- [**Contributing Guide**](../CONTRIBUTING.md) - How to contribute to LLM Lab
- [**Development Setup**](development/CI_CD_GUIDE.md) - Setting up development environment
- [**Project Roadmap**](development/FUTURE_DIRECTIONS.md) - Future plans and features
- [**Architecture Decisions**](development/project_plan.md) - Design rationale

### 📊 Examples & Tutorials
- [**Code Examples**](../examples/README.md) - Practical implementation examples
- [**Jupyter Notebooks**](../examples/notebooks/) - Interactive tutorials
- [**Use Case Demos**](../examples/use_cases/) - Complete workflow examples

### 🔄 Maintenance
- [**Changelog**](../CHANGELOG.md) - Version history and updates
- [**Migration Guide**](MIGRATION_GUIDE.md) - Upgrading between versions
- [**Security Policy**](../SECURITY.md) - Reporting vulnerabilities

## 🔍 Quick Reference

### Common Tasks

| Task | Documentation |
|------|---------------|
| Set up API keys | [Prerequisites](guides/PREREQUISITES.md#2-api-keys) |
| Run first benchmark | [Use Case 1](guides/USE_CASE_1_HOW_TO.md#basic-usage) |
| Create custom evaluator | [Custom Metrics](guides/CUSTOM_EVALUATION_METRICS.md) |
| Monitor production models | [Use Case 8](guides/USE_CASE_8_HOW_TO.md) |
| Fine-tune a model | [Use Case 6](guides/USE_CASE_6_HOW_TO.md) |
| Debug issues | [Troubleshooting](guides/TROUBLESHOOTING.md) |

### Key Concepts

| Concept | Description | Learn More |
|---------|-------------|------------|
| Providers | LLM API integrations | [Providers API](api/providers.md) |
| Evaluators | Response evaluation logic | [Evaluation API](api/evaluation.md) |
| Benchmarks | Standard test datasets | [Benchmarking Guide](guides/USE_CASE_1_HOW_TO.md) |
| Monitoring | Production tracking | [Monitoring Guide](guides/USE_CASE_8_HOW_TO.md) |
| Alignment | Safety and ethics | [Alignment Guide](guides/USE_CASE_7_HOW_TO.md) |

## 📝 Documentation Standards

- **Code Examples**: All API docs include working examples
- **Diagrams**: Architecture docs use Mermaid diagrams
- **Versioning**: Docs are versioned with the codebase
- **Search**: Use GitHub's search to find specific topics

## 🤝 Need Help?

- Check the [Troubleshooting Guide](guides/TROUBLESHOOTING.md)
- Search [existing issues](https://github.com/remyolson/lllm-lab/issues)
- Ask in [GitHub Discussions](https://github.com/remyolson/lllm-lab/discussions)
- Open a [new issue](https://github.com/remyolson/lllm-lab/issues/new)

---

*Documentation last updated: January 2025*