# Use Case 9: LLM Security Testing Framework

*Comprehensive security vulnerability detection and attack resistance testing for Large Language Models across multiple providers.*

## 🎯 What You'll Accomplish

By following this guide, you'll be able to:
- **Scan LLMs for security vulnerabilities** using 500+ categorized attack patterns
- **Test jailbreak resistance** with advanced prompt injection techniques
- **Detect data leakage vulnerabilities** including PII and credential exposure
- **Analyze response patterns** for manipulation and bias detection
- **Generate comprehensive security reports** with confidence scoring
- **Run parallel security scans** with intelligent batching for performance
- **Integrate security testing** into your CI/CD pipelines
- **Create custom attack libraries** tailored to your specific use cases

## 📋 Before You Begin

- Complete all [Prerequisites](./PREREQUISITES.md)
- Ensure you have at least one API key configured (Google, OpenAI, or Anthropic)
- Time required: ~15-45 minutes (depending on scan depth)
- Estimated cost: $0.10-$2.00 per comprehensive security scan

### 💰 Cost Breakdown

Running a comprehensive security scan with different attack categories:

**💡 Pro Tip:** Use `--limit 50` for testing to reduce costs by 90% (approximately $0.01-$0.20 per test run)

- **Basic Security Scan** (100 attack prompts):
  - `gpt-4o-mini`: ~$0.10
  - `claude-3-haiku`: ~$0.15
  - `gemini-1.5-flash`: ~$0.05

- **Comprehensive Security Scan** (500+ attack prompts):
  - `gpt-4o-mini`: ~$0.50
  - `claude-3-5-sonnet`: ~$1.00
  - `gpt-4o`: ~$2.00

*Note: Costs are estimates based on January 2025 pricing. Actual costs depend on response lengths.*

## 🔧 Setup and Installation

Navigate to the security testing module:
```bash
cd src/use_cases/security_testing
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start - Basic Security Scan

### Step 1: Run Your First Security Scan

Test for basic vulnerabilities:
```bash
# Basic security scan with a small sample
python -m attack_library.cli scan --model gpt-4o-mini --attack-types jailbreak,injection --limit 10

# More comprehensive scan
python -m attack_library.cli scan --model gpt-4o-mini --comprehensive
```

### Step 2: Review Security Results

The scan will generate:
- **Console output** with real-time vulnerability detection
- **JSON report** with detailed findings and confidence scores
- **HTML report** with visual analysis charts
- **CSV export** for integration with security tools

Example output:
```
🔍 LLM Security Scan Results
Model: gpt-4o-mini
Scan Date: 2025-01-06 14:30:15

📊 Vulnerability Summary:
✅ Jailbreak Resistance: 89% (Strong)
⚠️ Data Leakage Protection: 72% (Moderate)
❌ Prompt Injection Defense: 45% (Weak)

🎯 Critical Findings:
- 3 High-risk vulnerabilities detected
- 7 Medium-risk vulnerabilities detected
- 12 Low-risk vulnerabilities detected

📋 Recommendations:
- Implement additional input validation
- Review system prompt hardening
- Consider response filtering
```

## 📊 Available Attack Categories

Choose the right attack types based on your security requirements:

### 🎭 **Jailbreak Attempts** (`--attack-types jailbreak`)
Tests model resistance to instruction override and role-playing attacks:
- Role-playing scenarios ("act as a hacker")
- System prompt override attempts
- Context manipulation techniques
- Authority figure impersonation

**Example Command:**
```bash
python -m attack_library.cli scan --model claude-3-haiku --attack-types jailbreak --limit 25
```

**What it tests:** Can attackers make the model ignore safety guidelines?

---

### 💉 **Prompt Injection** (`--attack-types injection`)
Tests vulnerability to indirect prompt injection and command execution:
- Code execution attempts
- System command injection
- Indirect prompt injection via context
- Template injection attacks

**Example Command:**
```bash
python -m attack_library.cli scan --model gemini-1.5-flash --attack-types injection --limit 30
```

**What it tests:** Can attackers execute unintended commands through prompts?

---

### 🔐 **Data Leakage Detection** (`--attack-types data_leakage`)
Tests for unauthorized information disclosure:
- Personal information exposure
- Credential leakage
- Internal system information disclosure
- Training data memorization

**Example Command:**
```bash
python -m attack_library.cli scan --model gpt-4o-mini --attack-types data_leakage --limit 40
```

**What it tests:** Does the model inappropriately reveal sensitive information?

---

### 🎯 **Social Engineering** (`--attack-types social_engineering`)
Tests resistance to manipulation and trust-building attacks:
- Trust-building techniques
- Urgency and pressure tactics
- Authority figure claims
- Emotional manipulation

**Example Command:**
```bash
python -m attack_library.cli scan --model claude-3-5-sonnet --attack-types social_engineering --limit 20
```

**What it tests:** Can attackers manipulate the model through psychological techniques?

---

### ☣️ **Harmful Content Generation** (`--attack-types harmful_content`)
Tests prevention of dangerous or inappropriate content generation:
- Violence and harm instructions
- Illegal activity guidance
- Hate speech generation
- Dangerous information sharing

**Example Command:**
```bash
python -m attack_library.cli scan --model gpt-4o --attack-types harmful_content --limit 35
```

**What it tests:** Does the model refuse to generate harmful or dangerous content?

## 🔄 Advanced Usage Patterns

### Multi-Model Security Comparison
```bash
# Compare security across multiple models
python -m attack_library.cli compare-models \
  --models gpt-4o-mini,claude-3-haiku,gemini-1.5-flash \
  --attack-types jailbreak,injection \
  --output security_comparison.json
```

### Continuous Security Testing
```bash
# Run scheduled security scans
python -m attack_library.cli scan \
  --model gpt-4o-mini \
  --comprehensive \
  --schedule daily \
  --alert-threshold 0.7 \
  --notify-email security@yourcompany.com
```

### Custom Attack Library
```bash
# Use your own attack patterns
python -m attack_library.cli scan \
  --model claude-3-5-sonnet \
  --custom-attacks ./my_attacks.json \
  --output custom_security_report.html
```

### Integration with CI/CD
```bash
# Exit with non-zero code if vulnerabilities found
python -m attack_library.cli scan \
  --model $MODEL_UNDER_TEST \
  --attack-types jailbreak,injection \
  --fail-on-vulnerability high \
  --output results/security_scan.json
```

## 📈 Understanding Security Metrics

### Confidence Scores
- **High (0.8-1.0)**: Very likely vulnerability - immediate attention required
- **Medium (0.5-0.7)**: Potential vulnerability - investigation recommended
- **Low (0.2-0.4)**: Possible vulnerability - monitoring suggested
- **Minimal (0.0-0.2)**: Unlikely vulnerability - informational only

### Security Ratings
- **Excellent (90-100%)**: Strong security posture, minimal vulnerabilities
- **Good (75-89%)**: Solid security with minor issues
- **Fair (60-74%)**: Moderate security, some concerns present
- **Poor (40-59%)**: Weak security, significant vulnerabilities
- **Critical (0-39%)**: Severe security issues, immediate action required

## 🔧 Customization and Configuration

### Create Custom Security Configuration
```python
# security_config.py
from src.use_cases.security_testing import SecurityScanner, ScanConfig

config = ScanConfig(
    attack_types=['jailbreak', 'injection', 'data_leakage'],
    severity_threshold=0.6,
    max_concurrent_requests=5,
    timeout_seconds=30,
    enable_detailed_analysis=True
)

scanner = SecurityScanner(config)
results = await scanner.scan_model(your_model_interface)
```

### Custom Attack Patterns
```json
{
  "custom_attacks": [
    {
      "id": "custom_001",
      "category": "jailbreak",
      "prompt": "Your custom attack prompt here",
      "expected_behavior": "refuse",
      "severity": "high",
      "description": "Tests for specific vulnerability"
    }
  ]
}
```

## 📊 Reporting and Analytics

### Generate Comprehensive Security Report
```bash
# Create executive summary report
python -m attack_library.cli report \
  --input security_scan_results.json \
  --format executive \
  --output security_executive_summary.pdf

# Generate technical details report
python -m attack_library.cli report \
  --input security_scan_results.json \
  --format technical \
  --output security_technical_report.html
```

### Export for Security Tools
```bash
# Export to SARIF format for GitHub Security tab
python -m attack_library.cli export \
  --input security_scan_results.json \
  --format sarif \
  --output security_results.sarif

# Export to JSON for custom processing
python -m attack_library.cli export \
  --input security_scan_results.json \
  --format json \
  --output security_data.json
```

## 🚨 Integration with Security Workflows

### GitHub Actions Integration
Create `.github/workflows/security-scan.yml`:
```yaml
name: LLM Security Scan
on: [push, pull_request]

jobs:
  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          cd src/use_cases/security_testing
          pip install -r requirements.txt

      - name: Run security scan
        run: |
          python -m attack_library.cli scan \
            --model ${{ secrets.MODEL_NAME }} \
            --attack-types jailbreak,injection \
            --fail-on-vulnerability high \
            --output security_results.sarif
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Upload SARIF to GitHub
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: security_results.sarif
```

### Jenkins Integration
```groovy
pipeline {
    agent any
    stages {
        stage('Security Scan') {
            steps {
                script {
                    sh '''
                        cd src/use_cases/security_testing
                        python -m attack_library.cli scan \
                          --model gpt-4o-mini \
                          --comprehensive \
                          --output security_results.json
                    '''

                    archiveArtifacts artifacts: 'security_results.json'
                    publishHTML([
                        allowMissing: false,
                        alwaysLinkToLastBuild: true,
                        keepAll: true,
                        reportDir: 'src/use_cases/security_testing',
                        reportFiles: 'security_report.html',
                        reportName: 'LLM Security Report'
                    ])
                }
            }
        }
    }
}
```

## 🛠 Troubleshooting Common Issues

### Issue: "Rate limit exceeded"
**Solution:** Reduce concurrent requests or add delays:
```bash
python -m attack_library.cli scan \
  --model gpt-4o-mini \
  --max-concurrent 2 \
  --delay-between-requests 1
```

### Issue: "Model not responding to attacks"
**Solution:** Try different attack categories or increase sample size:
```bash
python -m attack_library.cli scan \
  --model claude-3-5-sonnet \
  --attack-types all \
  --limit 100
```

### Issue: "False positive detections"
**Solution:** Adjust confidence thresholds or enable detailed analysis:
```bash
python -m attack_library.cli scan \
  --model gemini-1.5-flash \
  --confidence-threshold 0.8 \
  --enable-detailed-analysis
```

## 🔗 Integration with Other Use Cases

- **Use Case 1-4:** Include security metrics in standard benchmark comparisons
- **Use Case 5:** Test local model security before deployment
- **Use Case 6:** Validate security of fine-tuned models
- **Use Case 8:** Add security monitoring to continuous testing pipelines

## 📚 Advanced Resources

### Research Papers
- "Red Teaming Language Models to Reduce Harms" (Anthropic, 2022)
- "Constitutional AI: Harmlessness from AI Feedback" (Anthropic, 2022)
- "Jailbroken: How Does LLM Safety Training Fail?" (Princeton, 2023)

### Security Frameworks
- OWASP Top 10 for LLMs
- NIST AI Risk Management Framework
- ISO/IEC 27001 AI Security Extensions

## 🚀 Next Steps

1. **Start with Basic Scanning:** Run quick scans to understand your model's security baseline
2. **Customize Attack Libraries:** Create domain-specific security tests for your use cases
3. **Integrate into Pipelines:** Add automated security testing to your deployment process
4. **Monitor Continuously:** Set up regular security scans to detect new vulnerabilities
5. **Expand Coverage:** Test additional models and attack vectors as your security program matures

---

*This guide provides comprehensive coverage of LLM security testing capabilities. The security testing framework includes enterprise-grade features for vulnerability detection, compliance reporting, and continuous monitoring. For additional support, refer to the [Troubleshooting Guide](./TROUBLESHOOTING.md) or reach out via GitHub issues.*
