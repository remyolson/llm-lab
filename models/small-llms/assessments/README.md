# System Assessment Results

This directory contains system assessment results from running `system_assessment.py`.

## Files

- `system_assessment_YYYYMMDD_HHMMSS.json` - Detailed assessment results with timestamp
- Each file contains:
  - System information (RAM, CPU, GPU, disk space)
  - Compatibility assessment for GPT-OSS 20B
  - Recommendations and preparation steps
  - Process analysis results

## Usage

Run a new assessment:
```bash
python system_assessment.py
```

View assessment history:
```bash
ls -la assessments/
```

## Understanding Results

- **can_run: true/false** - Whether your system can handle GPT-OSS 20B
- **confidence: high/medium/low** - Reliability assessment
- **issues[]** - Critical problems that prevent running
- **warnings[]** - Non-critical concerns
- **recommendations[]** - Suggested improvements

## Key Metrics

- **Minimum RAM**: 24GB for GPT-OSS 20B
- **Recommended RAM**: 32GB for reliable operation
- **Disk space**: 50GB minimum for model files and swap
- **CPU cores**: 4+ recommended

## Apple Silicon Notes

- Uses unified memory (shared CPU/GPU)
- Metal Performance Shaders acceleration available
- macOS manages swap automatically
- Temperature monitoring via thermal sensors
