# CI/CD Guide for LLLM Lab

This guide explains the continuous integration and deployment setup for the LLLM Lab project.

## Overview

The CI/CD pipeline is designed to ensure code quality, run comprehensive tests, and provide detailed reporting across multiple Python versions and test categories.

## Workflows

### 1. Main Testing Workflow (`.github/workflows/test.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches

**Test Matrix:**
- **Python versions**: 3.8, 3.9, 3.10, 3.11
- **Test types**: unit, integration, performance, compatibility
- **Optimization**: Expensive tests (performance, compatibility) only run on Python 3.11

**Jobs:**

#### Test Job
- Runs the main test suite across the matrix
- Installs dependencies with caching
- Executes different test types based on matrix configuration
- Uploads test artifacts and coverage reports

#### Lint Job
- Code formatting check with Black
- Import sorting check with isort
- Linting with flake8
- Type checking with mypy (non-blocking)

#### Security Job
- Security linting with bandit
- Dependency vulnerability scanning with safety
- Uploads security reports

#### Test Summary Job
- Aggregates results from all jobs
- Provides summary in GitHub Actions summary

### 2. Coverage Workflow (`.github/workflows/coverage.yml`)

**Triggers:**
- Push to `main` branch
- Pull requests to `main` branch

**Features:**
- Comprehensive coverage analysis
- Coverage threshold enforcement (70%)
- PR coverage comments
- Codecov integration
- Coverage differential analysis for PRs

## Test Categories

### Unit Tests
- **Location**: `tests/unit*`, `tests/providers/`
- **Purpose**: Fast, isolated component testing
- **Coverage**: Requires 70% minimum coverage
- **Runtime**: ~2-5 minutes

### Integration Tests
- **Location**: `tests/integration/`
- **Purpose**: Test real API integrations
- **Requirements**: API keys (conditional execution)
- **Runtime**: ~5-15 minutes

### Performance Benchmarks
- **Location**: `tests/performance/`
- **Purpose**: Performance regression detection
- **Requirements**: API keys (runs in quick mode in CI)
- **Runtime**: ~10-30 minutes

### Compatibility Tests
- **Location**: `tests/compatibility/`
- **Purpose**: Cross-provider compatibility validation
- **Requirements**: API keys (conditional execution)
- **Runtime**: ~5-20 minutes

## Configuration Files

### `pyproject.toml`
Central configuration for:
- Package metadata
- Dependencies
- Tool configurations (black, isort, coverage, pytest, mypy)
- Build system setup

### `requirements-dev.txt`
Development dependencies including:
- Testing frameworks
- Code quality tools
- Performance analysis tools
- Documentation tools

### `.pre-commit-config.yaml`
Pre-commit hooks for:
- Code formatting
- Linting
- Security checks
- Documentation validation
- Custom project-specific checks

### `Makefile`
Development workflow automation:
- Testing commands
- Code quality checks
- Build and release tasks
- Benchmarking utilities

## API Key Management

The CI/CD pipeline handles API keys securely:

1. **Required secrets** (configure in GitHub repository settings):
   - `OPENAI_API_KEY` (optional)
   - `ANTHROPIC_API_KEY` (optional)
   - `GOOGLE_API_KEY` (optional)
   - `CODECOV_TOKEN` (optional, for coverage reporting)

2. **Conditional execution**:
   - Integration tests run only if relevant API keys are available
   - Performance benchmarks run in "quick mode" without API keys
   - Compatibility tests fall back to mocked tests without API keys

3. **Security**:
   - API keys are never logged or exposed
   - Tests gracefully handle missing keys
   - Security scanning prevents accidental key exposure

## Coverage Requirements

### Minimum Thresholds
- **Overall coverage**: 70%
- **New code coverage**: 80% (for PRs)
- **Critical modules**: 90% (core providers)

### Coverage Reports
- **HTML**: Detailed interactive report
- **XML**: For external tools (Codecov)
- **JSON**: For automated analysis
- **Terminal**: Quick CI feedback

### Coverage Analysis
- Line coverage and branch coverage
- Differential coverage for PRs
- Coverage trend tracking
- Uncovered line identification

## Artifacts and Reports

### Test Artifacts
- Coverage reports (HTML, XML, JSON)
- Performance benchmark results
- Compatibility test reports
- Security scan results

### Retention Policy
- Test reports: 7 days
- Coverage reports: 30 days
- Security reports: 30 days
- Performance data: 14 days

## Development Workflow

### Local Development
```bash
# Setup development environment
make dev-setup

# Run pre-commit checks
make pre-commit-run

# Quick development check
make quick-check

# Full local testing
make check-all
```

### Pre-commit Hooks
Automatically run on commit:
- Code formatting (black, isort)
- Linting (flake8, bandit)
- Type checking (mypy)
- Security checks
- Custom validations

### Pull Request Workflow
1. **Create feature branch**
2. **Develop with local testing**
3. **Create pull request**
4. **Automated CI/CD runs**:
   - Unit tests on all Python versions
   - Integration tests (if API keys available)
   - Linting and security checks
   - Coverage analysis
5. **Review coverage report and test results**
6. **Merge after approval**

## Performance Monitoring

### Benchmark Tracking
- Response time trends
- Throughput measurements
- Memory usage patterns
- Success rate monitoring

### Performance Regression Detection
- Comparison with baseline metrics
- Automated alerts for significant changes
- Historical performance data

### Resource Usage
- CI job duration monitoring
- Test execution time tracking
- Resource consumption analysis

## Troubleshooting

### Common Issues

#### 1. Test Failures
```bash
# Check specific test category
pytest tests/unit* -v --tb=long

# Run with debugging
pytest tests/unit* -v -s --pdb
```

#### 2. Coverage Issues
```bash
# Generate detailed coverage report
make coverage

# Check missing coverage
coverage report --show-missing
```

#### 3. API Key Issues
```bash
# Check environment setup
make check-env

# Test with mocked APIs
pytest tests/ -k "not real_api"
```

#### 4. Linting Failures
```bash
# Fix formatting
make format

# Check specific linting issues
flake8 llm_providers tests --show-source
```

### CI/CD Debugging

#### 1. Workflow Failures
- Check GitHub Actions logs
- Review artifact uploads
- Verify environment variables
- Check dependency installation

#### 2. Coverage Issues
- Review coverage differential
- Check excluded files
- Verify test execution
- Review coverage configuration

#### 3. Performance Issues
- Monitor CI job duration
- Check resource usage
- Review test parallelization
- Optimize test selection

## Best Practices

### Code Quality
1. **Write tests first** (TDD approach)
2. **Maintain high coverage** (>80% for new code)
3. **Use type hints** consistently
4. **Follow coding standards** (enforced by pre-commit)
5. **Document complex logic**

### Testing Strategy
1. **Fast unit tests** for core logic
2. **Integration tests** for API interactions
3. **Performance tests** for regression detection
4. **Compatibility tests** for cross-provider validation

### CI/CD Optimization
1. **Use caching** for dependencies
2. **Parallel test execution** where possible
3. **Conditional execution** based on changes
4. **Artifact management** for debugging

### Security
1. **Never commit API keys**
2. **Use environment variables** for secrets
3. **Regular security scanning**
4. **Dependency vulnerability monitoring**

## Monitoring and Alerts

### GitHub Actions
- Workflow status notifications
- PR check status
- Scheduled workflow monitoring

### External Services
- **Codecov**: Coverage tracking and alerts
- **Safety**: Dependency vulnerability alerts
- **GitHub Security**: Dependabot alerts

### Custom Monitoring
- Performance trend analysis
- Test execution time tracking
- Coverage trend monitoring
- API usage tracking

## Future Enhancements

### Planned Improvements
1. **Automated benchmarking** on schedule
2. **Performance regression alerts**
3. **Multi-environment testing**
4. **Advanced security scanning**
5. **Documentation auto-generation**

### Integration Opportunities
1. **Slack/Discord notifications**
2. **JIRA integration** for issue tracking
3. **Advanced analytics** dashboards
4. **Load testing** integration
5. **Deployment automation**

---

This CI/CD setup ensures high-quality, well-tested code while providing comprehensive feedback to developers and maintaining project health metrics.
