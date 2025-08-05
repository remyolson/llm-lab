# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of LLM Lab seriously. If you have discovered a security vulnerability, please follow these steps:

1. **DO NOT** open a public issue
2. Email your findings to [INSERT SECURITY EMAIL]
3. Include the following information:
   - Type of vulnerability
   - Full paths of source file(s) related to the vulnerability
   - Location of the affected source code (tag/branch/commit or direct URL)
   - Step-by-step instructions to reproduce the issue
   - Proof-of-concept or exploit code (if possible)
   - Impact of the issue

## What to Expect

- Acknowledgment of your report within 48 hours
- A more detailed response within 7 days
- Regular updates on the progress
- Credit in the release notes (unless you prefer to remain anonymous)

## Security Best Practices for Users

1. **API Keys**: Never commit API keys to the repository
2. **Dependencies**: Keep all dependencies up to date
3. **Configuration**: Use environment variables for sensitive configuration
4. **Access Control**: Limit API key permissions to minimum required
5. **Monitoring**: Enable logging and monitoring for suspicious activity

## Disclosure Policy

- Security issues will be disclosed after a fix is available
- We will coordinate disclosure with affected parties
- CVEs will be requested for significant vulnerabilities
