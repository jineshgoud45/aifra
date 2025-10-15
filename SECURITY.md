# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them to our security team:

### Preferred Method

Email: **23eg109a16@anurag.edu.in**
                (OR)
       **24eg509a01@anurag.edu.in**         

### What to Include

Please include the following information:

- **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- **Full paths** of source file(s) related to the issue
- **Location** of the affected source code (tag/branch/commit or direct URL)
- **Step-by-step instructions** to reproduce the issue
- **Proof-of-concept** or exploit code (if possible)
- **Impact** of the issue, including how an attacker might exploit it

### Response Timeline

- **Acknowledgment**: Within 48 hours
- **Initial assessment**: Within 7 days
- **Status update**: Every 2 weeks
- **Fix timeline**: Depends on severity (see below)

### Severity Levels

| Severity | Response Time | Example |
|----------|---------------|---------|
| **Critical** | 24-48 hours | Remote code execution, data breach |
| **High** | 1 week | Authentication bypass, privilege escalation |
| **Medium** | 2-4 weeks | XSS, CSRF, information disclosure |
| **Low** | 4-8 weeks | Minor information leaks, rate limiting issues |

## Security Best Practices

When using this software:

### 1. Authentication & Secrets

```bash
# NEVER commit secrets
# Use environment variables
export FRA_AUTH_SECRET=$(openssl rand -hex 32)
export FRA_API_KEY="your-secret-key"
```

### 2. Input Validation

- All file uploads are validated (type, size, content)
- Path traversal protection is enabled
- SQL injection protection via parameterized queries

### 3. Rate Limiting

```python
# Enable rate limiting in production
FRA_ENABLE_RATE_LIMIT=true
FRA_REQUESTS_PER_MINUTE=60
```

### 4. Docker Security

```bash
# Run as non-root user (already configured)
docker run --user 1000:1000 fra-diagnostics

# Use read-only filesystem where possible
docker run --read-only fra-diagnostics

# Limit resources
docker run --memory=4g --cpus=2 fra-diagnostics
```

### 5. Network Security

```bash
# Use HTTPS in production
# Configure reverse proxy (nginx/traefik)
# Enable firewall rules
# Use VPN for sensitive data
```

## Known Security Features

### Implemented Protections

- ✅ **Path Traversal Protection**: Multi-layer filename sanitization
- ✅ **XXE Attack Prevention**: Using defusedxml instead of standard XML parser
- ✅ **Rate Limiting**: IP-based DoS protection
- ✅ **Input Validation**: Comprehensive bounds checking
- ✅ **Thread Safety**: Mutex locks for concurrent operations
- ✅ **SQL Injection**: N/A - No SQL database used
- ✅ **XSS Protection**: Streamlit handles output escaping
- ✅ **CSRF**: Streamlit session management

### Security Scanning

We use automated security scanning:

```bash
# Security audit
bandit -r . -f json -o bandit-report.json

# Dependency vulnerabilities
safety check --json

# Package audit
pip-audit
```

## Vulnerability Disclosure Policy

1. **Report received**: We acknowledge within 48 hours
2. **Assessment**: We investigate and assess severity
3. **Fix development**: We develop and test a fix
4. **Coordinated disclosure**: We work with you on timing
5. **Public disclosure**: After fix is deployed (typically 90 days)
6. **Credit**: We credit reporters (unless anonymous requested)

## Security Updates

Security updates are released as:

- **Patch versions** (1.0.x) for security fixes
- **Security advisories** published on GitHub
- **CHANGELOG.md** updated with security notes

Subscribe to:
- GitHub Security Advisories
- Release notifications
- Security mailing list (23eg109a16@anurag.edu.in) OR (24eg509a01@anurag.edu.in)

## Bug Bounty

We currently **do not** have a formal bug bounty program, but we:

- Publicly acknowledge security researchers
- Provide credit in security advisories
- Consider future rewards program

## Compliance

This project aims to comply with:

- **OWASP Top 10**: Web application security risks
- **CWE Top 25**: Most dangerous software weaknesses
- **IEC 60076-18**: Domain-specific standards

## Contact

- **Security issues**: 23eg109a16@anurag.edu.in OR 24eg509a01@anurag.edu.in
- **General questions**: 23eg109a16@anurag.edu.in OR 24eg509a01@anurag.edu.in
- **GitHub Security Advisory**: Use GitHub's private reporting

---

**Thank you for helping keep FRA Diagnostics Platform secure!** 🔒
