# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Polysome, please report it responsibly:

### How to Report

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, please report security issues via email to:
- **sander.moonemans@radboudumc.nl**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### What to Expect

- **Acknowledgment**: We will acknowledge receipt of your vulnerability report within 48 hours
- **Updates**: We will provide regular updates on the status of the issue
- **Fix Timeline**: We aim to release a fix within 30 days for critical issues
- **Credit**: With your permission, we will credit you in the fix announcement

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Security Considerations

### When Using Polysome

- **Model Files**: Only load models from trusted sources
- **User Input**: Validate and sanitize any user-provided data before processing
- **Environment Variables**: Keep sensitive environment variables (API keys, paths) secure
- **Docker**: Use appropriate security contexts when running Docker containers
- **Data Privacy**: Be mindful of data privacy when processing sensitive information with LLMs

### Best Practices

1. **Keep Dependencies Updated**: Regularly update Polysome and its dependencies
2. **Access Control**: Limit file system access to necessary directories only
3. **Resource Limits**: Set appropriate resource limits when running workflows
4. **Audit Logs**: Review logs for any suspicious activity
5. **Network Security**: When using remote models or APIs, use secure connections

## Known Limitations

- This framework executes user-provided workflow configurations - ensure workflows are from trusted sources
- LLM outputs should be validated before use in production systems
- File system operations require appropriate permissions

## Security Updates

Security updates will be published as new releases. Subscribe to repository notifications to stay informed.
