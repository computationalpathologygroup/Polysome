# Contributing to Polysome

Thank you for your interest in contributing to Polysome! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Your environment (OS, Python version, package version)
- Any relevant logs or error messages

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:
- A clear description of the feature
- Use cases and motivation
- Possible implementation approach (optional)

### Submitting Pull Requests

1. **Fork the repository** and create a new branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following these guidelines:
   - Follow the existing code style
   - Add tests for new functionality
   - Update documentation as needed
   - Keep commits focused and atomic

3. **Test your changes** locally:
   ```bash
   # Run tests
   pytest

   # Check code formatting
   black --check src/

   # Type checking (if applicable)
   mypy src/
   ```

4. **Commit your changes** with clear, descriptive commit messages:
   ```bash
   git commit -m "Add feature: brief description"
   ```

5. **Push to your fork** and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Describe your changes** in the pull request:
   - What problem does it solve?
   - How does it work?
   - Any breaking changes?
   - Related issues (if any)

## Development Setup

### Installation for Development

```bash
# Clone the repository
git clone https://github.com/computationalpathologygroup/Polysome.git
cd Polysome

# Install in development mode
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=polysome

# Run specific test file
pytest tests/test_specific.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Write docstrings for public functions and classes
- Keep functions focused and modular

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion on GitHub
- Reach out to the maintainers

Thank you for contributing to Polysome!
