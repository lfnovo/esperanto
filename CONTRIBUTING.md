# Contributing to Esperanto

First off, thank you for considering contributing to Esperanto! It's people like you that make Esperanto such a great tool.

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the issue list as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

* Use a clear and descriptive title
* Describe the exact steps which reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include any error messages

### Suggesting Enhancements

If you have a suggestion for a new feature or enhancement, first check the issue list to see if it's already been proposed. If it hasn't, you can create a new issue. Please provide:

* A clear and descriptive title
* A detailed description of the proposed feature
* An explanation of why this enhancement would be useful
* Examples of how the feature would be used

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run the tests (`pytest`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

#### Pull Request Guidelines

* Follow the existing code style
* Write tests for new features
* Update documentation for any changes
* Keep commits focused and atomic
* Use clear commit messages

## Development Setup

1. Clone the repository
```bash
git clone https://github.com/lfnovo/esperanto.git
cd esperanto
```

2. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,all]"
```

3. Run tests:
```bash
pytest
```

4. Run linting:
```bash
ruff check .
mypy .
```

## Code Style and Linting

We use `ruff` for code linting and formatting. Before submitting a PR, make sure to:

1. Run the linter to check for issues:
```bash
ruff check .
```

2. Fix auto-fixable issues:
```bash
ruff check . --fix
```

The project's ruff configuration is in `pyproject.toml` and enforces:
- Line length of 88 characters
- Standard Python style rules (E, F)
- Import sorting (I)

## Testing

* Write tests for any new features
* Ensure all tests pass before submitting a PR
* Run the full test suite with `pytest`

## Documentation

* Update the README.md if needed
* Add docstrings for new functions and classes
* Keep documentation clear and concise

## Questions?

Feel free to open an issue with your question. We'll do our best to help!
