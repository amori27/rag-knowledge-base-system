# Contributing to RAG Knowledge Base System

We love contributions! Here's how you can help.

## Getting Started

1. Fork the repository.
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/rag-knowledge-base-system.git
   ```
3. Create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   ```

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Code Style

- Follow PEP 8.
- Use type hints for all public APIs.
- Write docstrings in Google style.
- Keep lines under 88 characters.

## Testing

Run tests with:

```bash
python -m pytest
```

Make sure all tests pass before submitting your PR.

## Pull Request Process

1. Update `CHANGELOG.md` with your changes.
2. Ensure CI passes.
3. Request review from a maintainer.
4. Squash commits on merge.

## Reporting Issues

Use the [issue templates](.github/ISSUE_TEMPLATE/) — one for bugs, one for features.

## Code of Conduct

All participants must follow our [Code of Conduct](CODE_OF_CONDUCT.md).
