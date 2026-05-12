# Contributing to HierSolv

We welcome contributions! Here's how to contribute:

## Setup for Development

```bash
git clone https://github.com/yourusername/hiersolv.git
cd hiersolv
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v --cov=hiersolv
```

## Code Style

We use `black` and `flake8`:

```bash
black .
flake8 . --max-line-length=100
```

## Pull Requests

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Commit your changes with clear messages
3. Push to your fork
4. Open a PR with a clear description

## Issues

Please report bugs and request features via GitHub Issues with:
- Clear description
- Minimal reproducible example (for bugs)
- Your environment (Python version, PyTorch version, etc.)
