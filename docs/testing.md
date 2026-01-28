# Testing

## Frameworks

- **Unit tests:** `pytest`
- **Integration tests:** `pytest` with `pytest-cov`
- **End-to-end (E2E) tests:** `pytest` marked with `@pytest.mark.e2e`

## Test structure

```
.
└── tests/
    ├── unit/          # Fast, isolated tests for pure functions/helpers
    ├── integration/   # Exercises env setup, Mujoco/Gym integration, etc.
    └── e2e/           # Full training loop smoke tests (optional)
```

## Running tests locally

Install dev dependencies (example):

```bash
python -m pip install -r requirements-dev.txt
```

Run all tests:

```bash
python -m pytest
```

Run a specific suite:

```bash
python -m pytest tests/unit
python -m pytest tests/integration
python -m pytest -m e2e tests/e2e
```

Collect coverage:

```bash
python -m pytest --cov=. --cov-report=term-missing
```

## Required suites and coverage

- **Required:** Unit tests must pass for every change.
- **Recommended:** Integration tests should pass before releases or when changing
  environments or training loops.
- **Optional:** E2E tests are marked with `@pytest.mark.e2e` and can be run on
  demand.
- **Coverage:** Aim for **≥ 70%** combined coverage from unit + integration
  suites (excluding E2E).
