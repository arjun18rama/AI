# Repository Guidelines (Root)

These instructions apply to the entire repository. Subdirectories may include their own `AGENTS.md` files; those more-specific instructions override this document for files in their scope.

## Code style expectations
- Follow existing patterns in the codebase and keep changes consistent with nearby files.
- Prefer type hints for public functions and keep signatures explicit.
- Use `pathlib.Path` for filesystem paths and context managers for file I/O.
- Keep imports ordered (standard library, third-party, local) and avoid unused imports.
- Formatting: follow standard Python conventions (PEP 8). Do not add try/catch blocks around imports.

## Naming conventions
- `snake_case` for functions/variables, `PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants.
- Use descriptive names for configuration keys and keep YAML config keys stable once established.

## Checks/tests (local)
- No automated test suite is currently configured.
- If you add tests, document how to run them in this section and ensure they pass before committing.

## PR format requirements
- Title: short, imperative summary (e.g., "Add training config validation").
- Body:
  - **Summary**: 1–3 bullet points describing changes.
  - **Testing**: list commands run, or state "Not run (not configured)" when applicable.

## Commit message conventions
- Use imperative, present-tense subject lines (e.g., "Add opponent snapshot rotation").
- Keep the subject under 72 characters; add a body if the change is non-trivial.

## Architectural constraints & patterns
- `train.py` is the training entry point; keep training orchestration there.
- `configs/` contains YAML configuration used for training runs; prefer adding new config files over hardcoding values.
- `envs/` contains environment implementations; keep Gym-compatible API boundaries inside this package.
- `selfplay/` contains self-play utilities such as opponent pools; avoid coupling it directly to training CLI parsing.
