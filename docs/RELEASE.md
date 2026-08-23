# Publishing grpo-guard to PyPI

Not yet published (2026-08-23, user decision).  Everything below is
verified locally; publishing needs a PyPI API token.

## Build

```bash
uv build                      # dist/grpo_guard-<ver>-py3-none-any.whl + .tar.gz
```

## Verify the wheel before publishing

```bash
python -m venv /tmp/gg-test && /tmp/gg-test/bin/pip install dist/grpo_guard-*.whl
/tmp/gg-test/bin/grpo-guard contract-check --cases tests/frozen/f1_f4_v01 --out /tmp/cc
# expect: 24/24 passed, exit 0
```

## Publish (requires a PyPI API token)

1. Create a token at pypi.org → Account settings → API tokens
   (scope: the `grpo-guard` project).
2. `export UV_PUBLISH_TOKEN=pypi-...` (never commit it).
3. `uv publish` (defaults to PyPI; add `--publish-url
   https://upload.pypi.org/legacy/` explicitly if needed).
4. For a dry run first: TestPyPI via
   `uv publish --publish-url https://test.pypi.org/legacy/`.

## Notes

- Package name: `grpo-guard` (module `grpo_guard`) — name was free on
  PyPI at the time of writing (both `grpo-guard` and `grpo_guard`
  returned 404).
- `dist/` is gitignored; wheels are reproducible from any tagged commit.
- The package has CPU-only runtime deps (pydantic, numpy, pyyaml); the
  GPU stack is optional (`[gpu]` extra) per `compatibility_profile.yaml`.
