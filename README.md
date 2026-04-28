# Safe model-based exploration

## Documentation

- Onboarding and codebase entry points: [`docs/onboarding.md`](docs/onboarding.md)
- OPAX theory to implementation (iCEM path): [`docs/opax_theory_to_implementation.md`](docs/opax_theory_to_implementation.md)

## Legacy Brax Import Workaround

If you hit `ModuleNotFoundError: No module named 'brax.v1'`, patch this installed file in your active virtualenv:

- Path: `.venv/lib/python3.11/site-packages/mbpo/optimizers/policy_optimizers/sac/acting.py`

Replace this line:

```python
from brax.v1 import envs as envs_v1
```

with:

```python
from brax import envs as envs_v1
```

Then start a fresh Python process and rerun the experiment.