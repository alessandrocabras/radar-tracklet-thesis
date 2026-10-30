# Publish the reviewed code-only update

Run from this repository, not from the parent workspace. No commit or push
is performed by the preparation workflow.

```powershell
python -m pytest -q
python tools/check_release.py
git status --short
git diff --check
git add README.md pyproject.toml docs notebooks src tests tools .gitignore .gitattributes
git diff --cached --stat
git diff --cached
git commit -m "Align thesis companion with controlled radar experiments"
git push origin main
```

Use the Python environment in which this package and test dependencies are
installed. Review the staged diff before committing. Never add private data,
split manifests, models, scalers, results or executed notebook outputs.
The allowlist is a safeguard, not a substitute for checking publication rights.
Do not force-push or replace remote history to resolve a conflict.

