"""Conservative release check. Does not publish, stage, modify or delete files."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    ".gitignore", ".gitattributes", "README.md", "pyproject.toml",
    "docs/CHAPTER_ALIGNMENT.md", "docs/DATA_INTERFACE.md", "docs/PUBLISH_IT.md",
    "docs/SOURCE_PROVENANCE.json", "docs/LOCAL_VALIDATION.md",
    "src/radar_tracklets/__init__.py", "src/radar_tracklets/preprocessing.py",
    "src/radar_tracklets/streaming.py", "src/radar_tracklets/models.py",
    "src/radar_tracklets/workflows.py", "src/radar_tracklets/evaluation.py",
    "src/radar_tracklets/training.py",
    "src/radar_tracklets/geometry.py", "src/radar_tracklets/reporting.py",
    "src/radar_tracklets/analysis.py", "src/radar_tracklets/experiments.py",
    "src/radar_tracklets/__main__.py",
    "notebooks/elevation.ipynb", "notebooks/range.ipynb", "notebooks/azimuth.ipynb",
    "tests/test_protocol.py", "tests/test_models.py", "tools/check_release.py",
    "tests/test_followups.py",
}


def main():
    errors = []
    for relative in sorted(ALLOWED):
        path = ROOT / relative
        if not path.is_file():
            errors.append(f"Missing allowlisted file: {relative}")
            continue
        content = path.read_text(encoding="utf-8")
        for marker in ("C:\\Users\\", "C:/Users/", "BEGIN PRIVATE KEY", "ghp_", "github_pat_"):
            if marker in content and relative != "tools/check_release.py":
                errors.append(f"Potential private content in {relative}")
        if path.suffix == ".py":
            ast.parse(content, filename=relative)
        if path.suffix == ".ipynb":
            notebook = json.loads(content)
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] == "code":
                    if cell.get("outputs") or cell.get("execution_count") is not None:
                        errors.append(f"Executed/private output in {relative} cell {index}")
                    ast.parse("".join(cell["source"]), filename=f"{relative}:{index}")
                if cell.get("attachments"):
                    errors.append(f"Embedded attachment in {relative}")
    # Check every non-ignored candidate, including untracked files. This is an
    # allowlist, not just an extension denylist. Git metadata is never scanned.
    if (ROOT / ".git").exists():
        candidates = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT
        ).decode().split("\0")
        for name in filter(None, candidates):
            if name not in ALLOWED:
                errors.append(f"Unexpected publishable file: {name}")
    else:
        errors.append("No independent .git directory; initialize inside this repository first")
    if errors:
        print("RELEASE CHECK FAILED\n" + "\n".join(errors))
        return 1
    print(f"PASS: {len(ALLOWED)} allowlisted files; Python/notebook syntax and clean outputs checked.")
    print("No dataset/simulator disclosure authorized by this check. Review staged diff before push.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

