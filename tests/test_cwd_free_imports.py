import subprocess
import sys


def test_torch_utils_legacy_imports_from_foreign_cwd(tmp_path):
    """The pkl loader's import graph must not depend on the repo-root CWD."""
    result = subprocess.run(
        [sys.executable, "-c", "import torch_utils.legacy"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
