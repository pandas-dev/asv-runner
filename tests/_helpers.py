from __future__ import annotations

import subprocess
from pathlib import Path


def init_remote_and_storage(tmp_path: Path) -> tuple[Path, Path]:
    """Create a bare 'remote' with an initial commit on `pandas_test`,
    then clone it into `storage` so tests can push/refetch.
    """
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    storage = tmp_path / "storage"

    subprocess.run(["git", "init", "--bare", str(remote)], check=True)

    seed.mkdir()
    subprocess.run(["git", "init", "-b", "pandas_test", str(seed)], check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=seed, check=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=seed, check=True)
    (seed / "data").mkdir()
    (seed / "data" / "shas.txt").write_text("")
    subprocess.run(["git", "add", "-A"], cwd=seed, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=seed, check=True)
    subprocess.run(["git", "push", str(remote), "pandas_test"], cwd=seed, check=True)

    subprocess.run(
        ["git", "clone", "--branch", "pandas_test", str(remote), str(storage)],
        check=True,
    )
    return remote, storage
