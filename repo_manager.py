from pathlib import Path
import subprocess

class RepoManager:
    def __init__(self, base_dir: str = "repositories"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def prepare_repo(self, owner: str, repo: str, commit_sha: str) -> Path:
        repo_dir = self.base_dir / f"{owner}__{repo}"
        repo_url = f"https://github.com/{owner}/{repo}.git"
        if not repo_dir.exists():
            print(f"Cloning {owner}/{repo}...")
            self._run([
                    "git",
                    "clone",
                    "--no-checkout",
                    repo_url,
                    str(repo_dir),
                ]
            )
        else:
            print(f"Using existing clone: {repo_dir}")
        print("Fetching repository refs...")
        self._run(
            ["git", "fetch", "--all", "--tags", "--prune"],
            cwd=repo_dir,
        )
        print(f"Checking out commit: {commit_sha}")
        self._run(
            ["git", "checkout", "--force", commit_sha],
            cwd=repo_dir,
        )
        return repo_dir

    @staticmethod
    def _run(command: list[str], cwd: Path | None = None) -> None:
        subprocess.run(
            command,
            cwd=cwd,
            check=True,
        )
