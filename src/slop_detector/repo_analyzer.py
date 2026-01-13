"""Repository cloning and file enumeration."""

import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional

import chardet
from git import Repo
from git.exc import GitCommandError, InvalidGitRepositoryError
from pathspec import PathSpec


# Default patterns to ignore
DEFAULT_IGNORE_PATTERNS = [
    # Dependencies
    "node_modules/",
    "vendor/",
    "venv/",
    ".venv/",
    "__pycache__/",
    "*.pyc",
    ".tox/",
    "bower_components/",

    # Build outputs
    "dist/",
    "build/",
    "target/",
    "out/",
    "*.egg-info/",

    # IDE/Editor
    ".idea/",
    ".vscode/",
    "*.swp",
    "*.swo",
    ".project",
    ".classpath",

    # Binary and media
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.gif",
    "*.ico",
    "*.svg",
    "*.pdf",
    "*.zip",
    "*.tar",
    "*.gz",
    "*.woff",
    "*.woff2",
    "*.ttf",
    "*.eot",
    "*.mp3",
    "*.mp4",
    "*.webm",
    "*.mov",

    # Lock files (generated)
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Pipfile.lock",
    "poetry.lock",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
    "go.sum",

    # Misc
    ".git/",
    ".DS_Store",
    "Thumbs.db",
    "*.log",
    "*.min.js",
    "*.min.css",
    "*.map",
    "coverage/",
    ".coverage",
    ".nyc_output/",
]


class RepoAnalyzer:
    """Analyze a Git repository."""

    def __init__(
        self,
        ignore_patterns: Optional[list[str]] = None,
        max_file_size: int = 1024 * 1024,  # 1MB default
    ):
        self.ignore_patterns = ignore_patterns or DEFAULT_IGNORE_PATTERNS
        self.max_file_size = max_file_size
        self._temp_dir: Optional[str] = None
        self._repo: Optional[Repo] = None
        self._repo_path: Optional[Path] = None

    def clone_repo(self, url: str, depth: Optional[int] = None) -> Path:
        """Clone a repository and return the path."""
        self._temp_dir = tempfile.mkdtemp(prefix="slop-detector-")

        try:
            clone_kwargs = {"url": url, "to_path": self._temp_dir}
            if depth:
                clone_kwargs["depth"] = depth

            self._repo = Repo.clone_from(**clone_kwargs)
            self._repo_path = Path(self._temp_dir)
            return self._repo_path
        except GitCommandError as e:
            self.cleanup()
            raise RuntimeError(f"Failed to clone repository: {e}")

    def open_local_repo(self, path: str | Path) -> Path:
        """Open a local repository."""
        path = Path(path)
        if not path.exists():
            raise ValueError(f"Path does not exist: {path}")

        try:
            self._repo = Repo(path)
            self._repo_path = path
            return self._repo_path
        except InvalidGitRepositoryError:
            # Not a git repo, but we can still analyze files
            self._repo = None
            self._repo_path = path
            return self._repo_path

    def cleanup(self):
        """Clean up temporary directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None

    def get_repo_name(self, url: str) -> str:
        """Extract repository name from URL."""
        # Handle various URL formats
        patterns = [
            r"github\.com[:/]([^/]+/[^/]+?)(?:\.git)?$",
            r"gitlab\.com[:/]([^/]+/[^/]+?)(?:\.git)?$",
            r"bitbucket\.org[:/]([^/]+/[^/]+?)(?:\.git)?$",
            r"/([^/]+)/?$",  # Fallback: last path component
        ]

        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1).rstrip("/")

        return url

    def iter_files(self) -> Generator[Path, None, None]:
        """Iterate over all analyzable files in the repository."""
        if not self._repo_path:
            raise RuntimeError("No repository loaded")

        pathspec = PathSpec.from_lines("gitwildmatch", self.ignore_patterns)

        for root, dirs, files in os.walk(self._repo_path):
            root_path = Path(root)
            rel_root = root_path.relative_to(self._repo_path)

            # Filter directories in-place
            # Allow AI marker directories even if they start with .
            ai_marker_dirs = {".claude", ".cursor", ".aider", ".codeium", ".ai", ".copilot", ".github"}
            dirs[:] = [
                d for d in dirs
                if not pathspec.match_file(str(rel_root / d) + "/")
                and (not d.startswith(".") or d.lower() in ai_marker_dirs)
            ]

            for filename in files:
                file_path = root_path / filename
                rel_path = file_path.relative_to(self._repo_path)

                # Skip ignored files
                if pathspec.match_file(str(rel_path)):
                    continue

                # Skip files that are too large
                try:
                    if file_path.stat().st_size > self.max_file_size:
                        continue
                except OSError:
                    continue

                # Skip binary files
                if self._is_binary(file_path):
                    continue

                yield rel_path

    def read_file(self, rel_path: Path) -> Optional[str]:
        """Read file content, handling encoding."""
        if not self._repo_path:
            raise RuntimeError("No repository loaded")

        file_path = self._repo_path / rel_path
        try:
            # Try UTF-8 first
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                pass

            # Detect encoding
            with open(file_path, "rb") as f:
                raw = f.read()
                detected = chardet.detect(raw)
                encoding = detected.get("encoding", "utf-8")

            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                return f.read()

        except Exception:
            return None

    def _is_binary(self, path: Path) -> bool:
        """Check if a file is binary."""
        try:
            with open(path, "rb") as f:
                chunk = f.read(8192)
                # Check for null bytes
                if b"\x00" in chunk:
                    return True
                # Check ratio of non-text bytes
                text_chars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)) - {0x7f})
                non_text = sum(1 for b in chunk if b not in text_chars)
                return non_text / len(chunk) > 0.3 if chunk else False
        except Exception:
            return True

    def get_commit_history(self, max_commits: int = 100) -> list[dict]:
        """Get recent commit history."""
        if not self._repo:
            return []

        commits = []
        try:
            for commit in self._repo.iter_commits(max_count=max_commits):
                commits.append({
                    "sha": commit.hexsha,
                    "message": commit.message.strip(),
                    "author": str(commit.author),
                    "author_email": commit.author.email,
                    "date": datetime.fromtimestamp(commit.committed_date),
                    "files_changed": len(commit.stats.files),
                    "insertions": commit.stats.total["insertions"],
                    "deletions": commit.stats.total["deletions"],
                })
        except Exception:
            pass

        return commits

    def get_file_history(self, rel_path: Path, max_commits: int = 10) -> list[dict]:
        """Get commit history for a specific file."""
        if not self._repo:
            return []

        commits = []
        try:
            for commit in self._repo.iter_commits(paths=str(rel_path), max_count=max_commits):
                commits.append({
                    "sha": commit.hexsha[:8],
                    "message": commit.message.strip().split("\n")[0],
                    "author": str(commit.author),
                    "date": datetime.fromtimestamp(commit.committed_date),
                })
        except Exception:
            pass

        return commits

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
