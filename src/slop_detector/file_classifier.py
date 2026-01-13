"""Classify files by content type."""

from pathlib import Path
from typing import Optional

from .models import ContentType


# File extension to content type mappings
CODE_EXTENSIONS = {
    # Python
    ".py", ".pyx", ".pxd", ".pyi",
    # JavaScript/TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    # Systems
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hxx",
    # JVM
    ".java", ".kt", ".kts", ".scala", ".groovy",
    # .NET
    ".cs", ".fs", ".vb",
    # Go
    ".go",
    # Rust
    ".rs",
    # Ruby
    ".rb", ".rake",
    # PHP
    ".php",
    # Swift/Objective-C
    ".swift", ".m", ".mm",
    # Shell
    ".sh", ".bash", ".zsh", ".fish",
    # Other
    ".lua", ".pl", ".pm", ".r", ".R", ".jl", ".ex", ".exs",
    ".elm", ".clj", ".cljs", ".erl", ".hrl", ".hs", ".lhs",
    ".ml", ".mli", ".nim", ".cr", ".v", ".zig", ".d",
}

DOC_EXTENSIONS = {
    ".md", ".markdown", ".rst", ".txt", ".adoc", ".asciidoc",
    ".tex", ".org", ".wiki", ".rdoc",
}

CONFIG_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".xml", ".properties", ".env", ".env.example", ".env.local",
    ".editorconfig", ".prettierrc", ".eslintrc", ".babelrc",
}

DATA_EXTENSIONS = {
    ".csv", ".tsv", ".sql", ".sqlite", ".db",
}

# Filename patterns for documentation
DOC_FILENAMES = {
    "readme", "readme.md", "readme.txt", "readme.rst",
    "changelog", "changelog.md", "changes", "history",
    "contributing", "contributing.md", "contributors",
    "license", "license.md", "license.txt",
    "authors", "authors.md", "credits",
    "code_of_conduct", "code_of_conduct.md",
    "security", "security.md",
    "support", "support.md",
    "todo", "todo.md", "todo.txt",
    "faq", "faq.md",
}

# Filename patterns for configuration
CONFIG_FILENAMES = {
    "package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
    "tsconfig.json", "jsconfig.json",
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "pipfile", "pipfile.lock", "poetry.lock",
    "cargo.toml", "cargo.lock",
    "gemfile", "gemfile.lock",
    "go.mod", "go.sum",
    "composer.json", "composer.lock",
    "makefile", "cmakelists.txt",
    "dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".gitignore", ".gitattributes", ".gitmodules",
    ".dockerignore", ".npmignore",
    ".eslintrc.js", ".eslintrc.json", ".eslintrc.yml",
    ".prettierrc", ".prettierrc.js", ".prettierrc.json",
    "webpack.config.js", "vite.config.js", "rollup.config.js",
    "jest.config.js", "vitest.config.js",
    ".github/workflows/*.yml", ".github/workflows/*.yaml",
    "terraform.tfvars", "variables.tf", "main.tf",
}


class FileClassifier:
    """Classify files by their content type."""

    def __init__(self):
        self.code_extensions = CODE_EXTENSIONS
        self.doc_extensions = DOC_EXTENSIONS
        self.config_extensions = CONFIG_EXTENSIONS
        self.data_extensions = DATA_EXTENSIONS
        self.doc_filenames = DOC_FILENAMES
        self.config_filenames = CONFIG_FILENAMES

    def classify(self, path: Path) -> ContentType:
        """Classify a file by its path."""
        filename = path.name.lower()
        extension = path.suffix.lower()

        # Check filename patterns first
        if filename in self.doc_filenames:
            return ContentType.DOCUMENTATION

        if filename in self.config_filenames:
            return ContentType.CONFIGURATION

        # Check for docs directory
        path_parts = [p.lower() for p in path.parts]
        if any(p in {"docs", "doc", "documentation", "wiki"} for p in path_parts):
            if extension in self.doc_extensions or extension in {".md", ".rst", ".txt"}:
                return ContentType.DOCUMENTATION

        # Check by extension
        if extension in self.code_extensions:
            return ContentType.CODE

        if extension in self.doc_extensions:
            return ContentType.DOCUMENTATION

        if extension in self.config_extensions:
            return ContentType.CONFIGURATION

        if extension in self.data_extensions:
            return ContentType.DATA

        # Special cases
        if self._is_likely_config(path):
            return ContentType.CONFIGURATION

        if self._is_likely_doc(path):
            return ContentType.DOCUMENTATION

        return ContentType.UNKNOWN

    def _is_likely_config(self, path: Path) -> bool:
        """Check if file is likely a configuration file."""
        filename = path.name.lower()

        # Dotfiles are usually config
        if filename.startswith(".") and not filename.startswith(".."):
            return True

        # Files in certain directories
        path_parts = [p.lower() for p in path.parts]
        if any(p in {".github", ".circleci", ".gitlab"} for p in path_parts):
            return True

        # *rc files
        if filename.endswith("rc") and len(filename) > 2:
            return True

        return False

    def _is_likely_doc(self, path: Path) -> bool:
        """Check if file is likely documentation."""
        filename = path.name.lower()

        # Common doc patterns
        doc_patterns = ["readme", "changelog", "license", "contributing", "authors"]
        return any(pattern in filename for pattern in doc_patterns)

    def get_language(self, path: Path) -> Optional[str]:
        """Get the programming language for a code file."""
        extension = path.suffix.lower()

        language_map = {
            ".py": "python",
            ".pyx": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".kt": "kotlin",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".c": "c",
            ".h": "c",
            ".cpp": "cpp",
            ".hpp": "cpp",
            ".cs": "csharp",
            ".swift": "swift",
            ".sh": "bash",
            ".bash": "bash",
            ".lua": "lua",
            ".r": "r",
            ".R": "r",
            ".sql": "sql",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
        }

        return language_map.get(extension)
