#!/usr/bin/env python3
"""Verify release archives contain public metadata and the declared license."""

from __future__ import annotations

import argparse
import re
import tarfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from zipfile import ZipFile

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _single(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {pattern!r} in {directory}, found {len(matches)}"
        )
    return matches[0]


def _declared_version() -> str:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    project_block = pyproject.split("[project]", 1)[1].split("\n[", 1)[0]
    match = re.search(r'^version\s*=\s*"([^"]+)"\s*$', project_block, re.MULTILINE)
    if match is None:
        raise ValueError("pyproject.toml does not declare a static project version")
    return match.group(1)


def verify_distribution(directory: Path) -> tuple[Path, Path]:
    wheel = _single(directory, "*.whl")
    source = _single(directory, "*.tar.gz")
    version = _declared_version()

    with ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        license_names = [name for name in names if name.endswith(".dist-info/licenses/LICENSE")]
        if len(metadata_names) != 1 or len(license_names) != 1:
            raise ValueError("wheel must contain one METADATA file and one LICENSE file")
        metadata = BytesParser(policy=policy.default).parsebytes(
            archive.read(metadata_names[0])
        )
        expected_fields = {
            "Name": "llm-vis",
            "Version": version,
            "License-Expression": "Apache-2.0",
            "Requires-Python": ">=3.9",
            "Description-Content-Type": "text/markdown",
        }
        mismatched = {
            field: (metadata[field], expected)
            for field, expected in expected_fields.items()
            if metadata[field] != expected
        }
        if mismatched:
            raise ValueError(f"wheel metadata fields do not match: {mismatched}")
        if metadata.get_all("License-File") != ["LICENSE"]:
            raise ValueError("wheel must declare exactly one License-File: LICENSE")

        project_urls = set(metadata.get_all("Project-URL") or [])
        required_urls = {
            "Homepage, https://github.com/zihaomu/llm-vis",
            "Repository, https://github.com/zihaomu/llm-vis.git",
            "Issues, https://github.com/zihaomu/llm-vis/issues",
            "Documentation, https://github.com/zihaomu/llm-vis#readme",
            "Changelog, https://github.com/zihaomu/llm-vis/blob/main/CHANGELOG.md",
            "Security, https://github.com/zihaomu/llm-vis/security/policy",
        }
        if not required_urls.issubset(project_urls):
            missing_urls = required_urls - project_urls
            raise ValueError(f"wheel metadata is missing project URLs: {missing_urls}")

        requirements = metadata.get_all("Requires-Dist") or []
        if not any(requirement.lower().startswith("pydantic") for requirement in requirements):
            raise ValueError("wheel must declare pydantic as a core dependency")
        for optional_name in ("numpy", "torch"):
            matches = [
                requirement
                for requirement in requirements
                if requirement.lower().startswith(optional_name)
            ]
            if len(matches) != 1 or "extra == 'capture'" not in matches[0]:
                raise ValueError(
                    f"{optional_name} must appear exactly once and only in the capture extra"
                )

    with tarfile.open(source, "r:gz") as archive:
        names = set(archive.getnames())
        roots = {name.split("/", 1)[0] for name in names}
        if len(roots) != 1:
            raise ValueError(f"source distribution must have one root directory, found {roots}")
        root = next(iter(roots))
        forbidden_roots = {".git", ".venv", "artifacts", "dist"}
        bundled_forbidden = {
            name.split("/", 2)[1]
            for name in names
            if name.startswith(f"{root}/") and len(name.split("/", 2)) > 1
        } & forbidden_roots
        if bundled_forbidden:
            raise ValueError(
                f"source distribution contains forbidden roots: {bundled_forbidden}"
            )
        for required in (
            "LICENSE",
            "README.md",
            "CHANGELOG.md",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "pyproject.toml",
        ):
            if f"{root}/{required}" not in names:
                raise ValueError(f"source distribution is missing {required}")

    return wheel, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path, default=Path("dist"))
    args = parser.parse_args()
    wheel, source = verify_distribution(args.directory)
    print(f"distribution metadata valid: {wheel.name}, {source.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
