from __future__ import annotations

import re
from importlib.metadata import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_public_release_documents_exist_and_are_nonempty() -> None:
    for name in ("LICENSE", "CHANGELOG.md", "SECURITY.md", "CONTRIBUTING.md"):
        path = ROOT / name
        assert path.is_file(), name
        assert path.stat().st_size > 100, name

    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text


def test_installed_package_exposes_public_release_metadata() -> None:
    package = metadata("llm-vis")

    assert package["Name"] == "llm-vis"
    assert package["Version"] == "0.1.0"
    assert package["License-Expression"] == "Apache-2.0"
    assert "LICENSE" in (package.get_all("License-File") or [])
    assert package["Requires-Python"] == ">=3.9"
    assert "llm" in (package["Keywords"] or "").split(",")

    urls = set(package.get_all("Project-URL") or [])
    assert "Homepage, https://github.com/zihaomu/llm-vis" in urls
    assert "Repository, https://github.com/zihaomu/llm-vis.git" in urls
    assert "Issues, https://github.com/zihaomu/llm-vis/issues" in urls
    assert "Changelog, https://github.com/zihaomu/llm-vis/blob/main/CHANGELOG.md" in urls
    assert "Security, https://github.com/zihaomu/llm-vis/security/policy" in urls

    classifiers = set(package.get_all("Classifier") or [])
    assert "Development Status :: 3 - Alpha" in classifiers
    assert "Programming Language :: Python :: 3.9" in classifiers
    assert "Programming Language :: Python :: 3.12" in classifiers
    assert "Topic :: Scientific/Engineering :: Artificial Intelligence" in classifiers


def test_readme_long_description_uses_portable_links_and_core_install() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    markdown_links = re.findall(r"\]\(([^)]+)\)", readme)

    assert markdown_links
    assert all(link.startswith(("https://", "http://", "#")) for link in markdown_links)
    assert "https://raw.githubusercontent.com/zihaomu/llm-vis/main/" in readme
    assert "uv tool install ." in readme
