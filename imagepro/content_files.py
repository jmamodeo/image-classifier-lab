"""Read the teaching text that ships in ``imagepro/content``.

The lessons and the control explanations are TOML files inside the package.
Numbers the code also uses appear in the text as named placeholders such as
``{image_size}``; the modules that build the lessons fill them from the same
constants the training code reads, so the prose cannot drift from the code.
"""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from importlib.resources import files
from importlib.resources.abc import Traversable
from string import Formatter
from typing import Any

CONTENT_DIR: Traversable = files("imagepro").joinpath("content")


def read_toml(*parts: str) -> dict[str, Any]:
    """Parse one TOML file under ``imagepro/content``."""
    return tomllib.loads(CONTENT_DIR.joinpath(*parts).read_text(encoding="utf-8"))


def read_toml_dir(*parts: str) -> tuple[dict[str, Any], ...]:
    """Parse every TOML file in one content folder, in file-name order."""
    folder = CONTENT_DIR.joinpath(*parts)
    paths = sorted(
        (item for item in folder.iterdir() if item.name.endswith(".toml")),
        key=lambda item: item.name,
    )
    return tuple(tomllib.loads(path.read_text(encoding="utf-8")) for path in paths)


def fill(template: str, values: Mapping[str, object]) -> str:
    """Replace ``{name}`` placeholders with values from the code.

    Only bare names are allowed, and each must be in ``values``; anything else
    is a mistake in a content file, so it raises instead of rendering.
    """
    for _literal, field, spec, conversion in Formatter().parse(template):
        if field is None:
            continue
        if spec or conversion or field not in values:
            raise ValueError(f"unknown placeholder {{{field}}} in content text")
    return template.format_map(values)
