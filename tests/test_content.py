"""The teaching text ships as data files and renders with the code's numbers."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path
from string import Formatter

import pytest

from image_classifier_lab import curriculum, explanations
from image_classifier_lab.content_files import CONTENT_DIR, fill, read_toml, read_toml_dir
from image_classifier_lab.data import IMAGE_SIZE

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{[a-z_]+\}")


def _fields(text: str) -> set[str]:
    return {field for _literal, field, _spec, _conv in Formatter().parse(text) if field}


def _texts(value: object) -> list[str]:
    """Every string in a parsed content file, except formulas, which are LaTeX."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _texts(item)]
    if isinstance(value, dict):
        return [text for key, item in value.items() if key != "formula" for text in _texts(item)]
    return []


def test_content_folder_ships_inside_the_package() -> None:
    assert CONTENT_DIR.joinpath("concepts.toml").is_file()
    names = sorted(item.name for item in CONTENT_DIR.joinpath("lessons").iterdir())
    assert [name for name in names if name.endswith(".toml")] == [
        f"{section.number:02d}-{section.key}.toml" for section in curriculum.SECTIONS
    ]
    assert CONTENT_DIR.joinpath("README.md").is_file()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package_data = project["tool"]["setuptools"]["package-data"]["image_classifier_lab"]
    assert "content/**/*.toml" in package_data


def test_lessons_load_in_file_order() -> None:
    records = read_toml_dir("lessons")
    assert [record["number"] for record in records] == list(range(1, len(records) + 1))
    assert [record["key"] for record in records] == [s.key for s in curriculum.SECTIONS]


def test_every_placeholder_has_a_value_and_every_value_is_used() -> None:
    lesson_fields = {
        field
        for record in read_toml_dir("lessons")
        for text in _texts(record)
        for field in _fields(text)
    }
    concept_fields = {
        field for text in _texts(read_toml("concepts.toml")) for field in _fields(text)
    }
    assert lesson_fields == set(curriculum.placeholder_values())
    assert concept_fields == set(explanations.placeholder_values())


def test_rendered_text_has_no_placeholders_left() -> None:
    rendered = [
        text
        for section in curriculum.SECTIONS
        for text in (
            section.short,
            section.title,
            *section.objectives,
            *(block.heading for block in section.blocks),
            *(paragraph for block in section.blocks for paragraph in block.paragraphs),
            *(
                part
                for _key, exercise in section.exercise_items()
                for part in (
                    exercise.instructions,
                    exercise.look_for,
                    exercise.question,
                    exercise.answer,
                )
            ),
        )
    ]
    rendered += [
        text
        for concept in explanations.CONCEPTS.values()
        for text in (concept.title, concept.summary, concept.body)
    ]
    assert rendered
    assert not [text for text in rendered if PLACEHOLDER.search(text)]
    tensor = next(
        block for block in curriculum.SECTION_BY_KEY["images"].blocks if block.visual == "tensor"
    )
    assert f"3 × {IMAGE_SIZE} × {IMAGE_SIZE}" in tensor.paragraphs[0]


def test_relocated_previews_are_referenced_in_lessons_and_help() -> None:
    for key in ("images", "neuron"):
        assert "Under the hood" in curriculum.SECTION_BY_KEY[key].exercise.instructions
    for key in ("dynamics", "transfer"):
        paragraphs = (
            paragraph
            for block in curriculum.SECTION_BY_KEY[key].blocks
            for paragraph in block.paragraphs
        )
        assert any("Under the hood" in paragraph for paragraph in paragraphs)
    for key in ("transfer_learning", "augmentation"):
        assert "Under the hood" in explanations.CONCEPTS[key].body


def test_concept_bodies_join_paragraphs_with_blank_lines() -> None:
    record = read_toml("concepts.toml")["transfer_learning"]
    body = explanations.CONCEPTS["transfer_learning"].body
    assert body.split("\n\n") == record["body"]
    assert list(explanations.CONCEPTS) == list(explanations.CONCEPT_ORDER)


def test_fill_uses_the_values_it_is_given() -> None:
    assert fill("{image_size} px", {"image_size": 96}) == "96 px"
    assert fill("no placeholders", {}) == "no placeholders"


@pytest.mark.parametrize("template", ["{missing}", "{image_size:>5}", "{image_size!r}"])
def test_fill_rejects_unknown_or_formatted_placeholders(template: str) -> None:
    with pytest.raises(ValueError, match="unknown placeholder"):
        fill(template, {"image_size": 96})
