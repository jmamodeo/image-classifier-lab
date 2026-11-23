"""Packaging, teaching copy, and a boot of the Streamlit script."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from streamlit.testing.v1 import AppTest

import image_classifier_lab
from image_classifier_lab.curriculum import SECTIONS
from image_classifier_lab.data import MAX_ROTATION_DEGREES
from image_classifier_lab.explanations import CONCEPT_ORDER, CONCEPTS
from image_classifier_lab.training import DEFAULT_LEARNING_RATE, LEARNING_RATES, MIN_BATCH_SIZE

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CONCEPTS = {
    "transfer_learning",
    "learning_rate",
    "batch_size",
    "epochs",
    "validation",
    "overfitting",
    "augmentation",
    "confidence",
    "seed",
}


def test_version_matches_pyproject() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == image_classifier_lab.__version__
    assert "John Amodeo" in project["project"]["authors"][0]["name"]


def test_license_and_changelog_name_the_release() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "Copyright (c) 2025-2026 John Amodeo" in license_text
    assert "Permission is hereby granted" in license_text
    assert "## 1.0.0" in changelog
    assert f"## {image_classifier_lab.__version__}" in changelog
    assert "Novel mode" in changelog
    assert "Augmentation" in changelog


def test_concepts_cover_the_teaching_goals() -> None:
    assert set(CONCEPT_ORDER) >= REQUIRED_CONCEPTS
    for key in CONCEPT_ORDER:
        concept = CONCEPTS[key]
        assert concept.title
        assert concept.summary
        assert len(concept.body) > 80
    assert f"{MAX_ROTATION_DEGREES:g}" in CONCEPTS["augmentation"].body
    assert str(MIN_BATCH_SIZE) in CONCEPTS["batch_size"].body
    assert DEFAULT_LEARNING_RATE in LEARNING_RATES
    assert all(rate > 0 for rate in LEARNING_RATES)


def test_readme_embeds_committed_screenshots() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Lessons in AI from an Image Classifier\n")
    assert "actions/workflows/ci.yml/badge.svg" in readme
    images = re.findall(r"!\[[^\]]*\]\((docs/[^)]+)\)", readme)
    assert len(images) >= 5
    for relative in images:
        path = ROOT / relative
        assert path.is_file(), relative
        assert path.stat().st_size > 10_000


def test_app_script_boots() -> None:
    app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=60)
    app.run()
    assert not app.exception
    titles = [title.value for title in app.title]
    assert "Welcome to Image Classifier Lab" in titles
    assert [item.label for item in app.expander] == [
        *(f"Lesson {section.number}: {section.title}" for section in SECTIONS),
        "Under the hood",
    ]
    assert app.header[0].value == "Image Classifier"
    markdown = " ".join(getattr(item, "value", "") for item in app.markdown)
    assert "Learning objectives" in markdown
    batch = next(item for item in app.sidebar.number_input if "Batch" in item.label)
    assert batch.value == 4
    assert any("Exercise" in getattr(item, "value", "") for item in app.markdown)
    assert any(button.label == "Train model" for button in app.button)
    assert all(button.label != "Open the training lab" for button in app.button)


def test_exercise_done_persists_across_lessons() -> None:
    """Checked exercises survive interactions in other lessons."""
    app = AppTest.from_file(str(ROOT / "main.py"), default_timeout=60)
    total = sum(len(section.exercise_items()) for section in SECTIONS)
    app.run()
    assert not app.exception
    app.checkbox(key="done-images").check().run()
    assert not app.exception
    assert app.checkbox(key="done-images").value is True
    assert _exercises_marked(app) == f"1 of {total} exercises marked done."

    app.slider(key="neuron-bias").set_value(1.0).run()
    assert not app.exception
    assert _exercises_marked(app) == f"1 of {total} exercises marked done."

    app.slider(key="norm-pixel").set_value(63).run()
    assert not app.exception
    assert app.checkbox(key="done-images").value is True
    assert _exercises_marked(app) == f"1 of {total} exercises marked done."


def _exercises_marked(app: AppTest) -> str:
    marked = [caption.value for caption in app.caption if "exercises marked done" in caption.value]
    assert len(marked) == 1
    return marked[0]
