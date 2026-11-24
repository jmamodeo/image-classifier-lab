"""Streamlit interface for Lessons in AI from an Image Classifier."""

from __future__ import annotations

import html
import urllib.error
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import matplotlib.pyplot as plt
import streamlit as st
import torch
from matplotlib.figure import Figure
from PIL import Image, ImageOps

from image_classifier_lab import __version__
from image_classifier_lab.coreml import Export, Precision, export_model
from image_classifier_lab.data import (
    MAX_CLASSES,
    MIN_CLASSES,
    MIN_IMAGES_PER_CLASS,
    LabeledImages,
    SplitData,
    augmentation_demo_index,
    describe_split,
    fingerprint_groups,
    image_to_model_input,
    open_uploaded_file,
    prepare_image,
    preview_augmentations,
    split_dataset,
)
from image_classifier_lab.errors import DatasetError, TrainingError, UserFacingError
from image_classifier_lab.examples import (
    EXAMPLE_IMAGES_PER_CLASS,
    PUBLIC_IMAGES_PER_CLASS,
    make_cifar_dataset,
    make_shape_dataset,
)
from image_classifier_lab.explanations import CONCEPTS
from image_classifier_lab.hardware import NEURAL_ENGINE_REASON, detect_hardware
from image_classifier_lab.insights import summarize_training
from image_classifier_lab.lesson_ui import (
    LabSnapshot,
    append_run,
    render_curriculum,
    render_run_log,
    render_verification,
)
from image_classifier_lab.measure import format_bytes, make_run_record, summarize_timing
from image_classifier_lab.model import LearningMode, architecture_parameter_counts
from image_classifier_lab.plots import plot_confusion, plot_history, plot_probabilities
from image_classifier_lab.training import (
    DEFAULT_AUGMENT,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_SEED,
    DEFAULT_SHUFFLE,
    DEFAULT_VAL_PERCENT,
    LEARNING_RATES,
    MAX_BATCH_SIZE,
    MAX_EPOCHS,
    MIN_BATCH_SIZE,
    TRAINING_DEVICE_LABELS,
    TrainResult,
    predict_probabilities,
    resolve_batching,
    train_classifier,
)

PREVIEW_COUNT = 4
THUMBNAIL_LIMIT = 10
THUMBNAIL_PIXELS = 128
THUMBNAIL_WIDTH = 60
CARD_COLUMNS = (2, 3)
RESULT_TABS = ("Training curves", "Confusion matrix", "Per-class accuracy")
UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]
SOURCE_LABELS = {
    "shapes": "Built-in shapes",
    "public": "Built-in images",
    "upload": "Upload images",
}
_DOWNLOAD_MESSAGE = (
    "Could not download the ImageNet weights for transfer mode. Check the network "
    "connection, or switch to Novel mode, which starts from random weights and does "
    "not download anything."
)


@dataclass(frozen=True)
class Settings:
    """The controls that define one training run."""

    device_label: str
    device: str
    mode: LearningMode
    learning_rate: float
    batch_size: int
    epochs: int
    val_percent: int
    shuffle: bool
    augment: bool
    seed: int
    inference_backends: tuple[str, ...]
    precision: Precision
    coreml_reason: str | None

    @property
    def val_fraction(self) -> float:
        return self.val_percent / 100


@dataclass
class Artifact:
    """A finished run, kept in session state so prediction does not retrain."""

    model: torch.nn.Module
    class_names: tuple[str, ...]
    result: TrainResult
    device: str
    signature: tuple[object, ...]
    summary: tuple[str, ...]
    mode: LearningMode
    train_size: int
    val_size: int
    details: str
    val_images: tuple[Image.Image, ...]


APP_TITLE = "Lessons in AI from an Image Classifier"
WELCOME_TITLE = "Welcome to Image Classifier Lab"
_APP_ICON = (
    "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' "
    "fill='none' stroke='white' stroke-width='2' stroke-linecap='round' "
    "stroke-linejoin='round'%3E%3Crect x='3' y='3' width='18' height='18' rx='3'/%3E"
    "%3Ccircle cx='9' cy='9' r='2'/%3E%3Cpath d='m21 15-5-5L5 21'/%3E%3C/svg%3E"
)
# The classifier is drawn as an app window: keyed containers get an ``st-key-`` class.
_STYLES = f"""
<style>
h1#welcome-to-image-classifier-lab {{
    font-family: Georgia, serif;
    font-style: italic;
    font-weight: 700;
}}
.st-key-hardware-backward,
.st-key-hardware-forward,
.st-key-hardware-precision {{
    gap: 0.4rem;
}}
.st-key-hardware-backward [data-testid="stWidgetLabel"] p,
.st-key-hardware-precision [data-testid="stWidgetLabel"] p {{
    font-size: 1rem;
    font-weight: 600;
    line-height: 1.6;
}}
.st-key-hardware-backward [role="radiogroup"] {{
    min-height: 0;
}}
.st-key-hardware-forward,
.st-key-hardware-precision {{
    padding-top: 0.75rem;
    border-top: 1px solid rgba(23, 32, 42, 0.12);
}}
.st-key-hardware-training-gpu {{
    margin-top: -0.4rem;
}}
.st-key-classifier-app {{
    gap: 0;
    margin-top: 0.75rem;
    border: 1px solid rgba(23, 32, 42, 0.14);
    border-radius: 14px;
    box-shadow: 0 1px 2px rgba(23, 32, 42, 0.05), 0 22px 48px -22px rgba(23, 32, 42, 0.35);
}}
.st-key-app-titlebar {{
    padding: 0.7rem 1.25rem;
    border-radius: 13px 13px 0 0;
    background: linear-gradient(120deg, #10273d 0%, #1f4e79 62%, #2f6ea5 100%);
}}
.st-key-app-titlebar h2 {{
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 0;
    padding: 0;
    color: #ffffff;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}}
.st-key-app-titlebar h2::before {{
    content: "";
    flex: none;
    width: 1.9rem;
    height: 1.9rem;
    border-radius: 0.55rem;
    background: rgba(255, 255, 255, 0.14) url("{_APP_ICON}") center / 1.15rem no-repeat;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.25);
}}
.ic-chip {{
    display: flex;
    justify-content: flex-end;
}}
.ic-chip span {{
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.13);
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.2);
    color: #e6eef6;
    font-size: 0.78rem;
    white-space: nowrap;
}}
.st-key-app-toolbar,
.st-key-app-data,
.st-key-app-train,
.st-key-app-results,
.st-key-app-predict,
.st-key-app-export {{
    padding: 1rem 1.25rem 1.15rem;
    border-top: 1px solid rgba(23, 32, 42, 0.09);
}}
.st-key-app-toolbar {{
    padding-top: 0.8rem;
    padding-bottom: 0.8rem;
    border-top: none;
    background: rgba(23, 32, 42, 0.03);
}}
.st-key-app-statusbar {{
    padding: 0.45rem 1.25rem;
    border-top: 1px solid rgba(23, 32, 42, 0.09);
    border-radius: 0 0 13px 13px;
    background: rgba(23, 32, 42, 0.04);
}}
/* Streamlit pulls markdown up 1rem to absorb a paragraph margin these bands don't have. */
.st-key-app-titlebar [data-testid="stMarkdownContainer"],
.st-key-app-statusbar [data-testid="stMarkdownContainer"] {{
    margin-bottom: 0;
}}
.ic-head {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
}}
.ic-label {{
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    opacity: 0.62;
}}
.ic-meta {{
    font-size: 0.8rem;
    opacity: 0.62;
}}
.ic-statusbar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: 0.25rem 1rem;
    font-size: 0.76rem;
    opacity: 0.78;
}}
.ic-status::before {{
    content: "";
    display: inline-block;
    width: 0.5rem;
    height: 0.5rem;
    margin-right: 0.4rem;
    border-radius: 50%;
    background: #9aa5b1;
}}
.ic-status-ready::before {{
    background: #2f9e44;
}}
.ic-status-stale::before {{
    background: #e8590c;
}}
.ic-empty {{
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: {THUMBNAIL_WIDTH}px;
    border: 1.5px dashed rgba(23, 32, 42, 0.2);
    border-radius: 8px;
    font-size: 0.8rem;
    opacity: 0.7;
}}
.st-key-source [role="radiogroup"] {{
    display: inline-flex;
    flex-wrap: wrap;
    gap: 2px;
    padding: 3px;
    border-radius: 10px;
    background: rgba(23, 32, 42, 0.08);
}}
.st-key-source [role="radiogroup"] label {{
    margin: 0;
    padding: 0.3rem 0.95rem;
    border-radius: 8px;
    cursor: pointer;
}}
/* The radio dot: the label's first div in older Streamlit, nested a level deeper in newer. */
.st-key-source [role="radiogroup"] label > div:first-child,
.st-key-source [role="radiogroup"] label > div > div:not([data-testid="stMarkdownContainer"]) {{
    display: none;
}}
.st-key-source [role="radiogroup"] label:has(input:checked) {{
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(23, 32, 42, 0.16);
}}
.st-key-source [role="radiogroup"] label:has(input:checked) p {{
    color: #1f4e79;
    font-weight: 600;
}}
.st-key-source [role="radiogroup"] label:has(input:focus-visible) {{
    outline: 2px solid #1f4e79;
    outline-offset: 1px;
}}
.st-key-app-data img {{
    border-radius: 6px;
}}
[class*="st-key-class_card_"] [data-testid="stFileUploaderDropzoneInstructions"] {{
    display: none;
}}
[class*="st-key-remove_"] [data-testid="stButton"],
.st-key-train_model [data-testid="stButton"] {{
    display: flex;
    justify-content: flex-end;
}}
/* Newer Streamlit shrinks a button's container to fit it, so push the container right too. */
[class*="st-key-remove_"],
.st-key-train_model {{
    margin-left: auto;
}}
/* A proportional column squeezes the train button's label; give it a fixed width instead. */
@media (min-width: 641px) {{
    .st-key-app-train [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child {{
        flex: 1 1 0;
        min-width: 0;
    }}
    .st-key-app-train [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child {{
        flex: 0 0 9.5rem;
        width: 9.5rem;
    }}
}}
.st-key-prediction-result strong {{
    font-size: 1.5rem;
}}
/* Metric tiles are narrow inside the frame: wrap labels instead of truncating them. */
.st-key-app-results [data-testid="stMetricLabel"] {{
    min-height: 2.5em;
    align-items: flex-start;
}}
.st-key-app-results [data-testid="stMetricLabel"] * {{
    overflow: visible;
    white-space: normal;
    text-overflow: clip;
    line-height: 1.25;
}}
.st-key-app-results [data-testid="stMetricValue"] {{
    font-size: 1.65rem;
    font-weight: 600;
}}
/* Older Streamlit prints the table's row numbers; newer versions already hide them. */
.st-key-app-results [data-testid="stTableStyledTable"] thead th.blank,
.st-key-app-results [data-testid="stTableStyledTable"] tbody th {{
    display: none;
}}
</style>
"""


def main() -> None:
    """Run the app. Use ``streamlit run main.py``."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()
    settings = _sidebar()
    st.markdown(_STYLES, unsafe_allow_html=True)
    st.title(WELCOME_TITLE)
    st.caption(
        "Eight expandable lessons on how this image classifier learns, followed by "
        "the Image Classifier itself. The exercises name the settings "
        "to change. Transfer mode starts from ImageNet and trains a new final layer. "
        "Novel mode trains every layer from random weights."
    )
    lesson_slot = st.container()
    snapshot = _render_workspace(settings)
    with lesson_slot:
        render_curriculum(snapshot)


def _render_workspace(settings: Settings) -> LabSnapshot:
    """The classifier, framed as an app window, with its teaching notes below the frame."""
    st.markdown('<div id="training-lab"></div>', unsafe_allow_html=True)
    with st.container(key="classifier-app"):
        _render_titlebar(settings)
        with _section("app-toolbar"):
            source = _source_picker()
        with _section("app-data"):
            groups = _render_training_data(source)

        class_count = len(groups) if groups is not None else _slot_count(source)
        split: SplitData | None = None
        problem: tuple[str, str] | None = None
        if groups is not None:
            split, problem = _try_split(groups, settings)
        batch_note = _batch_note(split, settings) if split is not None else None
        if isinstance(batch_note, str) and batch_note.startswith("error:"):
            problem = ("error", batch_note.removeprefix("error:"))
            split = None
            batch_note = None

        with _section("app-train"):
            status, action = st.columns((3, 1), vertical_alignment="center")
            with status:
                _render_train_status(groups, split, problem)
            with action:
                clicked = st.button(
                    "Train model",
                    type="primary",
                    icon=":material/play_arrow:",
                    disabled=split is None,
                    key="train_model",
                )
            if clicked and split is not None and groups is not None:
                artifact = _run_training(split, groups, settings, batch_note)
                if artifact is not None:
                    st.session_state.artifact = artifact
                    st.session_state.coreml_exports = {}
                    st.session_state.coreml_benchmarks = {}

        stored = st.session_state.get("artifact")
        trained = stored if isinstance(stored, Artifact) else None
        signature = _signature(settings, groups)
        if trained is not None:
            with _section("app-results"):
                _render_results(trained, signature)
            with _section("app-predict"):
                _render_prediction(trained)
            with _section("app-export"):
                _render_export(trained, settings.precision, settings.coreml_reason)
        with _section("app-statusbar"):
            _render_statusbar(split, trained, signature)

    _render_under_the_hood(settings, class_count, split, batch_note, trained)
    return LabSnapshot(
        class_count=class_count,
        source=str(source),
        mode=settings.mode,
        batch_size=settings.batch_size,
        val_percent=settings.val_percent,
        seed=settings.seed,
        device_label=settings.device_label,
        device=settings.device,
        signature=signature,
        trained=trained,
        inference_backends=settings.inference_backends,
        precision=settings.precision,
    )


@contextmanager
def _section(key: str) -> Iterator[None]:
    """One band of the app window. The keyed block is styled and padded; the content
    sits in an inner block, which Streamlit sizes to fit inside that padding."""
    with st.container(key=key), st.container():
        yield


def _render_titlebar(settings: Settings) -> None:
    with _section("app-titlebar"):
        name, chip = st.columns((3, 2), vertical_alignment="center")
        with name:
            st.header("Image Classifier", anchor=False)
        details = f"ResNet-18 · {settings.mode.title} · {settings.device_label}"
        chip.markdown(
            f'<div class="ic-chip"><span>{html.escape(details)}</span></div>',
            unsafe_allow_html=True,
        )


def _section_head(label: str, meta: str = "") -> str:
    return (
        f'<div class="ic-head"><span class="ic-label">{html.escape(label)}</span>'
        f'<span class="ic-meta">{html.escape(meta)}</span></div>'
    )


def _source_picker() -> str:
    source = st.radio(
        "Image source",
        options=list(SOURCE_LABELS),
        format_func=SOURCE_LABELS.__getitem__,
        horizontal=True,
        key="source",
        index=2,
        label_visibility="collapsed",
    )
    st.caption(_source_caption(str(source)))
    return str(source)


def _source_caption(source: str) -> str:
    if source == "shapes":
        return (
            f"Circles, squares, and triangles drawn by the app, {EXAMPLE_IMAGES_PER_CLASS} "
            "per class. Simple on purpose, so a short run shows clear curves."
        )
    if source == "public":
        return (
            "Public CIFAR-10 images: airplane, automobile, and bird, "
            f"{PUBLIC_IMAGES_PER_CLASS} labeled images per class. Downloaded once, then "
            "cached on this machine."
        )
    return (
        f"Your own images: {MIN_CLASSES} to {MAX_CLASSES} classes, each with a label and at "
        f"least {MIN_IMAGES_PER_CLASS} images. PNG, JPEG, WebP, or BMP."
    )


def _render_train_status(
    groups: list[LabeledImages] | None,
    split: SplitData | None,
    problem: tuple[str, str] | None,
) -> None:
    if problem is not None:
        level, message = problem
        if level == "info":
            st.info(message)
        else:
            st.error(message)
    elif groups is None:
        st.caption("Fix the training data above to train.")
    elif split is not None:
        st.caption(
            f"{_count(len(split.train_images), 'training image')} · "
            f"{_count(len(split.val_images), 'validation image')} held out to score the run."
        )


def _render_statusbar(
    split: SplitData | None,
    trained: Artifact | None,
    signature: tuple[object, ...] | None,
) -> None:
    if trained is not None and signature == trained.signature:
        accuracy = trained.result.history[-1].val_accuracy
        state, tone = f"Trained · {accuracy:.0%} validation accuracy", "ready"
    elif trained is not None:
        state, tone = "Settings changed · train again", "stale"
    elif split is not None:
        state, tone = "Ready to train", "ready"
    else:
        state, tone = "Waiting for training data", "idle"
    st.markdown(
        '<div class="ic-statusbar">'
        f"<span>{html.escape(APP_TITLE)} {html.escape(__version__)} · "
        "John Amodeo · MIT (code) · CC BY 4.0 (lessons)</span>"
        f'<span class="ic-status ic-status-{tone}">{html.escape(state)}</span></div>',
        unsafe_allow_html=True,
    )


def _render_under_the_hood(
    settings: Settings,
    class_count: int,
    split: SplitData | None,
    batch_note: str | None,
    trained: Artifact | None,
) -> None:
    """Explanations and measurements behind the classifier, kept out of its frame."""
    with st.expander("Under the hood"):
        st.caption("What the classifier above is doing. The lessons point here.")
        _render_mode_cards(class_count, settings.mode)
        if split is not None:
            _render_preview(split, settings)
        if batch_note:
            st.caption(batch_note)
        if trained is not None:
            _render_run_notes(trained)
            st.markdown("**Trying an image**")
            st.caption(
                "Resize and normalize only, the same path as validation. No flip and no "
                "rotation. " + CONCEPTS["confidence"].summary
            )
        st.markdown("**Recent runs**")
        render_run_log()


def _ensure_state() -> None:
    if "class_ids" not in st.session_state:
        st.session_state.class_ids = [1, 2]
        st.session_state.next_class_id = 3


def _slot_count(source: str) -> int:
    if source in ("shapes", "public"):
        return 3
    return len(st.session_state.class_ids)


def _sidebar() -> Settings:
    concepts = CONCEPTS
    st.sidebar.header("Settings")
    st.sidebar.caption("The lessons name which of these to move, and which to leave.")
    hardware = detect_hardware()
    with st.sidebar.container(border=True, key="hardware-panel"):
        st.subheader("Hardware")
        cpu = next(option for option in hardware.inference if option.key == "cpu")
        gpu = next(option for option in hardware.inference if option.key in {"cuda", "mps", "gpu"})
        neural_engine = next(option for option in hardware.inference if option.key == "CPU_AND_NE")
        devices = {"cpu": "CPU"}
        if gpu.available and gpu.key in {"cuda", "mps"}:
            devices[gpu.key] = "GPU"
        with st.container(key="hardware-backward"):
            device = st.radio(
                "Training",
                options=tuple(devices),
                format_func=devices.__getitem__,
            )
            if not gpu.available:
                with st.container(key="hardware-training-gpu"):
                    st.radio(
                        "GPU unavailable",
                        options=("GPU",),
                        index=None,
                        disabled=True,
                        label_visibility="collapsed",
                        key="training-gpu-unavailable",
                    )
        backends = []
        with st.container(key="hardware-forward"):
            st.markdown("**Inference**")
            cpu_selected = st.checkbox(
                "CPU", value=cpu.available, disabled=not cpu.available, key="inference-cpu"
            )
            if cpu_selected and cpu.available:
                backends.append(cpu.key)

            neural_reasons = []
            if not hardware.apple_silicon:
                neural_reasons.append(NEURAL_ENGINE_REASON)
            elif neural_engine.reason is not None:
                neural_reasons.append(neural_engine.reason)
            if not cpu_selected:
                neural_reasons.append("Select CPU to enable Neural Engine.")
            with st.container(key="hardware-inference-neural"):
                neural_selected = st.checkbox(
                    "Neural Engine",
                    value=neural_engine.available,
                    disabled=bool(neural_reasons),
                    key=f"inference-{neural_engine.key}",
                )

            with st.container(key="hardware-inference-gpu"):
                checked = st.checkbox(
                    "GPU",
                    value=gpu.available,
                    disabled=not gpu.available,
                    key=f"inference-{gpu.key}",
                )
            if checked and gpu.available:
                backends.append(gpu.key)
            if neural_selected and not neural_reasons:
                backends.append(neural_engine.key)

        with st.container(key="hardware-precision"):
            precision: Precision = st.radio(
                "Precision",
                options=("float32", "float16"),
                key="coreml_precision",
            )
    coreml_reason = next(option.reason for option in hardware.inference if option.key == "CPU_ONLY")

    mode = st.sidebar.selectbox(
        "Learning mode",
        options=[LearningMode.TRANSFER, LearningMode.NOVEL],
        format_func=_mode_label,
        help=concepts["transfer_learning"].summary,
    )
    learning_rate = float(
        st.sidebar.select_slider(
            "Learning rate",
            options=LEARNING_RATES,
            value=DEFAULT_LEARNING_RATE,
            format_func=lambda value: f"{value:.4g}",
            help=concepts["learning_rate"].summary,
        )
    )
    st.sidebar.caption(f"Adam step size: {learning_rate:.4g}.")
    batch_size = int(
        st.sidebar.number_input(
            "Batch size",
            min_value=MIN_BATCH_SIZE,
            max_value=MAX_BATCH_SIZE,
            value=DEFAULT_BATCH_SIZE,
            step=1,
            help=concepts["batch_size"].summary,
        )
    )
    epochs = int(
        st.sidebar.slider(
            "Epochs",
            min_value=1,
            max_value=MAX_EPOCHS,
            value=DEFAULT_EPOCHS,
            help=concepts["epochs"].summary,
        )
    )
    val_percent = int(
        st.sidebar.slider(
            "Validation holdout (%)",
            min_value=10,
            max_value=50,
            value=DEFAULT_VAL_PERCENT,
            step=5,
            help=concepts["validation"].summary,
        )
    )
    st.sidebar.caption(concepts["validation"].summary)
    shuffle = bool(
        st.sidebar.checkbox(
            "Shuffle training images each epoch",
            value=DEFAULT_SHUFFLE,
            help=concepts["shuffle"].summary,
        )
    )
    augment = bool(
        st.sidebar.checkbox(
            "Augment training images",
            value=DEFAULT_AUGMENT,
            help=concepts["augmentation"].summary,
        )
    )
    seed = int(
        st.sidebar.number_input(
            "Seed",
            min_value=0,
            max_value=1_000_000,
            value=DEFAULT_SEED,
            step=1,
            help=concepts["seed"].summary,
        )
    )
    return Settings(
        device_label=devices[device],
        device=device,
        mode=mode,
        learning_rate=learning_rate,
        batch_size=batch_size,
        epochs=epochs,
        val_percent=val_percent,
        shuffle=shuffle,
        augment=augment,
        seed=seed,
        inference_backends=tuple(backends),
        precision=precision,
        coreml_reason=coreml_reason,
    )


def _mode_label(mode: LearningMode) -> str:
    if mode is LearningMode.TRANSFER:
        return "Transfer (pretrained backbone)"
    return "Novel (from scratch)"


def _render_training_data(source: str) -> list[LabeledImages] | None:
    """One card per class, the same for every source. ``None`` until the cards are usable."""
    head = st.empty()
    if source == "upload":
        groups, complete = _upload_groups()
    else:
        loaded = _builtin_groups(source)
        groups, complete = loaded or [], loaded is not None
    total = sum(len(group.images) for group in groups)
    head.markdown(
        _section_head(
            "Training data",
            f"{_count(len(groups), 'class', 'classes')} · {_count(total, 'image')}",
        ),
        unsafe_allow_html=True,
    )
    return groups if complete else None


def _builtin_groups(source: str) -> list[LabeledImages] | None:
    try:
        groups = make_shape_dataset() if source == "shapes" else _public_image_groups()
    except (DatasetError, RuntimeError, urllib.error.URLError, TimeoutError, OSError) as exc:
        st.error(str(exc))
        return None
    for position, group in enumerate(groups, start=1):
        with st.container(border=True, key=f"class_card_{source}_{position}"):
            _card_header(position)
            label_column, images_column = st.columns(CARD_COLUMNS, gap="medium")
            label_column.text_input(
                "Label",
                value=group.label,
                key=f"builtin_label_{source}_{position}",
                disabled=True,
            )
            with images_column:
                _render_thumbnails(group.images)
    return groups


@st.cache_resource(show_spinner="Downloading the public CIFAR-10 dataset...")
def _cached_public_image_groups() -> list[LabeledImages]:
    return make_cifar_dataset()


def _public_image_groups() -> list[LabeledImages]:
    try:
        return _cached_public_image_groups()
    except Exception as exc:
        raise RuntimeError(
            "Could not download the public CIFAR-10 dataset. Check your network connection "
            "and try Built-in images again."
        ) from exc


def _upload_groups() -> tuple[list[LabeledImages], bool]:
    """The editable cards. ``False`` when a file in any card cannot be read."""
    groups: list[LabeledImages] = []
    complete = True
    removable = len(st.session_state.class_ids) > MIN_CLASSES
    for position, class_id in enumerate(list(st.session_state.class_ids), start=1):
        with st.container(border=True, key=f"class_card_upload_{class_id}"):
            _card_header(position, int(class_id) if removable else None)
            label_column, images_column = st.columns(CARD_COLUMNS, gap="medium")
            with label_column:
                label = st.text_input("Label", key=f"label_{class_id}", placeholder="e.g. daisy")
                files = st.file_uploader(
                    "Images",
                    key=f"files_{class_id}",
                    accept_multiple_files=True,
                    type=UPLOAD_TYPES,
                )
            images: list[Image.Image] = []
            with images_column:
                try:
                    images = [
                        open_uploaded_file(file.getvalue(), file.name) for file in files or ()
                    ]
                except DatasetError as exc:
                    st.error(str(exc))
                    images = []
                    complete = False
                else:
                    _render_thumbnails(images)
            groups.append(LabeledImages(str(label).strip(), tuple(images)))
    st.button(
        "Add class",
        key="add_class",
        icon=":material/add:",
        disabled=len(st.session_state.class_ids) >= MAX_CLASSES,
        on_click=_add_class,
    )
    return groups, complete


def _card_header(position: int, removable_id: int | None = None) -> None:
    """The class number, and a Remove button for ``removable_id`` when one is given."""
    title, action = st.columns((5, 1), vertical_alignment="center")
    title.markdown(f"**Class {position}**")
    if removable_id is not None:
        with action:
            st.button(
                "Remove",
                key=f"remove_{removable_id}",
                type="tertiary",
                icon=":material/close:",
                on_click=_remove_class,
                args=(removable_id,),
            )


def _render_thumbnails(images: Sequence[Image.Image]) -> None:
    if not images:
        st.markdown(
            '<div class="ic-empty">Images you add appear here</div>',
            unsafe_allow_html=True,
        )
        return
    shown = images[:THUMBNAIL_LIMIT]
    st.image([_thumbnail(image) for image in shown], width=THUMBNAIL_WIDTH)
    caption = _count(len(images), "image")
    if len(images) > len(shown):
        caption += f" · showing {len(shown)}"
    st.caption(caption)


def _thumbnail(image: Image.Image) -> Image.Image:
    """A centered square crop, so every class shows an even strip of tiles."""
    side = min(THUMBNAIL_PIXELS, *image.size)
    return ImageOps.fit(image, (side, side), Image.Resampling.LANCZOS)


def _count(number: int, singular: str, plural: str | None = None) -> str:
    return f"{number} {singular if number == 1 else plural or singular + 's'}"


def _add_class() -> None:
    class_ids = list(st.session_state.class_ids)
    if len(class_ids) >= MAX_CLASSES:
        return
    class_ids.append(int(st.session_state.next_class_id))
    st.session_state.class_ids = class_ids
    st.session_state.next_class_id = int(st.session_state.next_class_id) + 1


def _remove_class(class_id: int) -> None:
    class_ids = [item for item in st.session_state.class_ids if item != class_id]
    if len(class_ids) < MIN_CLASSES:
        return
    st.session_state.class_ids = class_ids


def _is_unstarted(groups: list[LabeledImages]) -> bool:
    return all(not group.label and not group.images for group in groups)


def _try_split(
    groups: list[LabeledImages],
    settings: Settings,
) -> tuple[SplitData | None, tuple[str, str] | None]:
    if _is_unstarted(groups):
        return None, (
            "info",
            "Add a label and at least two images to each class, or switch to the built-in shapes.",
        )
    try:
        return split_dataset(groups, settings.val_fraction, settings.seed), None
    except DatasetError as exc:
        return None, ("error", str(exc))


def _render_mode_cards(num_classes: int, selected: LearningMode) -> None:
    st.markdown("**What this run will update**")
    st.caption(
        "Same images, loss, and learning rate either way. The difference is the starting "
        "weights and which parameters can change."
    )
    blurbs = {
        LearningMode.TRANSFER: "Final layer only. ImageNet backbone stays frozen.",
        LearningMode.NOVEL: "Every layer. Weights start random.",
    }
    columns = st.columns(2)
    for column, mode in zip(columns, (LearningMode.TRANSFER, LearningMode.NOVEL), strict=True):
        trainable, total = architecture_parameter_counts(num_classes, mode.value)
        with column, st.container(border=True):
            marker = "This run" if mode is selected else "For comparison"
            st.markdown(f"**{marker} · {mode.title}**")
            st.caption(blurbs[mode])
            st.metric("Trainable parameters", f"{trainable:,}")
            st.caption(f"of {total:,} in this ResNet-18")


def _render_preview(split: SplitData, settings: Settings) -> None:
    st.markdown("**Augmentation and the holdout**")
    st.caption(
        f"{len(split.train_images)} training images · {len(split.val_images)} validation images. "
        f"{describe_split(split)}."
    )
    subject = augmentation_demo_index(split.train_images)
    train_name = split.class_names[split.train_labels[subject]]
    prepared = prepare_image(split.train_images[subject])
    val_index = next(
        (
            index
            for index, label in enumerate(split.val_labels)
            if label == split.train_labels[subject]
        ),
        0,
    )
    val_name = split.class_names[split.val_labels[val_index]]
    validation = prepare_image(split.val_images[val_index])
    if settings.augment:
        views = preview_augmentations(split.train_images[subject], PREVIEW_COUNT, settings.seed)
        st.caption(
            "Each draw is a new left-right flip and a small rotation of one training image. "
            "Training draws again every time that image is loaded, once per epoch. "
            "Validation is resized only. The preview uses its own random numbers, so looking "
            "at it does not change the run."
        )
        columns = st.columns(PREVIEW_COUNT + 2)
        columns[0].image(_shrunk(prepared, 140), caption=f"Train · {train_name}")
        for index, view in enumerate(views, start=1):
            columns[index].image(_shrunk(view, 140), caption=f"Draw {index}")
        columns[PREVIEW_COUNT + 1].image(
            _shrunk(validation, 140), caption=f"Validation · {val_name}"
        )
        return
    st.caption(
        "Augmentation is off, so every epoch sees these resized pixels. "
        "Validation is resized the same way."
    )
    left, right = st.columns(2)
    left.image(_shrunk(prepared, 220), caption=f"Train · {train_name}")
    right.image(_shrunk(validation, 220), caption=f"Validation · {val_name}")


def _shrunk(image: Image.Image, width: int) -> Image.Image:
    """A copy at most ``width`` wide. Without a fixed width, st.image fits it to its column."""
    copy = image.copy()
    copy.thumbnail((width, copy.height), Image.Resampling.LANCZOS)
    return copy


def _batch_note(split: SplitData, settings: Settings) -> str | None:
    try:
        plan = resolve_batching(len(split.train_images), settings.batch_size, batchnorm=True)
    except TrainingError as exc:
        return f"error:{exc}"
    return plan.note


def _run_training(
    split: SplitData,
    groups: list[LabeledImages],
    settings: Settings,
    batch_note: str | None,
) -> Artifact | None:
    progress = st.progress(0.0, text=_prepare_text(settings))

    def on_epoch(history: tuple) -> None:
        last = history[-1]
        progress.progress(
            last.epoch / settings.epochs,
            text=(
                f"Epoch {last.epoch} of {settings.epochs}"
                f" · train loss {last.train_loss:.3f}"
                f" · validation loss {last.val_loss:.3f}"
            ),
        )

    try:
        model, result = train_classifier(
            split,
            mode=settings.mode,
            epochs=settings.epochs,
            learning_rate=settings.learning_rate,
            batch_size=settings.batch_size,
            seed=settings.seed,
            shuffle=settings.shuffle,
            augment=settings.augment,
            device=settings.device,
            on_epoch=on_epoch,
        )
    except UserFacingError as exc:
        progress.empty()
        st.error(str(exc))
        return None
    except (urllib.error.URLError, TimeoutError, ConnectionError, RuntimeError, OSError) as exc:
        progress.empty()
        if settings.mode is LearningMode.TRANSFER and _is_download_error(exc):
            st.error(_DOWNLOAD_MESSAGE)
            return None
        raise

    progress.progress(1.0, text="Training finished.")
    append_run(
        make_run_record(
            result,
            mode=settings.mode.title,
            learning_rate=settings.learning_rate,
            batch_size=settings.batch_size,
            epochs=settings.epochs,
            augment=settings.augment,
            device_label=settings.device_label,
            seed=settings.seed,
        )
    )
    summary = summarize_training(
        result.history,
        mode=settings.mode,
        trainable=result.trainable_parameters,
        total=result.total_parameters,
        train_size=len(split.train_images),
        val_size=len(split.val_images),
    )
    return Artifact(
        model=model,
        class_names=split.class_names,
        result=result,
        device=settings.device,
        signature=_signature(settings, groups) or (),
        summary=summary,
        mode=settings.mode,
        train_size=len(split.train_images),
        val_size=len(split.val_images),
        details=_details(settings, batch_note),
        val_images=split.val_images,
    )


def _prepare_text(settings: Settings) -> str:
    if settings.mode is LearningMode.TRANSFER:
        return (
            "Preparing ResNet-18. The first transfer run downloads about 45 MB of ImageNet weights."
        )
    return "Preparing ResNet-18 from random weights."


def _is_download_error(exc: BaseException) -> bool:
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    tokens = ("download", "certificate", "urlerror", "http", "network", "timed out")
    return any(token in message for token in tokens)


def _details(settings: Settings, batch_note: str | None) -> str:
    augment = "augmentation on" if settings.augment else "augmentation off"
    shuffle = "shuffle on" if settings.shuffle else "shuffle off"
    text = (
        f"{settings.mode.title} · {settings.device_label} · Adam learning rate "
        f"{settings.learning_rate:.4g} · batch {settings.batch_size} · "
        f"{settings.epochs} epochs · seed {settings.seed} · {augment} · {shuffle} · "
        f"{settings.val_percent}% holdout requested"
    )
    if batch_note:
        return f"{text}. {batch_note}"
    return text


def _signature(
    settings: Settings,
    groups: list[LabeledImages] | None,
) -> tuple[object, ...] | None:
    if groups is None:
        return None
    return (
        settings.mode.value,
        settings.device,
        round(float(settings.learning_rate), 6),
        int(settings.batch_size),
        int(settings.epochs),
        int(settings.val_percent),
        bool(settings.shuffle),
        bool(settings.augment),
        int(settings.seed),
        fingerprint_groups(groups),
    )


def _render_results(artifact: Artifact, signature: tuple[object, ...] | None) -> None:
    history = artifact.result.history
    st.markdown(
        _section_head("Results", f"{artifact.mode.title} · {_count(len(history), 'epoch')}"),
        unsafe_allow_html=True,
    )
    if signature != artifact.signature:
        st.warning(
            "Showing the previous run. The images or settings have changed, so train again "
            "before reading this as the current setup."
        )
    _render_metrics(artifact)
    curves, confusion, per_class = st.tabs(RESULT_TABS)
    with curves:
        _show_figure(plot_history(history))
    with confusion:
        _, matrix, _ = st.columns((1, 3, 1))
        with matrix:
            _show_figure(plot_confusion(artifact.result.confusion, artifact.class_names))
    with per_class:
        _render_class_table(artifact)


def _render_metrics(artifact: Artifact) -> None:
    history = artifact.result.history
    last = history[-1]
    seconds = sum(epoch.train_seconds + epoch.val_seconds for epoch in history)
    tiles = (
        ("Validation accuracy", f"{last.val_accuracy:.0%}"),
        ("Validation loss", f"{last.val_loss:.3f}"),
        ("Training accuracy", f"{last.train_accuracy:.0%}"),
        ("Training time", f"{seconds:.1f} s" if seconds > 0 else "—"),
    )
    for column, (label, value) in zip(st.columns(len(tiles)), tiles, strict=True):
        column.metric(label, value, border=True)


def _render_run_notes(artifact: Artifact) -> None:
    """How to read the Results section. Shown under the hood, not in the frame."""
    st.markdown("**How the run went**")
    st.caption(artifact.details)
    timing_line = _timing_line(artifact)
    if timing_line:
        st.caption(timing_line)
    for paragraph in artifact.summary:
        st.write(paragraph)
    if artifact.mode is LearningMode.NOVEL:
        st.caption(
            "Novel mode refreshes BatchNorm statistics from the training images before "
            "scoring the holdout, so validation matches the normalization a new photo will "
            "see. Training accuracy is still measured batch by batch."
        )
    else:
        st.caption(
            "Transfer mode keeps the backbone, including its BatchNorm statistics, fixed. "
            "Only the final layer changes."
        )
    st.caption(
        "Loss is the mean cross-entropy per image. Accuracy is the fraction labeled correctly. "
        "The Results chart puts training and validation on shared axes, so a gap is easy to see."
    )
    st.caption(
        "In the confusion matrix, rows are the true class and columns are the predicted class. "
        "Per-class accuracy counts how many held-out images of each class the model labeled "
        "correctly."
    )


def _timing_line(artifact: Artifact) -> str:
    """Wall-clock line for the run. Empty when the timer did not move."""
    timing = summarize_timing(artifact.result.history)
    if timing.images_per_second is None:
        return ""
    name = TRAINING_DEVICE_LABELS.get(artifact.device, artifact.device)
    text = (
        f"Measured on {name}: the training pass averaged {timing.mean_train_seconds:.2f} s "
        f"per epoch, {timing.images_per_second:.1f} images/s. That includes loading the "
        "images and the optimizer step, and it leaves out the validation pass."
    )
    if timing.peak_memory_bytes:
        text += (
            " Peak CUDA memory on the training side of an epoch: "
            f"{format_bytes(timing.peak_memory_bytes)}."
        )
    return text


def _render_class_table(artifact: Artifact) -> None:
    names: list[str] = []
    supports: list[int] = []
    corrects: list[int] = []
    accuracies: list[str] = []
    for index, name in enumerate(artifact.class_names):
        row = artifact.result.confusion[index]
        accuracy = artifact.result.per_class_accuracy[index]
        names.append(name)
        supports.append(sum(row))
        corrects.append(row[index])
        accuracies.append("—" if accuracy != accuracy else f"{accuracy:.0%}")
    st.table(
        {
            "Class": names,
            "Validation images": supports,
            "Correct": corrects,
            "Accuracy": accuracies,
        }
    )


def _render_export(artifact: Artifact, precision: Precision, reason: str | None) -> None:
    st.markdown(_section_head("Export", f"Core ML · {precision}"), unsafe_allow_html=True)
    key = (artifact.signature, precision)
    exports = st.session_state.setdefault("coreml_exports", {})
    convert_column, download_column = st.columns((1, 1.25), vertical_alignment="center")
    with convert_column:
        clicked = st.button(
            "Convert to Core ML",
            icon=":material/sync:",
            help="Precision is set under Hardware in the sidebar.",
        )
    if clicked and key not in exports:
        try:
            with st.spinner("Converting the model to Core ML..."):
                exports[key] = export_model(
                    artifact.model, artifact.class_names, artifact.val_images, precision
                )
        except Exception as exc:
            st.error(f"Could not convert the model to Core ML: {exc}")
    exported = exports.get(key)
    if isinstance(exported, Export):
        with download_column:
            st.download_button(
                "Download Core ML package",
                exported.zip_bytes,
                file_name="image-classifier-lab.mlpackage.zip",
                mime="application/zip",
                icon=":material/download:",
            )
        st.write(f"Package size ({precision}): {format_bytes(exported.size_bytes)}")
        render_verification(exported)
    if reason:
        st.info(f"{reason} You can still convert and download the package.")


def _render_prediction(artifact: Artifact) -> None:
    st.markdown(_section_head("Try an image"), unsafe_allow_html=True)
    upload = st.file_uploader(
        "Image to classify",
        type=UPLOAD_TYPES,
        key="prediction_file",
        label_visibility="collapsed",
    )
    if upload is None:
        return
    try:
        image = open_uploaded_file(upload.getvalue(), upload.name)
    except DatasetError as exc:
        st.error(str(exc))
        return
    try:
        probabilities = predict_probabilities(
            artifact.model,
            image_to_model_input(image),
            artifact.device,
        )
    except RuntimeError as exc:
        st.error(f"Could not run the model on {artifact.device}. Train again. ({exc})")
        return
    if len(probabilities) != len(artifact.class_names):
        st.error("The saved model does not match these classes. Train again.")
        return
    winner = max(range(len(probabilities)), key=probabilities.__getitem__)
    percent = probabilities[winner] * 100
    left, right = st.columns((1, 1.3), gap="large", vertical_alignment="center")
    with left, st.container(key="prediction-result"):
        st.image(image, width=240)
        st.markdown(f"**{artifact.class_names[winner]}**")
        st.markdown(f"{percent:.1f}% confidence")
    with right:
        _show_figure(plot_probabilities(artifact.class_names, probabilities))


def _show_figure(figure: Figure) -> None:
    st.pyplot(figure)
    plt.close(figure)
