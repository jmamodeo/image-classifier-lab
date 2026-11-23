"""Streamlit interface for Lessons in AI from an Image Classifier."""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass

import matplotlib.pyplot as plt
import streamlit as st
import torch
from matplotlib.figure import Figure
from PIL import Image

from image_classifier_lab import __version__
from image_classifier_lab.coreml import Export, Precision, export_model
from image_classifier_lab.data import (
    MAX_CLASSES,
    MIN_CLASSES,
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
from image_classifier_lab.examples import EXAMPLE_IMAGES_PER_CLASS, make_shape_dataset
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
UPLOAD_TYPES = ["png", "jpg", "jpeg", "webp", "bmp"]
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


def main() -> None:
    """Run the app. Use ``streamlit run main.py``."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_state()
    settings = _sidebar()
    st.title(APP_TITLE)
    st.caption(
        "Eight lessons on how this image classifier learns, and a training lab that "
        "stays on the page. The exercises name the settings "
        "to change. Transfer mode starts from ImageNet and trains a new final layer. "
        "Novel mode trains every layer from random weights."
    )
    lesson_slot = st.container()
    lab_slot = st.container()
    with lab_slot:
        snapshot = _render_workspace(settings)
    with lesson_slot:
        render_curriculum(snapshot)


def _render_workspace(settings: Settings) -> LabSnapshot:
    """The images, the run, and the prediction. Always on the page, under the lesson."""
    st.markdown('<div id="training-lab"></div>', unsafe_allow_html=True)
    st.subheader("Training lab")
    st.subheader("Images")
    source = st.radio(
        "Where the images come from",
        options=["upload", "example"],
        format_func=lambda value: "Upload images" if value == "upload" else "Built-in shapes",
        horizontal=True,
        key="source",
    )
    groups, read_error = _load_groups(source)
    if read_error is not None:
        st.error(read_error)

    class_count = len(groups) if groups is not None else _slot_count(source)
    _render_mode_cards(class_count, settings.mode)

    split: SplitData | None = None
    if groups is not None:
        split, problem = _try_split(groups, settings)
        if problem is not None:
            level, message = problem
            if level == "info":
                st.info(message)
            else:
                st.error(message)
        elif split is not None:
            _render_preview(split, settings)

    batch_note = _batch_note(split, settings) if split is not None else None
    if isinstance(batch_note, str) and batch_note.startswith("error:"):
        st.error(batch_note.removeprefix("error:"))
        split = None
        batch_note = None
    elif batch_note:
        st.caption(batch_note)

    if st.button("Train model", type="primary", disabled=split is None) and split is not None:
        if groups is None:
            st.error("Add images before training.")
        else:
            artifact = _run_training(split, groups, settings, batch_note)
            if artifact is not None:
                st.session_state.artifact = artifact
                st.session_state.coreml_exports = {}
                st.session_state.coreml_benchmarks = {}

    stored = st.session_state.get("artifact")
    trained = stored if isinstance(stored, Artifact) else None
    signature = _signature(settings, groups)
    if trained is not None:
        _render_results(trained, signature)
        _render_export(trained, settings.precision, settings.coreml_reason)
        _render_prediction(trained)
        st.markdown("**Recent runs**")
        render_run_log()

    st.divider()
    st.caption(f"{APP_TITLE} {__version__} · John Amodeo · MIT (code) · CC BY 4.0 (lessons)")
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


def _ensure_state() -> None:
    if "class_ids" not in st.session_state:
        st.session_state.class_ids = [1, 2]
        st.session_state.next_class_id = 3


def _slot_count(source: str) -> int:
    if source == "example":
        return 3
    return len(st.session_state.class_ids)


def _sidebar() -> Settings:
    concepts = CONCEPTS
    st.sidebar.header("Settings")
    st.sidebar.caption("The lessons name which of these to move, and which to leave.")
    hardware = detect_hardware()
    with st.sidebar.container(border=True):
        st.subheader("Hardware")
        devices = {option.key: option.label for option in hardware.training if option.available}
        device = st.radio(
            "Training device", options=tuple(devices), format_func=devices.__getitem__
        )
        for option in hardware.training:
            if option.reason is not None:
                st.checkbox(option.label, disabled=True, key=f"training-{option.key}")
                st.caption(option.reason)
        st.caption("Neural Engine: cannot train. Core ML runs it for inference only.")

        st.markdown("**Inference backends**")
        backends = []
        for option in hardware.inference:
            checked = st.checkbox(
                option.label,
                value=option.available,
                disabled=not option.available,
                key=f"inference-{option.key}",
            )
            if option.reason is not None:
                st.caption(option.reason)
                if (
                    option.key == "CPU_AND_NE"
                    and not hardware.apple_silicon
                    and option.reason != NEURAL_ENGINE_REASON
                ):
                    st.caption(NEURAL_ENGINE_REASON)
            if checked and option.available:
                backends.append(option.key)
        precision: Precision = st.radio(
            "Precision",
            options=("float32", "float16"),
            horizontal=True,
            key="coreml_precision",
            help="Applies to the Core ML export and Lesson 8. PyTorch stays in float32.",
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


def _load_groups(source: str) -> tuple[list[LabeledImages] | None, str | None]:
    try:
        if source == "example":
            return _example_groups(), None
        return _upload_groups(), None
    except DatasetError as exc:
        return None, str(exc)


def _example_groups() -> list[LabeledImages]:
    groups = make_shape_dataset()
    st.caption(
        "Built-in drawings: circles, squares, and triangles, "
        f"{EXAMPLE_IMAGES_PER_CLASS} each. They are simple on purpose so a short run can "
        "show the curves. This is a teaching set, not a benchmark."
    )
    columns = st.columns(len(groups))
    for column, group in zip(columns, groups, strict=True):
        with column:
            st.markdown(f"**{group.label}**")
            st.image(list(group.images[:3]), width=110)
            st.caption(f"{len(group.images)} images · showing 3")
    return groups


def _upload_groups() -> list[LabeledImages]:
    st.caption(
        f"Two classes to start, and up to {MAX_CLASSES}. "
        "Each class needs a label and at least two images."
    )
    groups: list[LabeledImages] = []
    for position, class_id in enumerate(list(st.session_state.class_ids), start=1):
        with st.container(border=True):
            header = st.columns([6, 1])
            header[0].markdown(f"**Class {position}**")
            if len(st.session_state.class_ids) > MIN_CLASSES and header[1].button(
                "Remove",
                key=f"remove_{class_id}",
            ):
                _remove_class(int(class_id))
                st.rerun()
            label = st.text_input("Label", key=f"label_{class_id}", placeholder="e.g. daisy")
            files = st.file_uploader(
                "Images",
                key=f"files_{class_id}",
                accept_multiple_files=True,
                type=UPLOAD_TYPES,
            )
            images: list = []
            if files:
                st.caption(f"{len(files)} images" if len(files) != 1 else "1 image")
                for file in files:
                    images.append(open_uploaded_file(file.getvalue(), file.name))
            groups.append(LabeledImages(str(label).strip(), tuple(images)))
    if st.button("Add class", disabled=len(st.session_state.class_ids) >= MAX_CLASSES):
        _add_class()
        st.rerun()
    return groups


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
        columns[0].image(prepared, caption=f"Train · {train_name}", width=140)
        for index, view in enumerate(views, start=1):
            columns[index].image(view, caption=f"Draw {index}", width=140)
        columns[PREVIEW_COUNT + 1].image(validation, caption=f"Validation · {val_name}", width=140)
        return
    st.caption(
        "Augmentation is off, so every epoch sees these resized pixels. "
        "Validation is resized the same way."
    )
    left, right = st.columns(2)
    left.image(prepared, caption=f"Train · {train_name}", width=220)
    right.image(validation, caption=f"Validation · {val_name}", width=220)


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
    st.subheader("How the run went")
    if signature != artifact.signature:
        st.warning(
            "Showing the previous run. The images or settings have changed, so train again "
            "before reading this as the current setup."
        )
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
    _show_figure(plot_history(artifact.result.history))
    st.caption(
        "Loss is the mean cross-entropy per image. Accuracy is the fraction labeled correctly. "
        "Train and validation share the axes so a gap is easy to see."
    )
    left, right = st.columns((1.15, 1))
    with left:
        st.markdown("**Validation confusion matrix**")
        st.caption("Rows are the true class. Columns are the predicted class.")
        _show_figure(plot_confusion(artifact.result.confusion, artifact.class_names))
    with right:
        st.markdown("**Per-class validation accuracy**")
        st.caption("Of the held-out images of this class, how many the model labeled correctly.")
        _render_class_table(artifact)


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
    st.subheader("Export to Core ML")
    st.caption(f"Export precision: {precision} (set in Hardware).")
    key = (artifact.signature, precision)
    exports = st.session_state.setdefault("coreml_exports", {})
    if st.button("Convert to Core ML") and key not in exports:
        try:
            with st.spinner("Converting the model to Core ML..."):
                exports[key] = export_model(
                    artifact.model, artifact.class_names, artifact.val_images, precision
                )
        except Exception as exc:
            st.error(f"Could not convert the model to Core ML: {exc}")
    exported = exports.get(key)
    if isinstance(exported, Export):
        st.download_button(
            "Download Core ML package",
            exported.zip_bytes,
            file_name="image-classifier-lab.mlpackage.zip",
            mime="application/zip",
        )
        st.write(f"Package size ({precision}): {format_bytes(exported.size_bytes)}")
        render_verification(exported)
    if reason:
        st.info(f"{reason} You can still convert and download the package.")


def _render_prediction(artifact: Artifact) -> None:
    st.subheader("Try an image")
    st.caption(
        "Resize and normalize only, the same path as validation. No flip and no rotation. "
        + CONCEPTS["confidence"].summary
    )
    upload = st.file_uploader("Image to classify", type=UPLOAD_TYPES, key="prediction_file")
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
    left, right = st.columns((1, 1.3))
    with left:
        st.image(image, width=320)
        st.markdown(f"**{artifact.class_names[winner]}**")
        st.markdown(f"{percent:.1f}% confidence")
        st.caption(CONCEPTS["confidence"].summary)
    with right:
        _show_figure(plot_probabilities(artifact.class_names, probabilities))


def _show_figure(figure: Figure) -> None:
    st.pyplot(figure)
    plt.close(figure)
