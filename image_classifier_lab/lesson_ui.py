"""Draw the curriculum. The training lab stays in ``image_classifier_lab.ui``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import matplotlib.pyplot as plt
import streamlit as st
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import nn

from image_classifier_lab.coreml import (
    COMPUTE_UNITS,
    MEASURE_RUNS,
    WARMUP_RUNS,
    Export,
    Precision,
    benchmark_export,
)
from image_classifier_lab.curriculum import (
    LAB_PAGE,
    SECTION_BY_KEY,
    SECTIONS,
    Block,
    Exercise,
    Section,
    keep_sentence,
    visual_names,
)
from image_classifier_lab.data import (
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    describe_split,
    image_to_model_input,
    prepare_image,
    split_dataset,
)
from image_classifier_lab.demos import (
    NEURON_BIAS,
    NEURON_INPUTS,
    NEURON_WEIGHTS,
    TOY_CONFUSION,
    TOY_NAMES,
    VERTICAL_EDGE,
    confusion_reading,
    grayscale,
    interesting_crop,
    neuron,
    pixel_pipeline,
    softmax,
)
from image_classifier_lab.diagrams import (
    conv1_feature_maps,
    display_filters,
    novel_conv1_weight,
    plot_bars,
    plot_batch_layout,
    plot_channel_planes,
    plot_convolution,
    plot_cross_entropy,
    plot_descent,
    plot_epoch_strip,
    plot_feature_maps,
    plot_filter_tiles,
    plot_memory_parts,
    plot_neuron,
    plot_normalization,
    plot_number_grid,
    plot_overfit_sketch,
    plot_residual,
    plot_softmax,
    varying_indexes,
)
from image_classifier_lab.errors import TrainingError
from image_classifier_lab.examples import make_shape_dataset
from image_classifier_lab.measure import (
    MEASURE_NOTES,
    Analysis,
    RunRecord,
    adam_step_defaults,
    analysis_for_mode,
    format_bytes,
    format_count,
    group_label,
    resnet18_structure,
)
from image_classifier_lab.model import LearningMode, architecture_parameter_counts
from image_classifier_lab.plots import VAL_COLOR, plot_confusion
from image_classifier_lab.training import TrainResult, resolve_batching

PAGE_KEY = "curriculum_page"
VISITED_KEY = "curriculum_visited"
LAST_SECTION_KEY = "curriculum_last_section"
DONE_KEY = "curriculum_done"
RUN_LOG_KEY = "run_log"
_RUN_LIMIT = 8
_EXERCISE_IDS = frozenset(key for section in SECTIONS for key, _ in section.exercise_items())


class TrainedRun(Protocol):
    """The fields of a finished run the lessons read. The lab owns the object."""

    model: torch.nn.Module
    class_names: tuple[str, ...]
    result: TrainResult
    device: str
    signature: tuple[object, ...]
    mode: LearningMode
    details: str
    val_images: tuple[Image.Image, ...]


@dataclass(frozen=True)
class LabSnapshot:
    """What the lessons need from the sidebar and the latest run."""

    class_count: int
    source: str
    mode: LearningMode
    batch_size: int
    val_percent: int
    seed: int
    device_label: str
    device: str
    signature: tuple[object, ...] | None
    trained: TrainedRun | None
    inference_backends: tuple[str, ...]
    precision: Precision


def run_log() -> tuple[RunRecord, ...]:
    stored = st.session_state.get(RUN_LOG_KEY, ())
    return tuple(item for item in stored if isinstance(item, RunRecord))


def append_run(record: RunRecord) -> None:
    """Keep the latest runs so an exercise can compare them."""
    st.session_state[RUN_LOG_KEY] = (*run_log(), record)[-_RUN_LIMIT:]


def render_run_log() -> None:
    """The recent-runs tables. Empty until the first successful training."""
    records = run_log()
    if not records:
        st.caption("No runs yet. After you train, the last few results stay in this table.")
        return
    st.caption("Newest run last. The charts above still show only the latest one.")
    st.table(
        {
            "Mode": [item.mode for item in records],
            "LR": [f"{item.learning_rate:.4g}" for item in records],
            "Batch": [item.batch_size for item in records],
            "Epochs": [item.epochs for item in records],
            "Augment": ["on" if item.augment else "off" for item in records],
            "Device": [item.device_label for item in records],
            "Train loss": [f"{item.final_train_loss:.3f}" for item in records],
            "Val loss": [f"{item.final_val_loss:.3f}" for item in records],
            "Val acc.": [f"{item.final_val_accuracy:.0%}" for item in records],
        }
    )
    timing: dict[str, list[object]] = {
        "Mode": [item.mode for item in records],
        "Batch": [item.batch_size for item in records],
        "Device": [item.device_label for item in records],
        "Train s/epoch": [_seconds(item.mean_train_seconds) for item in records],
        "Val s/epoch": [_seconds(item.mean_val_seconds) for item in records],
        "Images/s": [_rate(item.images_per_second) for item in records],
    }
    if any(item.peak_memory_bytes for item in records):
        timing["CUDA peak"] = [
            format_bytes(item.peak_memory_bytes) if item.peak_memory_bytes else "—"
            for item in records
        ]
    st.table(timing)
    st.caption(
        "Images per second counts training images scored, divided by the training "
        "pass. That pass includes loading the images and the optimizer step. "
        "Validation seconds are the holdout pass and are not in the images/s figure."
    )


def render_curriculum(snapshot: LabSnapshot) -> None:
    """Navigator plus the current lesson. The lab is rendered separately, below."""
    _sync_exercise_done()
    page = _current_page()
    if page != LAB_PAGE:
        _visit(page)
        st.session_state[LAST_SECTION_KEY] = page
    _navigator(page)
    if page == LAB_PAGE:
        last = st.session_state.get(LAST_SECTION_KEY)
        if isinstance(last, str) and last in SECTION_BY_KEY:
            section = SECTION_BY_KEY[last]
            st.caption(
                f"Training lab. Section {section.number}, {section.title}, is one of the "
                "buttons above."
            )
        else:
            st.caption("Training lab. The lesson buttons are above.")
        return
    _render_section(SECTION_BY_KEY[page], snapshot)


def _current_page() -> str:
    if PAGE_KEY not in st.session_state:
        st.session_state[PAGE_KEY] = SECTIONS[0].key
    page = st.session_state.get(PAGE_KEY, SECTIONS[0].key)
    if page == LAB_PAGE or page in SECTION_BY_KEY:
        return str(page)
    return SECTIONS[0].key


def open_page(page: str) -> None:
    """Select a lesson or the training lab. Used by the navigator buttons."""
    st.session_state[PAGE_KEY] = page


def _exercise_done() -> set[str]:
    """IDs whose exercise checkbox is checked.

    The checkbox widget is only created for the open lesson. Streamlit drops a
    widget key that was not rendered, so the durable record is this set.
    """
    stored = st.session_state.get(DONE_KEY)
    if not isinstance(stored, set):
        stored = set()
        st.session_state[DONE_KEY] = stored
    return stored


def _sync_exercise_done() -> None:
    """Copy any checkbox that is still in state into the durable set."""
    done = _exercise_done()
    for exercise_id in _EXERCISE_IDS:
        widget_key = f"done-{exercise_id}"
        if widget_key not in st.session_state:
            continue
        if st.session_state[widget_key]:
            done.add(exercise_id)
        else:
            done.discard(exercise_id)


def _visit(page: str) -> None:
    opened = set(st.session_state.get(VISITED_KEY, ()))
    opened.add(page)
    st.session_state[VISITED_KEY] = opened


def _navigator(page: str) -> None:
    total = len(SECTIONS)
    opened = len(set(st.session_state.get(VISITED_KEY, ())) & {item.key for item in SECTIONS})
    completed = _exercise_done()
    done = len(completed & _EXERCISE_IDS)
    st.progress(opened / total, text=f"{opened} of {total} sections opened")
    st.caption(f"{done} of {len(_EXERCISE_IDS)} exercises marked done.")
    previous, following, lab = st.columns([1, 1, 2])
    if page == LAB_PAGE:
        last = st.session_state.get(LAST_SECTION_KEY)
        if isinstance(last, str) and last in SECTION_BY_KEY:
            section = SECTION_BY_KEY[last]
            previous.button(
                f"Back to {section.number}",
                key="nav-back",
                on_click=open_page,
                args=(section.key,),
                use_container_width=True,
            )
    else:
        index = next(i for i, item in enumerate(SECTIONS) if item.key == page)
        previous_key = SECTIONS[max(index - 1, 0)].key
        next_key = SECTIONS[min(index + 1, total - 1)].key
        previous.button(
            "Previous",
            key="nav-prev",
            on_click=open_page,
            args=(previous_key,),
            disabled=index == 0,
            use_container_width=True,
        )
        following.button(
            "Next",
            key="nav-next",
            on_click=open_page,
            args=(next_key,),
            disabled=index == total - 1,
            use_container_width=True,
        )
    lab.button(
        "Training lab",
        key="nav-lab",
        type="primary" if page == LAB_PAGE else "secondary",
        on_click=open_page,
        args=(LAB_PAGE,),
        use_container_width=True,
    )
    columns = st.columns(total)
    for column, section in zip(columns, SECTIONS, strict=True):
        mark = "✓ " if all(key in completed for key, _ in section.exercise_items()) else ""
        column.button(
            f"{mark}{section.number} {section.short}",
            key=f"nav-{section.key}",
            help=section.title,
            type="primary" if page == section.key else "secondary",
            on_click=open_page,
            args=(section.key,),
            use_container_width=True,
        )


def _render_section(section: Section, snapshot: LabSnapshot) -> None:
    st.subheader(f"{section.number} of {len(SECTIONS)}  ·  {section.title}")
    with st.container(border=True):
        st.markdown("**Learning objectives**")
        for objective in section.objectives:
            st.markdown(f"- {objective}")
    for block in section.blocks:
        _render_block(block, snapshot)
    for exercise_id, exercise in section.exercise_items():
        _render_exercise(exercise, exercise_id)
    st.caption(
        "The training lab is on this page, below this lesson. The Training lab button "
        "hides the lesson so the lab sits at the top."
    )


def _render_block(block: Block, snapshot: LabSnapshot) -> None:
    if block.heading:
        st.markdown(f"**{block.heading}**")
    for paragraph in block.paragraphs:
        st.write(paragraph)
    if block.formula:
        st.latex(block.formula)
    if not block.visual:
        return
    renderer = VISUAL_RENDERERS.get(block.visual)
    if renderer is None:
        raise KeyError(f"no diagram named {block.visual}")
    renderer(snapshot)


def _render_exercise(exercise: Exercise, exercise_id: str) -> None:
    with st.container(border=True):
        st.markdown("**Exercise**")
        st.markdown("What to do")
        st.write(exercise.instructions)
        st.markdown("Leave at the defaults")
        st.write(keep_sentence(exercise.changed))
        st.markdown("What to look for")
        st.write(exercise.look_for)
        st.markdown("Check")
        st.write(exercise.question)
        with st.expander("Show the check"):
            st.write(exercise.answer)
        widget_key = f"done-{exercise_id}"
        if widget_key not in st.session_state:
            st.session_state[widget_key] = exercise_id in _exercise_done()
        st.checkbox("Mark this exercise done", key=widget_key)


def _show(figure: Figure) -> None:
    try:
        st.pyplot(figure)
    finally:
        plt.close(figure)


def _teaching_image():
    cached = st.session_state.get("_teaching_circle")
    if cached is not None:
        return cached
    image = prepare_image(make_shape_dataset()[0].images[0])
    st.session_state["_teaching_circle"] = image
    return image


def _run_matches(snapshot: LabSnapshot) -> bool:
    trained = snapshot.trained
    if trained is None or snapshot.signature is None:
        return False
    return snapshot.signature == trained.signature


def _channels(_snapshot: LabSnapshot) -> None:
    image = _teaching_image()
    _show(plot_channel_planes(image))
    crop, origin = interesting_crop(image, 8)
    _show(plot_number_grid(crop))
    st.caption(
        f"An 8×8 patch of the red channel, top-left at row {origin[0]}, "
        f"column {origin[1]} of the resized drawing. The numbers are 0–255, "
        "before scaling and normalization."
    )


def _tensor(_snapshot: LabSnapshot) -> None:
    _show(plot_batch_layout(IMAGE_SIZE))


def _normalize(_snapshot: LabSnapshot) -> None:
    image = _teaching_image()
    crop, origin = interesting_crop(image, 8)
    index = int(
        st.slider("Pixel in the 8×8 patch", min_value=0, max_value=63, value=0, key="norm-pixel")
    )
    row, column = divmod(index, 8)
    rgb = crop[row][column]
    unit, normalized = pixel_pipeline(rgb)
    scaled, shifted = plot_normalization(rgb, unit, normalized)
    _show(scaled)
    _show(shifted)
    st.caption(
        f"This pixel is row {origin[0] + row}, column {origin[1] + column}. "
        f"ImageNet mean {IMAGENET_MEAN}. ImageNet standard deviation {IMAGENET_STD}."
    )


def _neuron(_snapshot: LabSnapshot) -> None:
    bias = float(
        st.slider(
            "Bias",
            min_value=-1.0,
            max_value=1.0,
            value=float(NEURON_BIAS),
            step=0.1,
            key="neuron-bias",
        )
    )
    _show(plot_neuron(NEURON_INPUTS, NEURON_WEIGHTS, bias))
    parts, total = neuron(NEURON_INPUTS, NEURON_WEIGHTS, bias)
    pretty = ", ".join(f"{value:.2f}" for value in parts)
    st.caption(
        f"Inputs {NEURON_INPUTS}, weights {NEURON_WEIGHTS}. Products: {pretty}. Output {total:.2f}."
    )


def _final_layer(snapshot: LabSnapshot) -> None:
    analysis = resnet18_structure(snapshot.class_count)
    if analysis.final_in_features is None or analysis.final_out_features is None:
        st.error("This model has no final linear layer to describe.")
        return
    weights = analysis.final_in_features * analysis.final_out_features
    head = weights + analysis.final_out_features
    trainable, total = architecture_parameter_counts(snapshot.class_count, snapshot.mode.value)
    st.markdown(
        f"Final layer, read from this ResNet: **{analysis.final_in_features}** inputs, "
        f"**{analysis.final_out_features}** scores, **{head:,}** parameters "
        f"({analysis.final_in_features} × {analysis.final_out_features} weights "
        f"+ {analysis.final_out_features} biases)."
    )
    st.caption(
        f"Sidebar mode is {snapshot.mode.title}: {trainable:,} trainable of {total:,}, "
        f"for {snapshot.class_count} classes. Built-in shapes use 3 classes. "
        "An upload uses however many classes are on the form."
    )


def _softmax(_snapshot: LabSnapshot) -> None:
    columns = st.columns(3)
    scores = tuple(
        float(
            columns[index].slider(
                f"Score {index + 1}",
                min_value=-5.0,
                max_value=5.0,
                value=value,
                step=0.1,
                key=f"softmax-{index}",
            )
        )
        for index, value in enumerate((2.0, 0.5, -1.0))
    )
    _show(plot_softmax(scores))
    shares = softmax(scores)
    listed = ", ".join(f"{share:.1%}" for share in shares)
    st.caption(f"Shares: {listed}. Sum {sum(shares):.3f}.")


def _loss(_snapshot: LabSnapshot) -> None:
    _show(plot_cross_entropy())


def _descent(_snapshot: LabSnapshot) -> None:
    step = float(
        st.select_slider(
            "Toy step size",
            options=[0.05, 0.10, 0.25, 0.50, 1.00],
            value=0.10,
            key="toy-step",
        )
    )
    _show(plot_descent(step))
    st.caption(
        "The dot starts at weight −1. The bottom of the bowl is weight 2. "
        "This slider does not change the learning rate in the sidebar, and the "
        "update is plain gradient descent, not Adam."
    )


def _adam(_snapshot: LabSnapshot) -> None:
    beta1, beta2, decay = adam_step_defaults()
    st.caption(
        f"A stock Adam in this installation uses beta1 {beta1:g}, beta2 {beta2:g}, "
        f"and weight decay {decay:g}. The sidebar changes the learning rate and "
        "leaves the rest of Adam at those defaults."
    )


def _convolution(_snapshot: LabSnapshot) -> None:
    grid = grayscale(_teaching_image(), 28)
    limit = len(grid) - len(VERTICAL_EDGE)
    top = int(st.slider("Window row", min_value=0, max_value=limit, value=8, key="conv-row"))
    left = int(st.slider("Window column", min_value=0, max_value=limit, value=8, key="conv-col"))
    _show(plot_convolution(grid, top, left))
    st.caption(
        "A hand-built Sobel filter on one built-in drawing, reduced to a 28×28 "
        "luminance grid. The filter is not a weight from the network."
    )


def _residual(_snapshot: LabSnapshot) -> None:
    _show(plot_residual())


def _stages(snapshot: LabSnapshot) -> None:
    analysis = analysis_for_mode(snapshot.class_count, snapshot.mode, 1)
    shapes = dict(analysis.group_shapes)
    labels, values = _groups(analysis.group_parameters, shapes)
    _show(
        plot_bars(
            labels,
            values,
            xlabel="Parameters",
            title="Parameters in each part of this ResNet-18",
        )
    )
    mac_labels, mac_values = _groups(analysis.macs_by_group, shapes)
    _show(
        plot_bars(
            mac_labels,
            mac_values,
            xlabel="Multiply-adds for one image",
            title="Forward multiply-adds by part",
            color=VAL_COLOR,
        )
    )
    if analysis.stem_kernel is not None:
        st.caption(
            f"First convolution, read from the model: kernel {analysis.stem_kernel[0]}×"
            f"{analysis.stem_kernel[1]}, stride {analysis.stem_stride}, padding "
            f"{analysis.stem_padding}, {analysis.stem_in_channels} input channels, "
            f"{analysis.stem_out_channels} filters. One image is "
            f"{format_count(analysis.macs)} multiply-adds "
            f"({format_count(analysis.flops)} FLOPs, two per multiply-add)."
        )
    weight, caption = _filter_source(snapshot)
    _show(plot_filter_tiles(display_filters(weight, 8)))
    maps = conv1_feature_maps(weight, image_to_model_input(_teaching_image()))
    chosen = maps[varying_indexes(maps, 8)]
    _show(plot_feature_maps(chosen))
    st.caption(caption)


def _groups(
    pairs: Sequence[tuple[str, int]],
    shapes: dict[str, tuple[int, ...]],
) -> tuple[list[str], list[int]]:
    labels: list[str] = []
    values: list[int] = []
    for name, count in pairs:
        if count <= 0:
            continue
        labels.append(group_label(name, shapes.get(name)))
        values.append(count)
    return labels, values


def _filter_source(snapshot: LabSnapshot) -> tuple[torch.Tensor, str]:
    trained = snapshot.trained
    if trained is None or not hasattr(trained.model, "conv1"):
        return novel_conv1_weight(), (
            "No trained model yet. These are random first-layer filters from a seeded "
            "novel-mode network, before any training. Each filter is scaled on its own "
            "so the picture is visible; the network does not do that scaling. The "
            "feature maps are this layer applied to a built-in circle. Train in "
            "Transfer mode and come back to see the ImageNet filters, which stay frozen."
        )
    weight = trained.model.conv1.weight.detach().float().cpu().clone()
    stale = ""
    if not _run_matches(snapshot):
        stale = (
            " The sidebar has changed since this run, so the filters belong to the previous one."
        )
    if trained.mode is LearningMode.TRANSFER:
        text = (
            "First-layer filters from your latest Transfer run. That layer came from "
            "the ImageNet weights and this run did not update it. The feature maps are "
            "that layer applied to a built-in circle, so the picture stays comparable. "
            "Each filter is scaled on its own for display."
        )
    else:
        text = (
            "First-layer filters from your latest Novel run. They started random. "
            "The feature maps are that layer applied to a built-in circle. Each filter "
            "is scaled on its own for display."
        )
    return weight, text + stale


def _epoch_strip(snapshot: LabSnapshot) -> None:
    split = split_dataset(make_shape_dataset(), snapshot.val_percent / 100, snapshot.seed)
    try:
        plan = resolve_batching(len(split.train_images), snapshot.batch_size, batchnorm=True)
    except TrainingError as exc:
        st.error(str(exc))
        return
    _show(
        plot_epoch_strip(
            len(split.train_images),
            snapshot.batch_size,
            drop_last=plan.drop_last,
        )
    )
    extra = plan.note or "Every training image is in a batch of at least two."
    st.caption(
        f"Built-in shapes, seed {snapshot.seed}, holdout {snapshot.val_percent}%: "
        f"{len(split.train_images)} training images, {len(split.val_images)} validation. "
        f"{describe_split(split)}. {extra}"
    )


def _overfit(_snapshot: LabSnapshot) -> None:
    _show(plot_overfit_sketch())
    st.caption("Invented curves, to show the shape. The measured curves are in the training lab.")


def _modes(snapshot: LabSnapshot) -> None:
    counts = {
        mode: architecture_parameter_counts(snapshot.class_count, mode.value)
        for mode in (LearningMode.TRANSFER, LearningMode.NOVEL)
    }
    total = counts[LearningMode.NOVEL][1]
    labels = [mode.title for mode in counts]
    percents = [100.0 * trainable / total for trainable, _total in counts.values()]
    _show(
        plot_bars(
            labels,
            percents,
            xlabel="Percent of parameters that receive gradients",
            title=f"What is allowed to learn, {snapshot.class_count} classes",
        )
    )
    for mode, (trainable, overall) in counts.items():
        marker = "Sidebar" if mode is snapshot.mode else "The other mode"
        st.caption(f"{marker}, {mode.title}: {trainable:,} trainable of {overall:,}.")


def _batchnorm(_snapshot: LabSnapshot) -> None:
    state = torch.get_rng_state()
    try:
        momentum = float(nn.BatchNorm2d(1).momentum or 0.0)
    finally:
        torch.set_rng_state(state)
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("**Transfer**")
        st.write(
            "BatchNorm stays in evaluation. The stored ImageNet mean and variance "
            "are what every batch uses, and they are not updated."
        )
    with right, st.container(border=True):
        st.markdown("**Novel**")
        st.write(
            "BatchNorm trains with the batches. After each epoch the stored mean and "
            "variance are recomputed from the training images. That pass does not "
            "change the weights."
        )
    st.caption(
        f"PyTorch's BatchNorm momentum defaults to {momentum:g} in this installation. "
        "On a handful of images that default barely moves the stored averages, which "
        "is why Novel mode recomputes them."
    )


def _evaluation(snapshot: LabSnapshot) -> None:
    figure = plot_confusion(TOY_CONFUSION, TOY_NAMES)
    figure.axes[0].set_title("Reading practice — not a run")
    _show(figure)
    for line in confusion_reading(TOY_CONFUSION, TOY_NAMES):
        st.write(line)
    st.caption(
        "Those counts were chosen so the arithmetic is easy. They are not a result from the lab."
    )
    trained = snapshot.trained
    if trained is None:
        st.info("Train in the lab and your confusion matrix will appear here.")
        return
    if not _run_matches(snapshot):
        st.warning("The matrix below is the previous run. The images or settings have changed.")
    st.markdown("**Your latest run**")
    names = trained.class_names
    confusion = trained.result.confusion
    _show(plot_confusion(confusion, names))
    for line in confusion_reading(confusion, names):
        st.write(line)


def _hardware(snapshot: LabSnapshot) -> None:
    analysis = _cost(snapshot)
    st.markdown(
        f"**{format_count(analysis.macs)}** multiply-adds in one forward pass, "
        f"which is **{format_count(analysis.flops)}** FLOPs when each multiply-add "
        f"counts as two. Batch {analysis.batch}, {snapshot.class_count} classes, "
        f"{snapshot.mode.title}."
    )
    labels = [
        "Parameters",
        "Gradients",
        "Adam moments",
        "Buffers",
        "Forward outputs (parts list)",
    ]
    values = [
        analysis.parameter_bytes,
        analysis.gradient_bytes,
        analysis.adam_bytes,
        analysis.buffer_bytes,
        analysis.output_bytes_total,
    ]
    _show(plot_memory_parts(labels, values, parts_list=(False, False, False, False, True)))
    st.caption(
        "The blue bars are sizes of tensors that exist for the whole step. The orange "
        "bar adds up forward outputs that are not all stored at once."
    )
    kinds = ", ".join(
        f"{name} {format_bytes(size)}" for name, size in analysis.output_bytes_by_kind
    )
    st.caption(f"Forward outputs by layer type: {kinds}.")
    with st.expander("What these counts include"):
        for note in MEASURE_NOTES:
            st.write(note)


def _devices(snapshot: LabSnapshot) -> None:
    st.write(f"Training device in the sidebar: {snapshot.device_label}.")
    render_run_log()
    if any(item.peak_memory_bytes for item in run_log()):
        st.caption(
            "CUDA peak is the allocator's high-water mark on the training side of the "
            "epoch, including a Novel-mode BatchNorm refresh and not the validation pass. "
            "CPU and MPS runs leave that cell empty."
        )


def render_verification(exported: Export) -> None:
    """Show the held-out comparison for the selected package."""
    if exported.verification is not None:
        result = exported.verification
        st.write(
            f"Against float32 PyTorch on {result.image_count} validation images: "
            f"top-1 agreement {result.top1_agreement:.1%}; "
            f"max absolute probability difference {result.max_abs_difference:.6g}."
        )
    if exported.verification_error is not None:
        st.error(f"Could not verify the Core ML model: {exported.verification_error}")


def _on_device(snapshot: LabSnapshot) -> None:
    trained = snapshot.trained
    total = (
        trained.result.total_parameters
        if trained is not None
        else resnet18_structure(snapshot.class_count).total_parameters
    )
    st.write(
        f"{total:,} parameters: float32 weights take {format_bytes(total * 4)}; "
        f"float16 weights take {format_bytes(total * 2)}. "
        "Packages also contain metadata and other tensors."
    )
    if trained is None:
        st.info("Train a model to time inference. Export it in the lab to measure package size.")
        return
    if not _run_matches(snapshot):
        st.warning("These measurements use the previous run; the lab settings have changed.")
    precision = snapshot.precision
    backends = snapshot.inference_backends
    exports = st.session_state.get("coreml_exports", {})
    exported = exports.get((trained.signature, precision))
    if exported is not None:
        st.write(f"Measured {precision} package: {format_bytes(exported.size_bytes)}")
        render_verification(exported)
    else:
        st.info(f"Convert to Core ML in the lab to measure the {precision} package.")
    if not backends:
        st.info("Select an inference backend in Hardware to measure it.")
        return
    if exported is None and any(backend in COMPUTE_UNITS for backend in backends):
        return
    key = (trained.signature, precision, backends)
    benchmarks = st.session_state.setdefault("coreml_benchmarks", {})
    if st.button("Measure on-device inference"):
        try:
            with st.spinner("Measuring batch-one inference..."):
                benchmarks[key] = benchmark_export(
                    exported, trained.model, trained.val_images[0], backends=backends
                )
        except Exception as exc:
            st.error(f"Could not measure inference: {exc}")
    if key in benchmarks:
        timings = benchmarks[key]
        table = {
            "Device": [item.name for item in timings],
            "Median ms": [
                f"{item.median_ms:.3f}" if item.median_ms is not None else "Unavailable"
                for item in timings
            ],
        }
        if any(item.error for item in timings):
            table["Error"] = [item.error or "" for item in timings]
        st.table(table)
        st.caption(
            f"Batch 1, {WARMUP_RUNS} warmup runs, {MEASURE_RUNS} measured runs. "
            f"Core ML uses the selected export precision ({precision}); PyTorch uses float32."
        )


def _cost(snapshot: LabSnapshot) -> Analysis:
    return analysis_for_mode(snapshot.class_count, snapshot.mode, snapshot.batch_size)


def _seconds(value: float) -> str:
    if value <= 0:
        return "—"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def _rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.1f}"


VISUAL_RENDERERS = {
    "channels": _channels,
    "tensor": _tensor,
    "normalize": _normalize,
    "neuron": _neuron,
    "final_layer": _final_layer,
    "softmax": _softmax,
    "loss": _loss,
    "descent": _descent,
    "adam": _adam,
    "convolution": _convolution,
    "residual": _residual,
    "stages": _stages,
    "epoch_strip": _epoch_strip,
    "overfit": _overfit,
    "modes": _modes,
    "batchnorm": _batchnorm,
    "evaluation": _evaluation,
    "hardware": _hardware,
    "devices": _devices,
    "on_device": _on_device,
}


def renderers_cover_the_lessons() -> bool:
    """True when every diagram the curriculum names has a renderer."""
    return set(VISUAL_RENDERERS) == set(visual_names())
