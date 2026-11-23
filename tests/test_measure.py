"""Forward multiply-adds, memory parts, and the timing summary."""

from __future__ import annotations

import torch
from torch import nn

from image_classifier_lab.data import IMAGE_SIZE
from image_classifier_lab.measure import (
    MEASURE_NOTES,
    adam_bytes,
    adam_step_defaults,
    analysis_for_mode,
    analyze_forward,
    format_bytes,
    format_count,
    make_run_record,
    resnet18_structure,
    summarize_timing,
    with_batch,
)
from image_classifier_lab.model import LearningMode, architecture_parameter_counts
from image_classifier_lab.training import EpochMetrics, TrainResult


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, bias=True)
        self.fc = nn.Linear(8, 3)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(images)
        return self.fc(hidden.reshape(hidden.shape[0], -1))


def test_tiny_network_matches_a_hand_count() -> None:
    analysis = analyze_forward(Tiny(), torch.zeros(1, 1, 4, 4))
    assert analysis.macs_conv == 2 * 2 * 2 * 3 * 3 * 1
    assert analysis.macs_linear == 1 * 8 * 3
    assert analysis.macs == analysis.macs_conv + analysis.macs_linear
    assert analysis.flops == 2 * analysis.macs
    conv_out = 1 * 2 * 2 * 2 * 4
    linear_out = 1 * 3 * 4
    assert analysis.output_bytes_total == conv_out + linear_out
    assert analysis.parameter_bytes == analysis.total_parameters * 4
    assert analysis.adam_bytes == adam_bytes(analysis.trainable_parameters)


def test_resnet18_conv1_and_final_layer_match_the_shapes() -> None:
    analysis = resnet18_structure(3)
    shapes = dict(analysis.group_shapes)
    macs = dict(analysis.macs_by_group)
    assert shapes["conv1"] == (64, 112, 112)
    assert shapes["layer4"] == (512, 7, 7)
    assert shapes["fc"] == (3,)
    assert macs["conv1"] == 112 * 112 * 64 * 7 * 7 * 3
    assert macs["fc"] == 512 * 3
    assert analysis.stem_kernel == (7, 7)
    assert analysis.stem_stride == (2, 2)
    assert analysis.stem_padding == (3, 3)
    assert analysis.stem_out_channels == 64
    assert analysis.final_in_features == 512
    assert analysis.final_out_features == 3
    head = 512 * 3 + 3
    trainable, total = architecture_parameter_counts(3, LearningMode.TRANSFER.value)
    assert head == trainable
    assert analysis.total_parameters == total
    grouped = dict(analysis.group_parameters)
    assert grouped["layer4"] > grouped["layer3"] > grouped["layer2"] > grouped["layer1"]
    assert grouped["layer4"] == max(grouped.values())
    assert analysis.buffer_bytes > 0
    assert analysis.macs > macs["conv1"]


def test_batch_scales_arithmetic_and_not_parameters() -> None:
    one = analysis_for_mode(4, LearningMode.NOVEL, 1)
    four = analysis_for_mode(4, LearningMode.NOVEL, 4)
    assert four.macs == 4 * one.macs
    assert four.flops == 4 * one.flops
    assert four.output_bytes_total == 4 * one.output_bytes_total
    assert four.parameter_bytes == one.parameter_bytes
    assert four.adam_bytes == one.adam_bytes
    transfer = analysis_for_mode(4, LearningMode.TRANSFER, 4)
    assert transfer.macs == four.macs
    assert transfer.parameter_bytes == four.parameter_bytes
    assert transfer.trainable_parameters < four.trainable_parameters
    assert transfer.gradient_bytes == transfer.trainable_parameters * 4
    assert transfer.adam_bytes == 2 * transfer.trainable_parameters * 4
    assert with_batch(one, 1) is one


def test_structure_queries_do_not_touch_the_global_rng() -> None:
    resnet18_structure.cache_clear()
    torch.manual_seed(0)
    expected = torch.rand(4)
    torch.manual_seed(0)
    resnet18_structure(3)
    analysis_for_mode(5, LearningMode.TRANSFER, 2)
    assert torch.equal(torch.rand(4), expected)


def test_adam_moments_match_pytorch_and_the_notes_say_what_was_left_out() -> None:
    parameter = nn.Parameter(torch.zeros(5))
    optimizer = torch.optim.Adam([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["exp_avg"].numel() == 5
    assert state["exp_avg_sq"].numel() == 5
    assert state["exp_avg"].element_size() == 4
    assert adam_bytes(5) == 2 * 5 * 4
    beta1, beta2, decay = adam_step_defaults()
    assert (beta1, beta2, decay) == (0.9, 0.999, 0.0)
    text = " ".join(MEASURE_NOTES)
    assert "parts list" in text
    assert "Conv2d" in text
    assert "forward" in text.lower()
    assert "peak" in text.lower()


def test_timing_summary_uses_the_training_pass_only() -> None:
    history = (
        EpochMetrics(
            1, 1.0, 0.5, 1.1, 0.4, train_seconds=2.0, val_seconds=1.0, train_images=8, train_steps=2
        ),
        EpochMetrics(
            2, 0.5, 0.8, 0.7, 0.6, train_seconds=2.0, val_seconds=1.0, train_images=8, train_steps=2
        ),
    )
    summary = summarize_timing(history)
    assert summary.mean_train_seconds == 2.0
    assert summary.mean_val_seconds == 1.0
    assert summary.images_per_second == 4.0
    assert summary.seconds_per_step == 1.0
    assert summary.peak_memory_bytes is None
    quiet = summarize_timing((EpochMetrics(1, 1.0, 0.5, 1.0, 0.5),))
    assert quiet.images_per_second is None
    result = TrainResult(history, ((1, 0), (0, 1)), (1.0, 1.0), 4, 8)
    record = make_run_record(
        result,
        mode="Transfer",
        learning_rate=0.001,
        batch_size=4,
        epochs=2,
        augment=True,
        device_label="CPU",
        seed=0,
    )
    assert record.images_per_second == 4.0
    assert record.final_train_loss == 0.5
    assert "million" in format_count(2_500_000)
    assert "bytes" in format_bytes(analysis_for_mode(2, LearningMode.NOVEL, 1).parameter_bytes)
    assert IMAGE_SIZE == 224
