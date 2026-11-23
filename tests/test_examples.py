"""The built-in shapes are deterministic and legal training data."""

from __future__ import annotations

import hashlib
import io
import ssl
import urllib.request
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from image_classifier_lab import examples
from image_classifier_lab.data import split_dataset
from image_classifier_lab.examples import (
    EXAMPLE_IMAGES_PER_CLASS,
    PUBLIC_IMAGES_PER_CLASS,
    PUBLIC_LABELS,
    make_cifar_dataset,
    make_shape_dataset,
)


def test_shapes_are_stable_and_visibly_different() -> None:
    first = make_shape_dataset()
    second = make_shape_dataset()
    assert [group.label for group in first] == ["circle", "square", "triangle"]
    assert all(len(group.images) == EXAMPLE_IMAGES_PER_CLASS for group in first)
    assert first[0].images[0].size == second[0].images[0].size
    assert first[0].images[0].tobytes() == second[0].images[0].tobytes()
    assert first[0].images[0].tobytes() != first[0].images[1].tobytes()
    assert first[0].images[0].tobytes() != first[1].images[0].tobytes()


def test_shapes_can_be_split() -> None:
    split = split_dataset(make_shape_dataset(), 0.25, seed=0)
    assert split.class_names == ("circle", "square", "triangle")
    assert len(split.train_images) == 18
    assert len(split.val_images) == 6


def test_public_cifar_subset_is_labeled_and_deterministic(tmp_path) -> None:
    class FakeCIFAR10:
        class_to_idx = {
            "airplane": 0,
            "automobile": 1,
            "bird": 2,
        }

        def __init__(self, **_kwargs):
            self.targets = [0] * 30 + [1] * 30 + [2] * 30

        def __getitem__(self, index):
            color = (index * 3 % 256, index * 5 % 256, index * 7 % 256)
            return Image.new("RGB", (32, 32), color), self.targets[index]

    with patch("torchvision.datasets.CIFAR10", FakeCIFAR10):
        first = make_cifar_dataset(
            images_per_class=PUBLIC_IMAGES_PER_CLASS,
            seed=0,
            root=str(tmp_path),
        )
        second = make_cifar_dataset(
            images_per_class=PUBLIC_IMAGES_PER_CLASS,
            seed=0,
            root=str(tmp_path),
        )

    assert [group.label for group in first] == list(PUBLIC_LABELS)
    assert all(len(group.images) == PUBLIC_IMAGES_PER_CLASS for group in first)
    assert first[0].images[0].tobytes() == second[0].images[0].tobytes()


ARCHIVE = b"cifar archive bytes"


class MissingCIFAR10:
    """Stands in for torchvision's CIFAR10: absent until the archive is extracted."""

    filename = "cifar-10-python.tar.gz"
    url = "https://example.test/cifar-10-python.tar.gz"
    tgz_md5 = hashlib.md5(ARCHIVE).hexdigest()

    def __init__(self, root: str, train: bool, download: bool) -> None:
        assert download is False
        if not (Path(root) / "extracted").is_dir():
            raise RuntimeError("Dataset not found or corrupted.")


def test_cifar_download_uses_a_verifying_context_and_extracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contexts: list[object] = []

    def fake_urlopen(
        request: urllib.request.Request, *, context: object, timeout: int
    ) -> io.BytesIO:
        assert request.full_url == MissingCIFAR10.url
        contexts.append(context)
        return io.BytesIO(ARCHIVE)

    def fake_extract(archive: str, root: str) -> None:
        assert Path(archive).read_bytes() == ARCHIVE
        (Path(root) / "extracted").mkdir()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    with (
        patch("torchvision.datasets.CIFAR10", MissingCIFAR10),
        patch("torchvision.datasets.utils.extract_archive", fake_extract),
    ):
        dataset = examples._open_cifar(tmp_path)

    assert isinstance(dataset, MissingCIFAR10)
    assert len(contexts) == 1
    assert isinstance(contexts[0], ssl.SSLContext)
    assert contexts[0].verify_mode == ssl.CERT_REQUIRED
    assert not list(tmp_path.glob("*.part"))


def test_a_corrupt_cifar_download_is_discarded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: io.BytesIO(b"cut"))
    with (
        patch("torchvision.datasets.CIFAR10", MissingCIFAR10),
        pytest.raises(RuntimeError, match="incomplete"),
    ):
        examples._open_cifar(tmp_path)
    assert not (tmp_path / MissingCIFAR10.filename).exists()
    assert not list(tmp_path.glob("*.part"))
