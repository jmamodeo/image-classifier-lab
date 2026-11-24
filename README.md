# Lessons in AI from an Image Classifier

[![CI](https://github.com/jmamodeo/image-classifier-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/jmamodeo/image-classifier-lab/actions/workflows/ci.yml)

A Streamlit lab for learning how an image classifier works, and how that work lands on the hardware that runs it. Eight lessons sit above the Image Classifier. Upload a few labeled images, or pick one of the built-in sets, then train a ResNet-18 and read the charts.

Two modes share the images, the loss, and the learning rate. **Transfer** starts from ImageNet weights and trains a new final layer while the backbone stays frozen. **Novel** starts from random weights and trains every layer.

The **Hardware** sidebar lets you choose where to train, which inference backends to time, and the precision of the Core ML export. The training source offers built-in shapes, a small cached subset of the public CIFAR-10 dataset with airplane, automobile, and bird labels, or uploaded images.

![First open: eight lesson buttons, the objectives for Images as numbers, and the three color channels of a built-in drawing.](docs/home.png)

## The curriculum

The eight lessons are stacked vertically in numbered, collapsed containers. Expand any lesson to read it; several can stay open at once. A bar counts how many exercises have been marked done. Marking an exercise done is a checkbox, not a gate. The **Image Classifier** is always visible below the lessons, framed like a standalone app: choose a data source, check the class cards, train, read the results, try an image, and export to Core ML. Built-in shapes, built-in images, and uploads all fill the same class cards. A collapsed **Under the hood** section below the classifier holds the explanations: the parameter cards, the augmentation preview, notes on the last run, and recent runs. The settings stay in the sidebar the whole time.

Each lesson states its objectives, explains the idea with a diagram or a count taken from this model, and ends with exercises. Each exercise names which settings to change, lists the ones to leave at their defaults, says what to look for, and hides a short check in a popover until you open it.

| | Lesson | Exercise |
| --- | --- | --- |
| 1 | Images as numbers. Pixels, channels, the tensor shape, resize, and ImageNet normalization. | Choose Built-in shapes. Change nothing. Compare a training draw with the holdout picture. |
| 2 | A neuron and a linear layer. Weights, bias, the dot product, and softmax. | Change only Learning mode, and compare the two parameter cards in Under the hood. |
| 3 | Loss, gradients, and the learning rate. Cross-entropy, a step downhill, and what Adam rescales. | Train three times. Change only the learning rate: 0.0001, then 0.001, then 0.03. |
| 4 | Convolutions, depth, and residual connections. Filters, feature maps, and where the parameters sit. | Train once in Transfer, then once in Novel, and look at the first-layer filters. |
| 5 | Epochs, batches, and overfitting. The holdout, augmentation, and the seed. | Train once at 15 epochs. Then train at 5 epochs with augmentation off. |
| 6 | Transfer learning and training from scratch. Frozen layers and BatchNorm statistics. | Train Transfer, then Novel, at the defaults. Read the trainable counts and the BatchNorm caption. |
| 7 | Accuracy, confusion, and confidence. What the percent beside a prediction is. | Read the confusion matrix of a default run. |
| 8 | Arithmetic, memory, and the device. Multiply-adds, memory, on-device inference, and measurements from your own machine. | Compare CPU and GPU training across batch sizes. Compare export precisions and test the available inference devices, including Neural Engine where supported. |

![The exercise at the end of Images as numbers, naming the settings to leave alone.](docs/exercise.png)

The diagrams are drawn in the app. A few of them are labeled as sketches: the one-number bowl is plain gradient descent, not Adam, and the overfitting chart is an invented shape so the pattern is recognizable. The confusion grid titled "Reading practice — not a run" is a made-up matrix for the arithmetic. Everything else in a lesson is computed from this ResNet-18, from the built-in drawings, or from the run you just finished.

![The scores lesson, with a made-up confusion matrix labeled as reading practice.](docs/scores.png)

## What a run looks like

Built-in drawings, eight per class. The parameter cards in **Under the hood** count trainable parameters before any training starts. For three classes that is 1,539 in transfer mode and all 11,178,051 in novel mode.

![Built-in circles, squares, and triangles, the parameter cards, and four augmented draws of a training square beside the validation square.](docs/augmentation.png)

Train and validation share both axes. On a five-epoch transfer run at the defaults, the mean loss fell on both sets. The page reports the finals; this chart's run ended at training loss 0.223 and validation loss 0.258.

![Mean loss and accuracy for five epochs of transfer training on the built-in shapes.](docs/curves.png)

The same default run, on the CPU used to check this page, also reports wall-clock time. That machine averaged 1.07 seconds per training epoch and 16.8 images per second. Those two numbers are one run on one CPU. They are not a benchmark, and another machine will print its own.

![After training: the timing line, the parameter sentence, and the loss and accuracy charts.](docs/run.png)

Rows of the confusion matrix are the true class. Per-class accuracy is the fraction of that class's held-out images labeled correctly. On two images per class, 100% is those two images.

![Validation confusion matrix and per-class accuracy, all three classes at 100% on two images each.](docs/confusion.png)

A new image takes the validation path: resize and normalize, with no flip and no rotation. The percent is the softmax share.

![A held-out circle classified as circle at 94.5% softmax share.](docs/prediction.png)

After training, choose export precision in the **Hardware** sidebar, then convert and download a zipped Core ML `.mlpackage`. The export includes the image normalization and class labels. Conversion works on Linux and macOS. Core ML verification and timing require macOS; PyTorch timing does not.

![The export step on macOS: the export precision set in Hardware, the float32 package size, and its agreement with PyTorch on the six validation images.](docs/export.png)

The chart screenshots above are a default run: transfer mode, CPU, Adam at learning rate 0.001, batch size 4, 5 epochs, 20% holdout, seed 0, shuffle and augmentation on.

## Hardware controls and measurements

The **Hardware** sidebar has **Training**, **Inference**, and **Precision** controls. Training always uses float32 on CPU, CUDA, or Metal (MPS); “Try an image” uses the device that trained the model. Unavailable choices are disabled. The Neural Engine option appears below CPU and requires CPU to be selected and Core ML to be available.

All available inference devices start checked. Lesson 8 measures the checked devices on the same image: CPU and GPU use float32 PyTorch, while Neural Engine uses Core ML's CPU_AND_NE setting, which may also run layers on CPU. Export a package to include Neural Engine; CPU and GPU can be timed without one. Switch precision and export again to compare size, median latency, top-1 agreement, and the largest probability difference against float32 PyTorch.

The counts are computed from this ResNet-18 at the class count and batch size in the sidebar. A multiply-add is one multiplication of a weight by an input, added into a total, inside a convolution or a linear layer. Bias, BatchNorm, ReLU, and pooling are left out of the count. FLOPs, on this page, are twice the multiply-adds. A training step also runs a backward pass, which is not in that number.

Memory is split into parts: parameter bytes, gradient bytes for the trainable parameters, two float32 Adam buffers per trainable parameter, BatchNorm buffers, and a forward-output parts list. The parts list adds up the tensor each leaf layer returned. Those tensors are not all stored at once, so the total is not a memory-meter reading. Parameter bytes and Adam bytes do not grow with the batch. The forward-output list does.

Images per second come from the run: training images scored, divided by the time spent in the training pass, including loading the images and the optimizer step. Validation time is reported separately. When the run used CUDA, the page also reports the allocator's peak. CPU and Metal GPU (MPS) runs do not invent a peak.

The sentence in the screenshot below is the count for batch 4, 3 classes (the built-in shapes), Transfer. Change the class count or the batch size in the sidebar and the page recomputes.

![The hardware lesson with the Hardware sidebar: multiply-adds for one forward pass, and the memory parts for that model and batch.](docs/hardware.png)

The lesson describes on-device privacy, network latency, energy, and Core ML compute units. Compare training seconds per epoch by device and batch size in the run log. Inference timing does not prove which compute unit ran a Core ML layer.

This earlier Hardware panel on an Intel Mac Pro (Xeon W-3245, AMD Radeon Pro W5700X, macOS 26.5.2) predates the current device controls:

![The Hardware sidebar panel with the NVIDIA GPU and Core ML CPU_AND_NE options greyed out and a reason under each.](docs/hardware-panel.png)

An earlier Lesson 8 on the same Mac, with the then-available Core ML settings in one timing table:

![Lesson 8 on an Intel Mac Pro: the on-device section, package size, and the timing table for the five available backends.](docs/hardware-timing.png)

The measurements below came from that Mac on 2026-11-19, with other programs running, so treat them as one machine on one afternoon, not a benchmark. PyTorch publishes no Intel Mac wheels after 2.2, so this run used Python 3.12.16, PyTorch 2.2.2, and coremltools 9.0 rather than the `requirements.txt` pins.

Training seconds per epoch on the built-in shapes (median of 3 epochs; 64 images per class gives 153 training images; the default 8 per class gives 18). Metal (MPS) is the Mac's Metal GPU:

| Mode | Training images | Batch | CPU s/epoch | Metal (MPS) s/epoch |
| --- | --- | --- | --- | --- |
| Transfer | 18 | 4 | 0.314 | 0.149 |
| Transfer | 153 | 4 | 2.386 | 1.146 |
| Transfer | 153 | 16 | 2.158 | 0.486 |
| Transfer | 153 | 64 | 1.669 | 0.340 |
| Novel | 18 | 4 | 0.708 | 0.614 |
| Novel | 153 | 4 | 6.069 | 4.949 |
| Novel | 153 | 16 | 4.296 | 1.719 |
| Novel | 153 | 64 | 3.431 | 1.421 |

The first Novel epoch on Metal with 18 images took 24.215 s against 0.612 s and 0.614 s for the next two, so the first epoch on a GPU can include one-time setup. The first epochs of the larger Metal runs were also slower than the later ones in some rows (for example Novel, batch 16: 6.786 s, then 1.712 s and 1.719 s).

Batch-one inference, median of 10 runs after 3 warmup runs, repeated 3 times (the range across the repeats). PyTorch rows pool all 6 repeats because PyTorch stays in float32:

| Backend | float32 package (ms) | float16 package (ms) |
| --- | --- | --- |
| Core ML CPU_ONLY | 23.010 to 23.779 | 25.311 to 25.757 |
| Core ML CPU_AND_GPU | 4.399 to 7.433 | 3.887 to 4.405 |
| Core ML ALL | 4.430 to 5.589 | 3.780 to 4.621 |
| PyTorch CPU (float32) | 23.650 to 36.953 | same pool |
| PyTorch Metal GPU (MPS, float32) | 13.506 to 14.813 | same pool |

The float32 package was 44,726,483 bytes and the float16 package 22,383,231 bytes. Both matched PyTorch's top-1 label on all six validation images. The largest probability difference was 9.31e-09 for float32 and 2.64e-04 for float16.

## Quickstart

Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
# CPU-only machine: install the CPU builds of PyTorch first. Skip this line if you have an NVIDIA GPU.
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
streamlit run main.py
```

`requirements.txt` pins the versions the app was tested with. The CPU builds satisfy those pins, so the last install keeps them. With a GPU, skip the PyTorch line and `pip install -r requirements.txt` takes the default PyTorch wheel.

The first transfer run downloads the ImageNet ResNet-18 weights, about 45 MB, from `download.pytorch.org`. Novel mode trains from random initialization and does not download weights.

## How a run works

1. Each class needs a label and at least two images, so one image can be held out and one can train. You can keep two to eight classes. The holdout is stratified: the same fraction of every class, at least one image each way. The seed chooses which images are held out.
2. Images are resized to 224×224. A training image may then be flipped left-right (probability 0.5) and rotated by up to 15°. That draw happens when the image is loaded for a batch, so each epoch sees new tensors. Validation images are resized only. Both modes use ImageNet mean and standard deviation, so the comparison is about the weights.
3. Transfer mode loads `ResNet18_Weights.IMAGENET1K_V1`, replaces the 1000-way layer with one for your classes, freezes the backbone, and leaves its BatchNorm layers in evaluation mode. The ImageNet running statistics stay fixed while the new layer trains. Novel mode builds the same ResNet-18 with `weights=None` and trains every parameter. The new final layer starts random in both modes.
4. Adam steps on the trainable parameters. The plotted loss is the mean cross-entropy per image: the sum of the per-image losses divided by the number of images scored.
5. After each novel-mode epoch, BatchNorm's stored averages are recomputed from the training images. On a handful of images the default momentum would leave those averages near their initialization, and the validation loss would describe that mismatch. Transfer mode leaves the ImageNet statistics where they are.
6. One figure shows training and validation loss, and training and validation accuracy. Under it, a confusion matrix and per-class accuracy on the holdout. A short note says whether the two losses moved together. A table keeps the last few runs, so a learning-rate sweep or a Transfer-versus-Novel pair can be compared after the charts move on.
7. Prediction resizes and normalizes one image the same way as validation. The percent beside the label is that image's softmax share across your classes.

The seed sets initialization, the holdout, the shuffle order, and the augmentation sequence. The same seed on a CPU repeats the run. GPU kernels can still differ by a small amount.

Batch size starts at 2. ResNet-18's BatchNorm layers need more than one image in a batch. If the last training batch would be a single image, that image waits for another epoch rather than producing a NaN loss.

One-line help on the sidebar controls still defines the setting under the cursor. The lessons are the longer explanation.

## Layout

```
main.py                               Streamlit entry point
image_classifier_lab/curriculum.py    the eight lessons and their exercises
image_classifier_lab/lesson_ui.py     lesson page, progress, and the run table
image_classifier_lab/diagrams.py      the figures drawn in the lessons
image_classifier_lab/demos.py         the small calculations behind those figures
image_classifier_lab/measure.py       multiply-adds, memory parts, and timing summaries
image_classifier_lab/coreml.py        Core ML export, verification, and inference timing
image_classifier_lab/hardware.py      training and inference availability
image_classifier_lab/data.py          loading, the holdout, augmentation
image_classifier_lab/model.py         ResNet-18 and the parameter counts
image_classifier_lab/training.py      the loop, metrics, prediction
image_classifier_lab/insights.py      a few sentences about the finished curves
image_classifier_lab/plots.py         loss, accuracy, confusion, class shares
image_classifier_lab/examples.py      the built-in shapes
image_classifier_lab/explanations.py  one-line help for the sidebar settings
image_classifier_lab/content/         the lesson and explanation text, as TOML files (CC BY 4.0)
image_classifier_lab/content_files.py reads the content files and fills in the code's numbers
image_classifier_lab/ui.py            the page
CITATION.cff                          citation metadata
LICENSE                               MIT license (everything outside image_classifier_lab/content/)
LICENSE-CONTENT                       CC BY 4.0 (everything in image_classifier_lab/content/)
```

## Development

Contributor setup, for running the checks. Use it instead of the Quickstart install, in its own virtual environment. It installs the app in editable mode with the `dev` extras.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest --cov
```

Changes go through a pull request into `main` once CI is green. Releases and DOIs are cut by the maintainer.

Tests build every model with `weights=None` and block the weight download, so they stay offline and on CPU. The suite drives the real page with Streamlit's `AppTest`, including a one-epoch training run, and `pytest --cov` fails if coverage of `image_classifier_lab` drops below the floor set in `pyproject.toml`. GitHub Actions runs an import smoke test, Ruff, and `pytest --cov` on Python 3.11, 3.12, and 3.13.

## Boundaries

- A few images are enough to move the curves. The holdout is then only a handful of images, so validation accuracy jumps around. Read the shape of the curves, and treat a perfect confusion matrix on two images per class as a demonstration.
- The percent beside a prediction is the softmax share across the classes you trained.
- Augmentation is a horizontal flip and a rotation of at most 15°.
- Transfer mode trains the new final layer. The ImageNet backbone, including its BatchNorm statistics, stays fixed. That is the contrast the page is built to show.
- The optimizer is Adam for a fixed number of epochs.
- The example drawings differ in color and in outline, so a short run has an easy signal. They are a teaching set.
- The hardware counts cover convolutions and linear layers. They are not a full-chip benchmark, and the forward-output total is not peak memory.

## Citation

If you use this software, please cite it. GitHub's "Cite this repository" button uses [`CITATION.cff`](CITATION.cff).

## License

Copyright (c) 2025-2026 John Amodeo.

- **Teaching content**: everything in [`image_classifier_lab/content/`](image_classifier_lab/content/) (the lesson titles, objectives, explanations, exercises and checks, and the help text for the sidebar settings) is under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). See [LICENSE-CONTENT](LICENSE-CONTENT). Reuse is welcome with attribution to John Amodeo and a link to this repository.
- **Everything else**, including the code, this README, and the screenshots in `docs/`, is under the MIT license. See [LICENSE](LICENSE).

See the [CHANGELOG](CHANGELOG.md) for release history.
