# Changelog

## Unreleased

### Changed

- The Python package is renamed to `image_classifier_lab`, matching the repository name, and the distribution name is now `image-classifier-lab`. Imports, the content folder (`image_classifier_lab/content/`, still CC BY 4.0), and the Core ML export file name (`image-classifier-lab.mlpackage`) follow the new name, and the page footer shows the app title.
- The lesson text and the sidebar explanations moved out of `image_classifier_lab/curriculum.py` and `image_classifier_lab/explanations.py` into TOML files under `image_classifier_lab/content/`, which ship inside the package. The rendered text is unchanged. Everything in `image_classifier_lab/content/` is licensed under CC BY 4.0 (`LICENSE-CONTENT`); everything else stays MIT.
- The README Development section states that changes go through a pull request into `main` once CI is green, and that releases, DOIs, history rewrites, and force-pushes need the maintainer's OK.

## 1.1.1 - 2026-11-19

### Changed

- The page title, the README title, and the citation title are now "Lessons in AI from an Image Classifier".

## 1.1.0 - 2026-11-19

### Added

- A guided curriculum of eight sections, from pixels and a linear layer through convolutions, training dynamics, transfer learning, evaluation, and the hardware that runs the arithmetic. Each section states its objectives and has exercises that name which settings to change.
- Diagrams drawn in the app: channel values, softmax, a one-number step-size sketch, a hand-built filter, the residual block, and parameter and multiply-add counts read from this ResNet-18.
- Wall-clock time per training epoch and images per second on the learner's own run. Forward multiply-adds, parameter bytes, Adam state, and a forward-output parts list are computed from the model. A CUDA allocator peak is reported only when the run used CUDA.
- A Hardware sidebar for the training device, inference backends, and Core ML export precision, with reasons for disabled choices.
- Core ML classifier export in float32 or float16, with a downloadable package and validation agreement on macOS.
- An on-device inference section in the hardware lesson, with package sizes and machine-measured inference timings on macOS.
- Hardware exercises for CPU and GPU training, Neural Engine inference, export precision, and Core ML ALL.
- `coremltools` as a dependency, pinned to 9.0 in `requirements.txt`.
- AppTest coverage of the page and the lessons, and a CI coverage floor of 95%.
- README screenshots of the export step, the Hardware panel, and the Lesson 8 timing table, with CPU and Metal GPU training times and Core ML and PyTorch inference times measured on an Intel Mac Pro.

### Changed

- The concept panels on the main page are now the curriculum. One-line help remains on the settings.
- Lesson 8 times only the selected backends and uses the sidebar precision for the Core ML package. CPU_AND_NE is disabled without Apple silicon, and PyTorch timing works without Core ML.
- The version is 1.1.0.

## 1.0.0 - 2026-11-18

First teaching release of ImagePro.

### Fixed

- Novel mode trains every ResNet-18 parameter from random initialization. The previous code loaded random weights and then froze the backbone, so only the new final layer learned, on features that never moved.
- Augmentation runs when an image is loaded for a batch. Each epoch draws a new horizontal flip and a new small rotation. Previously the random transforms ran once at upload, and every epoch reused those same tensors.
- The plotted loss is the mean cross-entropy per image. Previously the chart showed the sum of per-batch means, which grew when the data was split into more batches.

### Added

- Two or more classes, with add and remove controls.
- A per-class validation holdout, and train versus validation loss and accuracy on one figure.
- A validation confusion matrix and per-class accuracy.
- Plain-language explanations for learning rate, batch size, epochs, transfer learning, augmentation, overfitting, confidence, the holdout, shuffle, and the seed.
- A seed that fixes initialization, the holdout, shuffle, and augmentation.
- A built-in set of circle, square, and triangle drawings so the app can be tried without photos.
- Parameter counts for both modes, shown before training.
- Tests, Ruff, and GitHub Actions on Python 3.11–3.13.

### Changed

- Dependencies are the ones the app imports: PyTorch, torchvision, Streamlit, Matplotlib, and Pillow.
- Number inputs and sliders replace free-text settings.
- Transfer mode loads ImageNet weights with the `weights=` argument. The deprecated `pretrained=` flag is gone.
- In transfer mode the frozen backbone's BatchNorm layers stay in evaluation mode, so their ImageNet statistics do not drift while the final layer trains.
