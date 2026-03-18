# Military Aircraft Finder

A compact PyTorch image-classification project that detects and labels military aircraft types from images using a MobileNetV3 backbone. This repository includes data transforms, helper utilities, training and evaluation logic, and an inference script for single-image predictions.

Website: https://slmgx.live

Table of Contents
- Project overview
- Features
- Repository structure
- Requirements
- Dataset layout
- Quick start
  - Training (core.py)
  - Inference (test.py)
  - Single image prediction
- Files explained
  - core.py
  - data_transforms.py
  - helper_functions.py
  - test.py
- Typical workflow
- Tips & troubleshooting
- Contributing
- License
- Acknowledgements & Credits

---

Project overview
----------------
This repository provides an end-to-end example for training an image-classification model to identify military aircraft types. It uses torchvision’s MobileNetV3 (small) as the feature extractor and a single linear classifier head adapted to your dataset’s number of classes.

Features
--------
- MobileNetV3 backbone for efficient model training and inference.
- Standard torchvision augmentations and normalization.
- Training loop with evaluation and checkpointing for the best test accuracy.
- Standalone inference script to load saved weights and predict on single images.
- Utility functions for accuracy, plotting, reproducibility, and dataset handling.

Repository structure
--------------------
- core.py — Main training and evaluation script; includes a single-image predict helper and saves best model weights as `military_aircraft_model.pth`.
- data_transforms.py — Transform pipeline, ImageFolder datasets and DataLoader creation, exposed via `data_transforms` and `transform_dict`.
- military_aircraft_model.pth - pre trained ready to use model
- helper_functions.py — Assorted utilities for accuracy, plotting, download helpers, seed setting, and more (consolidated helpers).
- test.py — Inference script to load `military_aircraft_model.pth` and classify a single image.
- military_aircraft_data_set/ — Expected dataset folder (user-supplied).
- README.md — This document.

Requirements
------------
Recommended Python: 3.8+

Minimum packages (example install):
pip install torch torchvision pillow tqdm matplotlib requests

If you plan to train on GPU, install a CUDA-enabled PyTorch variant following instructions at https://pytorch.org.

Dataset layout
--------------
Place your dataset with this structure (expected by data_transforms.py):

military_aircraft_data_set/
  ├─ Train/
  │   ├─ class_A/
  │   │   ├─ img001.jpg
  │   │   └─ ...
  │   ├─ class_B/
  │   └─ ...
  └─ Test/
      ├─ class_A/
      └─ ...

Each subdirectory under Train and Test is treated as a distinct class by torchvision.datasets.ImageFolder.

Quick start
-----------

1) Prepare dataset
- Ensure `military_aircraft_data_set/Train` and `military_aircraft_data_set/Test` exist and contain per-class subfolders of images.

2) Train (core.py)
- Purpose: Create dataloaders, instantiate MobileNetV3 small, train the model, evaluate on test set, and save the best weights.
- Run:
  - Edit hyperparameters (optional): `epochs`, `lr`, or `batch_size` in `data_transforms.py`/`core.py`.
  - Execute: python core.py
- Behavior highlights:
  - The script automatically detects device: `cuda` if available, otherwise `cpu`.
  - MobileNetV3 small is loaded and the final classifier layer is replaced to match the number of classes from `transform_dict["train_class_names"]`.
  - Best-performing model weights (by test accuracy) are saved to `military_aircraft_model.pth`.

3) Inference (test.py)
- Purpose: Recreate model architecture, load saved state dict, and predict a single image.
- Steps:
  - Place `military_aircraft_model.pth` in the repository root (or adjust the path inside `test.py`).
  - Set `custom_image_path` to your target image.
  - Execute: python test.py
- The script prints the predicted class and confidence. It includes basic FileNotFoundError handling for the input image.

4) Single image prediction helper
- Both `core.py` and `test.py` expose `predict_on_image(image_path, model, transform, class_names, device)`:
  - Loads an image via PIL, applies transforms, runs inference, and returns (predicted_class, confidence_score).

Files explained
---------------

core.py
- Sets up device, model (MobileNetV3 small), loss (CrossEntropyLoss), optimizer (Adam), training and testing loops.
- Key functions:
  - train_step(model, transform_dict, loss_fn, optimizer, accuracy_fn, device): runs one epoch of training, returns average train loss/accuracy.
  - test_step(...): evaluates on the test DataLoader and returns average test loss/accuracy.
  - predict_on_image(...): single-image inference using PIL + transforms.
- Checkpointing: Saves `military_aircraft_model.pth` when test accuracy improves.

data_transforms.py
- Defines `data_transforms` (Resize, RandomHorizontalFlip, RandomRotation, ColorJitter, ToTensor, Normalize).
- Creates torchvision.datasets.ImageFolder instances for Train and Test and DataLoaders with default batch_size of 32.
- Exposes `transform_dict` containing:
  - train_class_names, test_class_names
  - train_loader, test_loader
  - class_to_idx maps
  - batch_size

helper_functions.py
- A consolidated collection of utilities for experimentation and visualization:
  - walk_through_dir(dir_path): print directory and file counts for dataset exploration.
  - accuracy_fn(y_true, y_pred): returns accuracy percentage.
  - plot_decision_boundary / plot_predictions / plot_loss_curves: plotting helpers for analytics and diagnostics.
  - pred_and_plot_image(...): convenience to predict and plot a single image with its predicted label/probability.
  - set_seeds(seed): set torch and CUDA seeds for reproducibility.
  - download_data(source, destination, remove_source=True): download and extract zipped dataset from a URL.
  - print_train_time(start, end, device): print training duration.
- Credit: Many utilities and patterns are adapted from public PyTorch learning resources — special thanks to Daniel Bourke for the inspiration and helper function examples.

test.py
- Re-creates the MobileNetV3 small architecture (with classifier head adjusted) and loads the saved state dict for inference.
- Uses the same transforms found in `data_transforms.py` to ensure input preprocessing consistency.

Typical workflow
-----------------
1. Prepare data in the expected folder structure.
2. Inspect dataset with helper_functions.walk_through_dir(...) if desired.
3. Run python core.py to train. Monitor training/test loss & accuracy prints.
4. Use the saved `military_aircraft_model.pth` with test.py for quick single-image predictions, or integrate the model into an application.

Tips & troubleshooting
----------------------
- FileNotFoundError when predicting: set `custom_image_path` to a valid absolute or relative image path.
- Mismatched classifier size: Ensure the number of classes in training dataset remains the same when loading saved weights. Both core.py and test.py resize the final linear layer according to `len(transform_dict["train_class_names"])`.
- GPU not used: confirm you installed a CUDA-enabled PyTorch build and that `torch.cuda.is_available()` returns True.
- Batch size and augmentation: Data augmentations (random flip, rotation, color jitter) are helpful for generalization but tune them as necessary for your dataset characteristics.
- Determinism: call `set_seeds(seed)` from `helper_functions.py` at script start for more reproducible runs. Full determinism on GPUs may require additional flags and can impact performance.
- Large models on limited GPU memory: reduce batch size or use model checkpoint/resume strategies.

Contributing
------------
Contributions, issues, and feature requests are welcome. Suggested workflow:
- Fork the repository.
- Create a branch (feature/your-feature).
- Make changes and include tests/examples if applicable.
- Open a pull request describing the changes.

License
-------
GNU General Public License v3.0

Acknowledgements & Credits
--------------------------
- Helper functions, patterns and many educational examples were adapted from public PyTorch learning resources. A special thank you to Daniel Bourke for the helpful tutorials and consolidated helper function ideas used in this project.
- PyTorch & torchvision projects for model implementations and APIs.

Contact
-------
Project author: SL-MGx03  
Personal website: https://slmgx.live

If you would like, this README can be added directly to the repository as a PR and I can also prepare a recommended requirements.txt and a LICENSE file.
