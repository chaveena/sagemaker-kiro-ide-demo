# Implementation Plan: Satellite Land Cover Demo

## Overview

Build a minimal, end-to-end SageMaker demo that trains a small from-scratch PyTorch CNN on a EuroSAT subset (forest / water / cropland / urban) via a managed training job. The deliverables are a single guided notebook (`satellite_land_cover_demo.ipynb`), a standalone `train.py` entry point, and one `requirements.txt`.

Implementation proceeds bottom-up: declare dependencies, build the reusable subset-prep and model/argument logic (which the property tests target), then assemble the notebook that orchestrates setup → prep → S3 upload → estimator `fit()` → result reporting. All code is Python.

## Tasks

- [x] 1. Declare project dependencies
  - Create `requirements.txt` with exactly `sagemaker>=2.200`, `torch`, `torchvision`, and `matplotlib`
  - `matplotlib` supports the two in-notebook visualizations (sample-tiles grid and predicted-vs-actual grid)
  - This is the single dependency specification file for the demo
  - _Requirements: 5.4, 5.5_

- [ ] 2. Implement the training script (`train.py`)
  - [~] 2.1 Define the from-scratch CNN
    - Implement `SmallCNN(num_classes=4)`: three Conv/ReLU/MaxPool blocks (3→16→32→64) over 3×64×64 input, flatten, Linear(64*8*8→128)→ReLU, Linear(128→4)
    - No pretrained weights; final layer emits exactly four logits
    - _Requirements: 3.2, 3.3_

  - [~] 2.2 Write property test for model output shape
    - **Feature: satellite-model-demo, Property 4: For any input batch of B valid 3×64×64 image tensors, the model's forward pass SHALL produce an output of shape (B, 4).**
    - **Validates: Requirements 3.3**
    - Generate random valid batch sizes B; assert `SmallCNN(4)(x).shape == (B, 4)`; minimum 100 iterations

  - [~] 2.3 Implement argument parsing
    - Parse `--epochs` and `--batch-size` hyperparameters; default `--data-dir` from `SM_CHANNEL_TRAINING` and `--model-dir` from `SM_MODEL_DIR`
    - _Requirements: 3.4, 3.5, 3.6_

  - [~] 2.4 Write property test for hyperparameter round-trip
    - **Feature: satellite-model-demo, Property 5: For any non-negative integer values supplied for --epochs and --batch-size, the parser SHALL return those exact values.**
    - **Validates: Requirements 3.6**
    - Generate random non-negative integers; parse and assert returned values are unchanged; minimum 100 iterations

  - [~] 2.5 Implement data loading, training loop, and artifact save
    - Build an `ImageFolder` DataLoader from `--data-dir`; train `SmallCNN` with cross-entropy + Adam for `--epochs`
    - Save the trained model to `--model-dir` (e.g. `model.pth`) and wire everything under a `main()` entry point
    - _Requirements: 3.1, 3.4, 3.5_

- [~] 3. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Implement the dataset subset preparation logic
  - [~] 4.1 Implement subset preparation function
    - Download EuroSAT via torchvision, map source categories (Forest, SeaLake, AnnualCrop, Industrial) to the four target classes, cap at `IMAGES_PER_CLASS`, and write an `ImageFolder`-style layout (`data/<class>/*.jpg`)
    - Also set aside a few held-out tiles per class as an in-memory list of `(image, actual_label)` pairs (not written to disk, not uploaded) for the post-training prediction grid
    - Return per-class counts computed from files written to disk plus the in-memory held-out list
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 7.3_

  - [~] 4.2 Write property test for subset size and per-class cap
    - **Feature: satellite-model-demo, Property 1: For any per-class cap N smaller than the smallest full EuroSAT category size, the prepared subset SHALL contain at most N images per class and a total strictly less than the full EuroSAT dataset size.**
    - **Validates: Requirements 2.2**
    - Generate random caps N; assert each class has ≤ N images and total < full dataset size; minimum 100 iterations

  - [~] 4.3 Write property test for label domain
    - **Feature: satellite-model-demo, Property 2: For any prepared subset, the set of distinct class labels SHALL equal exactly {forest, water, cropland, urban}.**
    - **Validates: Requirements 2.3**
    - Generate random subset selections; assert the label set equals the four target classes; minimum 100 iterations

  - [~] 4.4 Write property test for reported vs on-disk counts
    - **Feature: satellite-model-demo, Property 3: For any subset selection, the per-class counts reported SHALL equal the actual files written per class directory, and the reported total SHALL equal their sum.**
    - **Validates: Requirements 2.5**
    - Generate random subset selections; assert reported per-class counts equal on-disk file counts and total equals their sum; minimum 100 iterations

- [~] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 6. Build the guided notebook (`satellite_land_cover_demo.ipynb`)
  - [~] 6.1 Add the story and Kiro/SageMaker narration
    - Markdown intro presenting the workflow as a continuous story about tracking habitat and land change, describing the four classes, and calling out where Kiro assists inside SageMaker JupyterLab
    - _Requirements: 1.2, 1.3, 6.1, 6.2_

  - [~] 6.2 Add SageMaker setup cells
    - Create a `sagemaker.Session`, resolve the Execution_Role via default resolution, and select the default bucket; narrate why the role and bucket are required in plain language
    - _Requirements: 1.4, 5.1, 5.2, 5.3_

  - [~] 6.3 Add data preparation and reporting cells
    - Call the subset prep logic from task 4.1 and print per-class image counts; keep the returned in-memory held-out tiles for the later prediction grid
    - _Requirements: 1.2, 2.1, 2.2, 2.3, 2.5_

  - [~] 6.4 Add sample-tiles preview cell (before training)
    - Implement a `show_sample_grid(samples)` matplotlib helper and call it on a few local tiles per target class, annotating each with its class label; narrate that this shows what the model learns from and runs on already-local images without altering the training job
    - _Requirements: 7.1, 7.4_

  - [~] 6.5 Add S3 upload cell
    - Upload the prepared subset to the default bucket under the demo prefix to form the training input channel (held-out tiles remain in memory and are not uploaded)
    - _Requirements: 2.4_

  - [~] 6.6 Add estimator configuration and launch cells
    - Configure a `sagemaker.pytorch.PyTorch` estimator with `entry_point="train.py"`, small hyperparameters, and a small instance type; narrate key arguments; call `fit()` with the S3 input channel so training runs on managed infrastructure
    - _Requirements: 1.4, 4.1, 4.2, 4.3, 4.4_

  - [~] 6.7 Add result and failure reporting cells
    - On success, print `estimator.model_data`; on failure, catch the exception and surface the SageMaker `FailureReason` from the job description
    - _Requirements: 4.5, 4.6_

  - [~] 6.8 Add load-trained-model cell (after training)
    - Implement a `load_trained_model(model_data_uri, download_fn)` helper that downloads `model.tar.gz` from `estimator.model_data`, extracts `model.pth`, rebuilds `SmallCNN(num_classes=4)`, loads the saved `state_dict`, and switches to eval mode; narrate that this brings the managed job's result back locally purely for visualization
    - _Requirements: 7.2, 7.4_

  - [~] 6.9 Add predicted-vs-actual grid cell
    - Implement a `show_prediction_grid(model, held_out, to_tensor)` matplotlib helper that runs the loaded model on the in-memory held-out tiles, maps each prediction through the four target classes, and annotates each tile with its predicted and actual label; call it on the held-out set
    - _Requirements: 7.3, 7.4_

  - [~] 6.10 Write unit test for failure reporting
    - Given a failed job description containing a `FailureReason`, assert the reporting logic surfaces that reason
    - _Requirements: 4.6_

  - [~] 6.11 Write property test for prediction grid
    - **Feature: satellite-model-demo, Property 6: For any held-out set of K tiles and any model output of shape (K, 4), the predicted-vs-actual grid SHALL annotate exactly K tiles, and each annotated predicted label SHALL be one of the four target classes {forest, water, cropland, urban}.**
    - **Validates: Requirements 7.1, 7.3**
    - Generate random held-out set sizes K and random (K, 4) model outputs; assert exactly K tiles are annotated and every predicted label is in {forest, water, cropland, urban}; minimum 100 iterations

- [~] 7. Verify top-to-bottom notebook flow
  - Confirm cells are ordered setup → prep → report → sample preview → upload → configure → fit → report → load model → prediction grid, runnable top-to-bottom without out-of-sequence execution
  - _Requirements: 1.1, 1.5_

- [ ] 8. Write integration/smoke tests
  - [~] 8.1 S3 upload smoke test
    - Upload the subset and confirm objects exist at the expected prefix
    - _Requirements: 2.4_

  - [~] 8.2 Managed training job smoke test
    - Launch the estimator and confirm the job runs with `train.py` as entry point and the S3 input channel, and that `estimator.model_data` reports the artifact URI
    - _Requirements: 4.1, 4.2, 4.3, 4.5_

  - [~] 8.3 Training environment I/O smoke test
    - Confirm `train.py` reads `SM_CHANNEL_TRAINING` and writes an artifact to `SM_MODEL_DIR`
    - _Requirements: 3.4, 3.5_

- [~] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional (property, unit, and integration tests) and can be skipped for a faster MVP.
- Each task references specific requirements for traceability; property test tasks also reference their design property.
- Checkpoints ensure incremental validation.
- The subset-prep, model, and argument-parsing logic are built before the notebook so the property tests can target them directly.
- Property tests run a minimum of 100 iterations because they rely on randomized inputs.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["2.1", "2.3", "4.1"] },
    { "id": 1, "tasks": ["2.2", "2.4", "2.5", "4.2", "4.3", "4.4"] },
    { "id": 2, "tasks": ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9"] },
    { "id": 3, "tasks": ["6.10", "6.11", "8.1", "8.2", "8.3"] }
  ]
}
```
