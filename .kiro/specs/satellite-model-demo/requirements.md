# Requirements Document

## Introduction

This feature delivers a minimal, understandable demonstration of how to use Amazon SageMaker JupyterLab together with Kiro to train a machine learning model from satellite imagery. The demo is framed around a conservation-oriented land cover / land use classification use case: distinguishing forest, water, cropland, and urban areas from satellite tiles as a way to track habitat and land change over time.

The demo uses the public EuroSAT dataset (a small subset for speed), a small custom PyTorch convolutional neural network (CNN) trained from scratch, and a SageMaker managed training job launched from a Jupyter notebook using the SageMaker PyTorch estimator. The deliverables are a single guided notebook, a standalone `train.py` training entry-point script, and minimal supporting configuration. The primary intent is clarity and a working end-to-end experience, not model accuracy or production readiness.

## Glossary

- **Demo**: The complete set of deliverables (notebook, training script, and supporting files) that demonstrate training a satellite-imagery model on SageMaker.
- **Notebook**: The single Jupyter notebook that narrates and orchestrates the end-to-end workflow inside SageMaker JupyterLab.
- **Training_Script**: The standalone `train.py` entry-point script executed by the SageMaker managed training job.
- **Training_Job**: A SageMaker managed training job launched via the SageMaker PyTorch estimator from the Notebook.
- **Estimator**: The SageMaker PyTorch estimator object configured in the Notebook to launch the Training_Job.
- **Model**: The small custom PyTorch CNN defined in the Training_Script and trained from scratch.
- **Dataset**: The EuroSAT land cover / land use image dataset, used as a small subset in this Demo.
- **Dataset_Subset**: A reduced portion of the EuroSAT Dataset selected to keep data footprint small and training fast.
- **Land_Cover_Classes**: The set of target categories the Model predicts (forest, water, cropland, urban).
- **S3_Bucket**: The Amazon S3 location used to stage training data and receive model artifacts.
- **Execution_Role**: The AWS IAM role SageMaker uses to run the Training_Job and access the S3_Bucket.
- **User**: A person following the Demo inside SageMaker JupyterLab.

## Requirements

### Requirement 1

**User Story:** As a data scientist working on conservation use cases, I want a single guided notebook that walks through the full workflow, so that I can understand how to train a satellite-imagery model on SageMaker without prior SageMaker experience.

#### Acceptance Criteria

1. THE Demo SHALL provide exactly one Notebook that orchestrates the end-to-end workflow from data preparation through Training_Job completion.
2. THE Notebook SHALL include narrative text that explains the purpose of each step before the corresponding code cell.
3. THE Notebook SHALL describe the land cover / land use classification use case and its relevance to tracking habitat and land change.
4. WHERE a step requires SageMaker-specific setup, THE Notebook SHALL explain the reason for that setup in plain language.
5. THE Notebook SHALL execute its code cells in top-to-bottom order without requiring the User to run cells out of sequence.

### Requirement 2

**User Story:** As a data scientist, I want the demo to use a small subset of the EuroSAT dataset, so that data download and training complete quickly with a small footprint.

#### Acceptance Criteria

1. THE Demo SHALL use the EuroSAT Dataset as the source of satellite imagery.
2. THE Notebook SHALL prepare a Dataset_Subset that is smaller than the full EuroSAT Dataset.
3. THE Demo SHALL classify satellite tiles into the four Land_Cover_Classes: forest, water, cropland, and urban.
4. THE Notebook SHALL upload the prepared Dataset_Subset to the S3_Bucket for use by the Training_Job.
5. WHEN the Dataset_Subset preparation completes, THE Notebook SHALL report the number of images per Land_Cover_Class.

### Requirement 3

**User Story:** As a data scientist, I want a small, readable custom PyTorch CNN in a standalone script, so that I can understand exactly what model is being trained and adapt it later.

#### Acceptance Criteria

1. THE Demo SHALL provide a standalone Training_Script named `train.py` that serves as the Training_Job entry point.
2. THE Training_Script SHALL define a custom PyTorch CNN Model trained from scratch without pretrained weights.
3. THE Training_Script SHALL classify input images into the four Land_Cover_Classes.
4. THE Training_Script SHALL read training data from the input path provided by the SageMaker training environment.
5. WHEN training completes, THE Training_Script SHALL save the trained Model artifact to the SageMaker model output path.
6. THE Training_Script SHALL accept training hyperparameters, including epoch count and batch size, as command-line arguments.

### Requirement 4

**User Story:** As a data scientist, I want the notebook to launch a SageMaker managed training job using the PyTorch estimator, so that training runs on managed infrastructure rather than inside the notebook.

#### Acceptance Criteria

1. THE Notebook SHALL configure a SageMaker PyTorch Estimator that references the Training_Script as its entry point.
2. THE Notebook SHALL launch the Training_Job on SageMaker managed training infrastructure rather than executing training within the Notebook process.
3. THE Notebook SHALL pass the S3_Bucket location of the Dataset_Subset to the Training_Job as its input data channel.
4. THE Estimator SHALL be configured with hyperparameter values sized to keep the Training_Job short for demonstration purposes.
5. WHEN the Training_Job completes, THE Notebook SHALL report the S3 location of the produced Model artifact.
6. IF the Training_Job fails, THEN THE Notebook SHALL surface the SageMaker failure reason to the User.

### Requirement 5

**User Story:** As a data scientist new to SageMaker, I want minimal and well-explained IAM, S3, and packaging setup, so that I can run the demo without unnecessary configuration overhead.

#### Acceptance Criteria

1. THE Notebook SHALL obtain the SageMaker Execution_Role using the SageMaker JupyterLab environment's default role resolution.
2. THE Notebook SHALL determine the S3_Bucket using the SageMaker default bucket when the User does not specify one.
3. WHERE the Demo requires an AWS resource or permission, THE Notebook SHALL state which resource or permission is required and why.
4. THE Demo SHALL limit its Python dependencies to those required to prepare data, launch the Training_Job, and run the Training_Script.
5. THE Demo SHALL declare its Python dependencies in a single dependency specification file.

### Requirement 6

**User Story:** As a presenter, I want the notebook to highlight the Kiro and SageMaker JupyterLab experience, so that the audience sees how these tools support the workflow.

#### Acceptance Criteria

1. THE Notebook SHALL include narration that identifies where Kiro assists in the workflow inside SageMaker JupyterLab.
2. THE Notebook SHALL present the workflow as a continuous story about tracking habitat and land change.
3. THE Demo SHALL run to completion within SageMaker JupyterLab using the deliverables provided, without requiring external tooling beyond AWS and the declared Python dependencies.

### Requirement 7

**User Story:** As a presenter, I want the notebook to show sample images before training and prediction results after training, so that the audience can see what the model learns from and visually confirm that it works.

#### Acceptance Criteria

1. BEFORE launching the Training_Job, THE Notebook SHALL display a grid of sample tiles drawn from the prepared Dataset_Subset, each annotated with its Land_Cover_Class label.
2. WHEN the Training_Job completes successfully, THE Notebook SHALL load the trained Model artifact back into the Notebook for inference.
3. AFTER loading the trained Model, THE Notebook SHALL run inference on a held-out set of tiles and display a grid of those tiles annotated with predicted and actual Land_Cover_Class labels.
4. THE visualizations SHALL run as notebook cells using already-local images and SHALL NOT alter the managed Training_Job.
