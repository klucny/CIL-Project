# Monocular Depth Estimation

This project trains a model for monocular depth estimation, a computer vision task that involves predicting the depth of a scene from a single RGB image.

## AI Usage Declaration

*Tool used:* Gemini \
*Files affected:* main.py, models.py, train_test.py, dataset.py \
*Purpose:* Used for Python syntax and how to use specific libraries.

## Project Structure

The project is organized into the following directories and files:

- `main.py`: The main script for training, evaluating, and testing the model.
- `scripts/`: Contains the core Python modules for the project.
  - `dataset.py`: Defines the `CILDataset` and `CannyDataset` classes for loading and preprocessing the dataset. `CannyDataset` includes Canny edge detection preprocessing.
  - `models.py`: Contains the definitions for the neural network models (`CNN`, `CNNSmall`, `CannyCNN`).
  - `train_test.py`: Includes the `train`, `eval`, and `run_grading_tests` functions for training, evaluating, and testing the model.
  - `create_submission.py`: A script to generate the `submission.csv` file for the Kaggle competition.
- `models/`: This directory is used to save and load model checkpoints.
- `results/`: This directory stores the depth predictions generated during grading tests.

## How to Use

### Prerequisites

- Python 3.x
- PyTorch 2.11.0
- OpenCV (for Canny edge detection)
- Other dependencies listed in `requirements.txt`

### Available Models

The project provides three neural network models for depth estimation:

- **CNN**: A U-Net-style encoder-decoder architecture based on ResNet50. Uses pre-trained weights and features skip connections between encoder and decoder layers.
- **CNNSmall**: A smaller, simpler CNN model suitable for faster training and inference with lower memory requirements.
- **CannyCNN**: An enhanced version of CNN that incorporates Canny edge detection features alongside RGB input. The edge information is processed through an additional encoder and element-wise multiplied with the RGB features at the bottleneck layer.
- **CannyCNNSkip**: Our final model with ASPP, Canny Edges and an Attention Gate.
- **CNNASPP**: Out model with only the ASPP for baseline comparison.

The default model is **CannyCNNSkip**. To use a different model, specify it via the `--model` argument.

### Training the Model

To train the model, run the `main.py` script with the desired arguments.

**Arguments:**

- `--student-cluster`: Use the dataset path on the student cluster.
- `--batch-size`: The batch size for training (default: 8).
- `--num-epochs`: The number of epochs for training (default: 5).
- `--model`: The name of the model to use (`CNN`, `CannyCNNSkip`, ..., default: `CannyCNNSkip`).
- `--resume`: The file name of the checkpoint to continue training from.
- `--cleanup`: The iteration of model checkpoints to keep for saving file space, `0` means to keep all (default: `5`).

**Example:**

```bash
python main.py --batch-size 16 --num-epochs 10 --model CannyCNNSkip
```

### Running Evaluation Tests

To run evaluation tests on the entire dataset, use the `--eval` flag with a checkpoint file.

**Arguments:**

- `--eval`: The name of the model checkpoint file to use for evaluation on the whole dataset.
- `--student-cluster`: Use the dataset path on the student cluster.
- `--model`: The name of the model to use for evaluation (e.g., `CNN`, `CNNSmall`, `CannyCNN`, default: `CannyCNN`).
- `--batch-size`: The batch size for evaluation (default: 8).

**Example:**

```bash
python main.py --eval CNN_best_20260427-155325.pth --model CannyCNN --batch-size 16
```

### Running Grading Tests

To run the grading tests and generate a submission file, use the `--grading_tests` flag.

**Arguments:**

- `--grading_tests`: Run the grading tests instead of training.
- `--checkpoint`: The name of the model checkpoint file to use for the tests (required).
- `--student-cluster`: Use the dataset path on the student cluster.
- `--model`: The name of the model to use for the tests (`CNN`, `CNNSmall`, `CannyCNN`, default: `CannyCNNSkip`).
- `--batch-size`: The batch size for testing (default: 8).

**Example:**

```bash
python main.py --grading_tests --checkpoint CannyCNNSkip_best_20260427-155325.pth --model CannyCNNSkip
```

This will run the tests on the test dataset and create a `submission.csv` file in the project directory.
