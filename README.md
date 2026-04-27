# Monocular Depth Estimation

This project trains a model for monocular depth estimation, a computer vision task that involves predicting the depth of a scene from a single RGB image.

## Project Structure

The project is organized into the following directories and files:

- `main.py`: The main script for training, evaluating, and testing the model.
- `scripts/`: Contains the core Python modules for the project.
  - `dataset.py`: Defines the `CILDataset` class for loading and preprocessing the dataset.
  - `models.py`: Contains the definitions for the neural network models (e.g., `CNN`, `CNNSmall`).
  - `train_test.py`: Includes the `train`, `eval`, and `run_grading_tests` functions for training, evaluating, and testing the model.
  - `create_submission.py`: A script to generate the `submission.csv` file for the Kaggle competition.
- `models/`: This directory is used to save and load model checkpoints.

## How to Use

### Prerequisites

- Python 3.x
- PyTorch
- Other dependencies (e.g., `argparse`, `os`)

### Training the Model

To train the model, run the `main.py` script with the desired arguments.

**Arguments:**

- `--student-cluster`: Use the dataset path on the student cluster.
- `--batch-size`: The batch size for training (default: 8).
- `--num-epochs`: The number of epochs for training (default: 5).
- `--model`: The name of the model to use (e.g., `CNN`, `CNNSmall`, default `CNN`).

**Example:**

```bash
python main.py --batch-size 16 --num-epochs 10 --model CNNSmall
```

### Running Grading Tests

To run the grading tests and generate a submission file, use the `--grading_tests` flag.

**Arguments:**

- `--grading_tests`: Run the grading tests instead of training.
- `--checkpoint`: The name of the model checkpoint file to use for the tests.
- `--student-cluster`: Use the dataset path on the student cluster.
- `--model`: The name of the model to use for the tests (e.g., `CNN`, `CNNSmall`, default `CNN`).

**Example:**

```bash
python main.py --grading_tests --checkpoint best_model.pth --student-cluster
```

This will run the tests and create a `submission.csv` file in the `scripts` directory.
