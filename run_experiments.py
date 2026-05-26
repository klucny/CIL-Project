import argparse

import torch
import torch.nn.functional as F
from scripts.dataset import CannyDataset
from scripts.models import CNN, CannyCNNSkip
# Assuming CNNASPP is in models.py from our previous step
from scripts.models import CNNASPP
from scripts.train_test import train
from torch.utils.data import DataLoader, random_split, Dataset 

# --- 1. Define Alternative Loss Functions ---

def l1_loss(pred, target, eps=1e-9):
    """Standard L1 (Mean Absolute Error) Loss, masking out empty ground truth pixels."""
    gt_mask = (target > eps)
    if torch.sum(gt_mask) == 0:
        raise Exception("No valid pixels in given image, cannot compute loss")
    
    preds_safe = torch.clamp(pred, min=eps)
    return F.l1_loss(preds_safe[gt_mask], target[gt_mask])

def mse_loss(pred, target, eps=1e-9):
    """Standard L2 (Mean Squared Error) Loss, masking out empty ground truth pixels."""
    gt_mask = (target > eps)
    if torch.sum(gt_mask) == 0:
        raise Exception("No valid pixels in given image, cannot compute loss")
    
    preds_safe = torch.clamp(pred, min=eps)
    return F.mse_loss(preds_safe[gt_mask], target[gt_mask])

# --- 2. Experiment Runner Function ---

def run_ablation_study(train_loader, test_loader, device, num_epochs=20):
    # Define models to test
    models_to_test = {
        #"BaseCNN": CNN, # DONE
        #"CNNASPP": CNNASPP,
        "CannyCNNSkip": CannyCNNSkip
    }

    # Define loss functions to test (None uses your default SiRMSE loss from the Net class)
    losses_to_test = {
        "SiRMSE": None, 
        #"L1_Loss": l1_loss, # Not useful
        #"MSE_Loss": mse_loss # Not useful
    }

    results = {}

    run = 0
    for model_name, ModelClass in models_to_test.items():
        for loss_name, custom_loss_fn in losses_to_test.items():
            run += 1
            # Already done:
            if model_name == "BaseCNN" and loss_name == "SiRMSE":
                print(f"Skipping {model_name} with {loss_name} since it's already done.")
                continue
            if model_name == "BaseCNN" and loss_name == "L1_Loss":
                print(f"Skipping {model_name} with {loss_name} since it's already done.")
                continue

            print(f"\n{'='*50}")
            print(f"Starting run: Model = {model_name} | Loss = {loss_name}")
            print(f"{'='*50}\n")

            # 1. Initialize fresh model
            model = ModelClass().to(device)

            # 2. Override the compute_loss method dynamically if a custom loss is selected
            if custom_loss_fn is not None:
                # We use a lambda to bypass the 'self' argument that your train loop 
                # expects when calling model.compute_loss(out, gt)
                model.compute_loss = lambda pred, target, fn=custom_loss_fn: fn(pred, target)

            # 3. Setup Optimizer (AdamW is highly recommended over standard Adam)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

            # NOTE: For WandB to log correctly, you may want to quickly update your `train_test.py` 
            # wandb.init() call to use: name=f"{model.__class__.__name__}_{loss_name}"
            
            # 4. Run your existing train function
            best_model_path = train(
                train_loader=train_loader,
                test_loader=test_loader,
                model=model,
                num_epochs=num_epochs,
                optimizer=optimizer,
                device=device,
                start_epoch=0,
                cleanup=5,
                loss_name=loss_name
            )

            # Somehow bugged, so we only do one run
            if run == 1:
                break

            results[f"{model_name}_{loss_name}"] = best_model_path
        
        if run == 1:
            break

    print("\n--- Ablation Study Complete ---")
    for exp, path in results.items():
        print(f"{exp}: {path}")
        
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monocular Depth Estimation Training')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    TRAIN_TEST_SPLIT_RATIO = 0.8  # e.g. 0.8 -> 80% of the data used for training, 20% for testing
    NUM_EPOCHS = args.num_epochs

    print(f"Running experiments with num_epochs={NUM_EPOCHS} and batch_size={BATCH_SIZE}")

    dataset_type : Dataset= CannyDataset.empty_constructor() # We can default to CannyDataset as it inherits from CILDataset and has the same interface, and supports all models
    dataset = type(dataset_type)('./data/monodepth_kaggle2026/train')
    generator = torch.Generator().manual_seed(10)

    dataset_size = len(dataset)
    train_size = int(TRAIN_TEST_SPLIT_RATIO * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)

    print(f"Dataset size: {dataset_size} | Train size: {train_size} | Test size: {test_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Starting ablation study...")
    run_ablation_study(train_loader, test_loader, device, num_epochs=NUM_EPOCHS)