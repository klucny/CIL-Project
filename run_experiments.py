import torch
import torch.nn.functional as F
from scripts.models import CNN, CannyCNNSkip
# Assuming CNNASPP is in models.py from our previous step
from scripts.models import CNNASPP 

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
        "BaseCNN": CNN,
        "CNNASPP": CNNASPP,
        "CannyCNNSkip": CannyCNNSkip
    }

    # Define loss functions to test (None uses your default SiRMSE loss from the Net class)
    losses_to_test = {
        "SiRMSE": None, 
        "L1_Loss": l1_loss,
        "MSE_Loss": mse_loss
    }

    results = {}

    for model_name, ModelClass in models_to_test.items():
        for loss_name, custom_loss_fn in losses_to_test.items():
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
                cleanup=5
            )

            results[f"{model_name}_{loss_name}"] = best_model_path

    print("\n--- Ablation Study Complete ---")
    for exp, path in results.items():
        print(f"{exp}: {path}")
        
    return results
