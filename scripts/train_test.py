import torch
from torch.utils.data import DataLoader
from scripts.dataset import CILDataset, CannyDataset
from scripts.models import Net, Canny
import os
from datetime import datetime
import time
import numpy as np
import wandb
import copy

def train(train_loader: DataLoader, test_loader: DataLoader, model: Net, num_epochs: int,
          optimizer: torch.optim.Optimizer, device: torch.device, start_epoch: int = 0, cleanup: int = 5) -> None | str:
    print("Training the model...")
    if cleanup > 0:
        print (f"Checkpoint cleanup enabled: Old checkpoints will be deleted every {cleanup} epochs.")
    model.to(device)

    # 1. Generate a single timestamp for this entire run
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    saved_models_path: str = "./models/"
    os.makedirs(saved_models_path, exist_ok=True)

    # logging with wandb
    wandb.init(
        project="cil-depth-estimation", # This creates a project folder in your WandB dashboard
        name=model.__class__.__name__,  # This automatically names the run "CannyCNNSkip"
        config={
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
        }
    )

    best_model_state = None
    best_train_loss: float = float("inf")
    best_test_loss: float = float("inf")

    # learning rate scheduler
    last_epoch = (start_epoch * len(train_loader)) - 1 if start_epoch > 0 else -1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.0003,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs,
            pct_start=0.3,
            last_epoch=last_epoch
        )
    # mixed precision training with torch.amp
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss: float = 0.0

        print(f"Epoch: {epoch + 1}/{num_epochs}")

        if isinstance(model, Canny):
            # TODO: adapt to use edges
            for batch_idx, (image, edges, gt) in enumerate(train_loader):
                image = image.to(device)
                edges = edges.to(device)
                gt = gt.to(device)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    out = model.forward(image, edges)
                    loss = model.compute_loss(out, gt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()

                train_loss += loss.item()

                # print current epoch, batch and loss every 100 batches
                if batch_idx % 100 == 0:
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

                    wandb.log({"train_loss": loss.item(), "epoch": epoch + 1, "batch": batch_idx,})

            print(f"Training loss: {train_loss / len(train_loader)}")

            # switch model to eval and compute the test loss
            model.eval()
            test_loss: float = 0.0
            with torch.no_grad():
                for batch_idx, (image, edges, gt) in enumerate(test_loader):
                    image = image.to(device)
                    gt = gt.to(device)
                    edges= edges.to(device)
                    with torch.amp.autocast('cuda'):
                        out = model.forward(image, edges)
                        loss = model.compute_loss(out, gt)
                    test_loss += loss.item()

            test_loss_avg = test_loss / len(test_loader)

            # ONLY update the best model state here, based on test performance
            if test_loss_avg < best_test_loss:
                best_test_loss = test_loss_avg
                best_model_state: dict = {
                        'epoch': epoch + 1,
                        'model_state_dict': copy.deepcopy(model.state_dict()),
                        'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())
                    }

                print(f"New best test loss ({best_test_loss:.4f})! Saving best model...")
                # 2. Use the run_timestamp instead of generating a new one
                model_name: str = f"{model.__class__.__name__}_best_{run_timestamp}_epoch_{epoch + 1}.pth"
                torch.save(best_model_state, saved_models_path + model_name)

        else:
            for batch_idx, (image, gt) in enumerate(train_loader):
                image = image.to(device)
                gt = gt.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    out = model.forward(image)
                    loss = model.compute_loss(out, gt)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                train_loss += loss.item()

                # print current epoch, batch and loss every 100 batches
                if batch_idx % 100 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

            print(f"Training loss: {train_loss / len(train_loader)}")


            # switch model to eval and compute the test loss
            model.eval()
            test_loss: float = 0.0
            with torch.no_grad():
                for batch_idx, (image, gt) in enumerate(test_loader):
                    image = image.to(device)
                    gt = gt.to(device)
                    with torch.amp.autocast('cuda'):
                        out = model.forward(image)
                        loss = model.compute_loss(out, gt)
                    test_loss += loss.item()

            test_loss_avg = test_loss / len(test_loader)

            if test_loss_avg < best_test_loss:
                best_test_loss = test_loss_avg
                best_model_state: dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(optimizer.state_dict())
                }

                print(f"New best test loss ({best_test_loss:.4f})! Saving best model...")
                # 2. Use the run_timestamp instead of generating a new one
                model_name: str = f"{model.__class__.__name__}_best_{run_timestamp}_epoch_{epoch + 1}.pth"
                torch.save(best_model_state, saved_models_path + model_name)


        test_loss_avg = test_loss / len(test_loader)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Test Loss in epoch {epoch+1}/{num_epochs}: {test_loss / len(test_loader)}")
        print(f"Current Learning Rate: {current_lr}")
        wandb.log({"test_loss": test_loss_avg, "epoch": epoch + 1, "learning_rate": current_lr})

        print(f"Epoch duration (in minutes): {(time.time() - epoch_start_time) / 60:.2f}")

        # 3. Cleanup logic: Delete epoch % cleanup checkpoints that belong to this run - if cleanup is 0, this is disabled
        current_epoch = epoch + 1
        if cleanup > 0 and current_epoch % cleanup == 0:
            print(f"--- Running cleanup for intermediate models (Epoch {current_epoch}) ---")
            for f in os.listdir(saved_models_path):
                # Ensure we only touch files from THIS run that have an epoch number
                if run_timestamp in f and "_epoch_" in f:
                    try:
                        # Extract the epoch number from the filename
                        ep_str = f.split("_epoch_")[1].split(".pth")[0]
                        ep_num = int(ep_str)
                        # Delete if it's not a multiple of cleanup
                        if ep_num % cleanup != 0:
                            os.remove(os.path.join(saved_models_path, f))
                            print(f"Cleaned up space: Deleted {f}")
                    except ValueError:
                        pass # Ignore if filename structure is somehow unexpected

    print("Training finished")

    if best_model_state:
        print("Saving final best model")
        model_name: str = f"{model.__class__.__name__}_best_{run_timestamp}_final.pth"
        torch.save(best_model_state, saved_models_path + model_name)

        return saved_models_path + model_name

    return None


def eval(dataloader: DataLoader, model: Net, device: torch.device):
    print("Evaluating the model...")
    model.to(device)
    total_loss: float = 0.0
    with torch.no_grad():
        if isinstance(model, Canny):
            #TODO: adapt to use edges
            for batch_idx, (image, edges, gt) in enumerate(dataloader):
                print(f"Batch: {batch_idx}/{len(dataloader)}")
                image = image.to(device)
                gt = gt.to(device)
                edges = edges.to(device)
                with torch.amp.autocast('cuda'):
                    out = model.forward(image, edges)
                    loss = model.compute_loss(out, gt)
                total_loss += loss.item()
        else:
            for batch_idx, (image, gt) in enumerate(dataloader):
                print(f"Batch: {batch_idx}/{len(dataloader)}")
                image = image.to(device)
                gt = gt.to(device)
                with torch.amp.autocast('cuda'):
                    out = model.forward(image)
                    loss = model.compute_loss(out, gt)
                total_loss += loss.item()

        print(f"Average loss: {total_loss / len(dataloader)}")
        model.eval()


def run_grading_tests(data_loader: DataLoader, model: Net, device: torch.device):
    print("Running grading tests...")
    # results = torch.zeros((len(data_loader.dataset), 560, 560), dtype=torch.float32)

    model.to(device)
    # check which type of dataset it is (Canny or CIL)
    with torch.no_grad():
        if isinstance(model, Canny):
            # TODO: adapt to use edges
            for batch_idx, (image, edges, name) in enumerate(data_loader):
                # print(f"Batch: {batch_idx+1}/{len(data_loader)}")
                image = image.to(device)
                edges = edges.to(device)
                with torch.amp.autocast('cuda'):
                    # Small flip trick to improve performance if the model predicts slightly better on one side of the image than the other (e.g. due to positional encoding or similar)
                    out_standard = model.forward(image, edges)
                    image_flipped = torch.flip(image, dims=[3])
                    edges_flipped = torch.flip(edges, dims=[3])
                    out_flipped = model.forward(image_flipped, edges_flipped)
                    out_unflipped = torch.flip(out_flipped, dims=[2])
                    out = (out_standard + out_unflipped) / 2.0

                
                # write batch outputs to result folder
                for idx in range(len(out)):
                    path_to_test_result: str = os.path.join("./results", "test_" + str(name[idx]) + ".npy")
                    os.makedirs("./results", exist_ok=True)
                    # float32 apparently prevents weird edge case error
                    np.save(path_to_test_result, out[idx, :, :].to(torch.float32).cpu().numpy())
        else:
            for batch_idx, (image, name) in enumerate(data_loader):
                # print(f"Batch: {batch_idx+1}/{len(data_loader)}")
                image = image.to(device)
                with torch.amp.autocast('cuda'):
                    # Same trick as above
                    out_standard = model.forward(image)
                    image_flipped = torch.flip(image, dims=[3])
                    out_flipped = model.forward(image_flipped)
                    out_unflipped = torch.flip(out_flipped, dims=[2])
                    out = (out_standard + out_unflipped) / 2.0

                # write batch outputs to result folder
                for idx in range(len(out)):
                    path_to_test_result: str = os.path.join("./results", "test_" + str(name[idx]) + ".npy")
                    os.makedirs("./results", exist_ok=True)
                    # float32 apparently prevents weird edge case error
                    np.save(path_to_test_result, out[idx, :, :].to(torch.float32).cpu().numpy())

        print(f"Wrote results to ./results/")