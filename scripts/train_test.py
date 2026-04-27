import torch
from torch.utils.data import DataLoader
from scripts.models import Net
import os
from datetime import datetime
import time
import numpy as np


def train(train_loader: DataLoader, test_loader: DataLoader, model: Net, num_epochs: int,
          optimizer: torch.optim.Optimizer, device: torch.device) -> None | str:
    print("Training the model...")
    model.to(device)

    best_model_state = None
    best_loss: float = float("inf")


    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss: float = 0.0

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        for batch_idx, (image, gt) in enumerate(train_loader):
            image = image.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            out = model.forward(image)
            loss = model.compute_loss(out, gt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # print current epoch, batch and loss every 100 batches
            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader.dataset)}], Loss: {loss.item():.4f}')

            #check if a better model is found and save its state dict if so (TODO: check if this actually improves the results or just leads to overfitting)
            if (loss.item() < best_loss):
                best_loss = loss.item()
                best_model_state: dict = model.state_dict()

        if best_model_state:
            print(f"Saving best model")
            saved_models_path: str = "./models/"
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name: str = f"{model.__class__.__name__}_best_{timestamp}_epoch_{epoch}.pth"
            os.makedirs(saved_models_path, exist_ok=True)
            torch.save(best_model_state, saved_models_path + model_name)

        # switch model to eval and compute the test loss
        model.eval()
        test_loss: float = 0.0
        with torch.no_grad():
            for batch_idx, (image, gt) in enumerate(test_loader):
                image = image.to(device)
                gt = gt.to(device)
                out = model.forward(image)
                loss = model.compute_loss(out, gt)
                test_loss += loss.item()

        print(f"Test Loss in epoch {epoch}/{num_epochs}: {test_loss / len(test_loader)}")

        print(f"Epoch duration (in minutes): {(time.time() - epoch_start_time) / 60:.2f}")

    print("Training finished")

    if best_model_state:
        # Save weights of the best model
        print("Saving best model")
        saved_models_path: str = "./models/"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name: str = f"{model.__class__.__name__}_best_{timestamp}.pth"
        os.makedirs(saved_models_path, exist_ok=True)
        torch.save(best_model_state, saved_models_path + model_name)

        return saved_models_path + model_name

    return None


def eval(dataloader: DataLoader, model: Net, device: torch.device):
    print("Evaluating the model...")
    model.to(device)
    total_loss: float = 0.0
    with torch.no_grad():
        for batch_idx, (image, gt) in enumerate(dataloader):
            print(f"Batch: {batch_idx}/{len(dataloader)}")
            image = image.to(device)
            gt = gt.to(device)
            out = model.forward(image)
            loss = model.compute_loss(out, gt)
            total_loss += loss.item()

        print(f"Average loss: {total_loss / len(dataloader.dataset)}")
        model.eval()


def run_grading_tests(data_loader: DataLoader, model: Net, device: torch.device):
    print("Running grading tests...")
    # results = torch.zeros((len(data_loader.dataset), 560, 560), dtype=torch.float32)

    model.to(device)
    # print(results.shape[0])
    with torch.no_grad():
        for batch_idx, (image, name) in enumerate(data_loader):
            print(f"Batch: {batch_idx}/{len(data_loader)}")
            image = image.to(device)
            out = model.forward(image)

            for idx in range(len(out)):
                # results[batch_idx*data_loader.batch_size : min(batch_idx*data_loader.batch_size+data_loader.batch_size, results.shape[0]), :, :] = out
                path_to_test_result: str = os.path.join("./results", "test_" + str(name[idx]) + ".npy")
                os.makedirs("./results", exist_ok=True)
                np.save(path_to_test_result, out[idx, :, :].cpu().numpy())

        print(f"Wrote results to ./results/")
