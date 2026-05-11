import torch
from torch.utils.data import DataLoader
from scripts.dataset import CILDataset, CannyDataset
from scripts.models import Net, Canny
import os
from datetime import datetime
import time
import numpy as np


def train(train_loader: DataLoader, test_loader: DataLoader, model: Net, num_epochs: int,
          optimizer: torch.optim.Optimizer, device: torch.device) -> None | str:
    print("Training the model...")
    model.to(device)

    best_model_state = None
    best_train_loss: float = float("inf")
    best_test_loss: float = float("inf")
    # NEEDED FOR AMP
    scaler = torch.amp.GradScaler('cuda')


    for epoch in range(num_epochs):
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

                # Reduce precsion for faster training with AMP
                with torch.amp.autocast('cuda'):
                    out = model.forward(image, edges)
                    loss = model.compute_loss(out, gt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # print current epoch, batch and loss
                if batch_idx % 50 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

                # check if a better model is found and save its state dict if so (TODO: check if this actually improves the results or just leads to overfitting)
                if (loss.item() < best_train_loss):
                    best_train_loss = loss.item()
                    best_model_state: dict = model.state_dict()

            print(f"Training loss: {train_loss / len(train_loader)}")

            # switch model to eval and compute the test loss
            model.eval()
            test_loss: float = 0.0
            with torch.no_grad():
                for batch_idx, (image, edges, gt) in enumerate(test_loader):
                    image = image.to(device)
                    edges = edges.to(device)
                    gt = gt.to(device)
                    
                    # AMP 
                    with torch.amp.autocast('cuda'):
                        out = model.forward(image, edges)
                        loss = model.compute_loss(out, gt)
                        
                    test_loss += loss.item()
                    
                    if batch_idx % 50 == 0:
                        print(f'   Test Batch [{batch_idx}/{len(test_loader)}], Temp Loss: {loss.item():.4f}')
        
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict() # GRAB STATE HERE
                print(f"Saving best model (Test loss improved)")
                saved_models_path: str = "./models/"
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                model_name: str = f"{model.__class__.__name__}_best_{timestamp}_epoch_{epoch+1}.pth"
                os.makedirs(saved_models_path, exist_ok=True)
                torch.save(best_model_state, saved_models_path + model_name)

        else:
            for batch_idx, (image, gt) in enumerate(train_loader):
                image = image.to(device)
                gt = gt.to(device)
                optimizer.zero_grad()

                # AMP Autocast
                with torch.amp.autocast('cuda'):
                    out = model.forward(image)
                    loss = model.compute_loss(out, gt)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # print current epoch, batch and loss every 100 batches
                if batch_idx % 2000 == 0:
                    print(
                        f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

                #check if a better model is found and save its state dict if so (TODO: check if this actually improves the results or just leads to overfitting)
                if (loss.item() < best_train_loss):
                    best_train_loss = loss.item()
                    best_model_state: dict = model.state_dict()

            print(f"Training loss: {train_loss / len(train_loader)}")


            # switch model to eval and compute the test loss
            model.eval()
            test_loss: float = 0.0
            with torch.no_grad():
                for batch_idx, (image, edges, gt) in enumerate(test_loader):
                    image = image.to(device)
                    edges = edges.to(device)
                    gt = gt.to(device)
                    
                    # AMP
                    with torch.amp.autocast('cuda'):
                        out = model.forward(image, edges)
                        loss = model.compute_loss(out, gt)
                        
                    test_loss += loss.item()
                    
                    if batch_idx % 50 == 0:
                        print(f'   Test Batch [{batch_idx}/{len(test_loader)}], Temp Loss: {loss.item():.4f}')

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict() # GRAB STATE HERE
                print(f"Saving best model (Test loss improved)")
                saved_models_path: str = "./models/"
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                model_name: str = f"{model.__class__.__name__}_best_{timestamp}_epoch_{epoch+1}.pth"
                os.makedirs(saved_models_path, exist_ok=True)
                torch.save(best_model_state, saved_models_path + model_name)



        print(f"Test Loss in epoch {epoch+1}/{num_epochs}: {test_loss / len(test_loader)}")

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
        if isinstance(model, Canny):
            #TODO: adapt to use edges
            for batch_idx, (image, edges, gt) in enumerate(dataloader):
                print(f"Batch: {batch_idx}/{len(dataloader)}")
                image = image.to(device)
                gt = gt.to(device)
                edges = edges.to(device)
                out = model.forward(image, edges)
                loss = model.compute_loss(out, gt)
                total_loss += loss.item()
        else:
            for batch_idx, (image, gt) in enumerate(dataloader):
                print(f"Batch: {batch_idx}/{len(dataloader)}")
                image = image.to(device)
                gt = gt.to(device)
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
            for batch_idx, (image, edges, name) in enumerate(data_loader):
                image = image.to(device)
                edges = edges.to(device)
                
                # AMP
                with torch.amp.autocast('cuda'):
                    out = model.forward(image, edges)

                for idx in range(len(out)):
                    path_to_test_result: str = os.path.join("./results", "test_" + str(name[idx]) + ".npy")
                    os.makedirs("./results", exist_ok=True)
                    # float32 apparently prevents weird edge case error
                    np.save(path_to_test_result, out[idx, :, :].to(torch.float32).cpu().numpy())
        else:
            for batch_idx, (image, name) in enumerate(data_loader):
                image = image.to(device)
                
                # AMP
                with torch.amp.autocast('cuda'):
                    out = model.forward(image)

                for idx in range(len(out)):
                    path_to_test_result: str = os.path.join("./results", "test_" + str(name[idx]) + ".npy")
                    os.makedirs("./results", exist_ok=True)
                    # float32 apparently prevents weird edge case error
                    np.save(path_to_test_result, out[idx, :, :].to(torch.float32).cpu().numpy())

        print(f"Wrote results to ./results/")
