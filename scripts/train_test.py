import torch
from  torch.utils.data import DataLoader
from scripts.models import Net
import os
from datetime import  datetime

def train(dataloader : DataLoader, model : Net, num_epochs : int, optimizer : torch.optim.Optimizer ,device : torch.device) -> None | str:
    print("Training the model...")
    model.to(device)
    model.train()

    best_model_state = None
    best_loss : float = float("inf")

    for epoch in range(num_epochs):
        total_loss : float = 0.0

        print(f"Epoch: {epoch}/{num_epochs}")
        for batch_idx, (image, gt) in enumerate(dataloader):
            image = image.to(device)
            gt = gt.to(device)

            optimizer.zero_grad()

            out = model.forward(image)

            loss = model.compute_loss(out, gt)
            print(loss.item())
            # loss = torch.nn.MSELoss()(out, gt)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()


            # print current epoch, batch and loss every 100 batches
            if batch_idx % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}')

            if(loss.item() < best_loss):
                best_loss = loss.item()
                best_model_state :  dict = model.state_dict()

        # if best_model_state and batch_idx % 500 == 0:
        #     # Save weights of the best model
        #     print("Saving best model")
        #     saved_models_path: str = "./models/"
        #     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        #     model_name: str = f"{model.__class__.__name__}_best_{timestamp}_epochs_{num_epochs}.pth"
        #     os.makedirs(saved_models_path, exist_ok=True)
        #     torch.save(best_model_state, saved_models_path + model_name)


    print("Training finished")

    if best_model_state:
        # Save weights of the best model
        print("Saving best model")
        saved_models_path : str = "./models/"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_name : str = f"{model.__class__.__name__}_best_{timestamp}.pth"
        os.makedirs(saved_models_path, exist_ok=True)
        torch.save(best_model_state, saved_models_path + model_name)

        return saved_models_path + model_name

    return None

def eval(dataloader : DataLoader, model : Net, device : torch.device):
    print("Evaluating the model...")
    model.to(device)

    for batch_idx, (image, gt) in enumerate(dataloader):
        print(f"Batch: {batch_idx}/{len(dataloader)}")
        image = image.to(device)
        gt = gt.to(device)
        out = model.forward(image)
        loss = model.compute_loss(out, gt)
        print(loss.item())
    model.eval()