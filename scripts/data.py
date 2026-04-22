import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np


# Subclass of Torch Dataset to load images and their corresponding ground truth depths
class CILDataset(Dataset):
    def __init__(self, path_to_data : str):
        self.path_to_data = path_to_data

        self.image_paths = sorted([os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.png')])
        self.np_gt_paths = sorted([os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.npy')])

        if len(self.image_paths) != len(self.np_gt_paths):
            raise Exception("Number of images and ground truths do not match.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:

        # load image and convert to tensor
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image_tensor = transforms.ToTensor()(image)

        # load gt and convert to tensor
        gt_path = self.np_gt_paths[idx]
        gt = np.load(gt_path)
        ground_truth = torch.from_numpy(gt).to(dtype=torch.float32)


        return image_tensor, ground_truth




