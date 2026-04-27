import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
import os
import numpy as np


# Subclass of Torch Dataset to load images and their corresponding ground truth depths
class CILDataset(Dataset):
    def __init__(self, path_to_data : str, test_dataset : bool = False):
        self.path_to_data = path_to_data
        #this flag is used to specify if the dataset is a test dataset (i.e. it does not contain ground truth depth maps) or a training/validation dataset (i.e. it contains ground truth depth maps).
        self.test_dataset = test_dataset

        self.image_paths = sorted([os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.png')])

        if not self.test_dataset:
            self.np_gt_paths = sorted([os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.npy')])

            if len(self.image_paths) != len(self.np_gt_paths):
                raise Exception("Number of images and ground truths do not match.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, str]:
        # load image and convert to tensor
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # image_tensor = transforms.ToTensor()(image)
        if self.test_dataset:
            transformations = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transformations(image)
        else:
            transformations = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.2),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image_tensor = transformations(image)

        if not self.test_dataset:
            # load gt and convert to tensor
            gt_path = self.np_gt_paths[idx]
            gt = np.load(gt_path)
            ground_truth = torch.from_numpy(gt).to(dtype=torch.float32)
            return image_tensor, ground_truth

        # in case it is a test dataset, return the image tensor and the image name (without the path and extension) to be used for saving the predictions later.
        return image_tensor, img_path[-14:-8]




