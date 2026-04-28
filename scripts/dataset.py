from tkinter import image_types

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as vision_F
import os
import cv2 as cv
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

        self.flipped = False

    @classmethod
    def empty_constructor(cls):
        obj = cls.__new__(cls)
        obj.path_to_data = ""
        obj.test_dataset = False
        obj.image_paths = []
        obj.np_gt_paths = []
        obj.flipped = False
        return obj

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, str]:
        # load image and convert to tensor
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        transformations = transforms.Compose([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.test_dataset:

            image_tensor = transformations(image)

            # in case it is a test dataset, return the image tensor and the image name (without the path and extension) to be used for saving the predictions later.
            return image_tensor, img_path[-14:-8]
        else:

            # load gt and convert to tensor
            gt_path = self.np_gt_paths[idx]
            gt = np.load(gt_path)
            ground_truth = torch.from_numpy(gt).to(dtype=torch.float32)

            image_tensor = transformations(image)

            if torch.rand(1) < 0.5:
                color_transform = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                image_tensor = color_transform(image_tensor)



            if torch.rand(1) < 0.5:
                image_tensor = vision_F.hflip(image_tensor)
                ground_truth = vision_F.hflip(ground_truth)
                self.flipped = True

            return image_tensor, ground_truth




class CannyDataset(CILDataset):
    def __init__(self) -> None:
        return None

    def __init__(self, path_to_data : str, test_dataset : bool = False):
        super().__init__(path_to_data, test_dataset)
        self.path_to_data = path_to_data

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:

        res = super().__getitem__(idx)
        image_tensor : torch.Tensor = res[0]
        gt_or_name : str | torch.Tensor = res[1]



        # load image, convert to grayscale, run canny edge detection
        img_path = self.image_paths[idx]
        img = cv.imread(img_path)

        img_grayscale= cv.cvtColor(img, cv.COLOR_RGB2GRAY)


        edges = cv.Canny(img_grayscale, 100, 200)

        # convert the edges back to a tensor and concatenate it with the original image tensor
        edges_tensor = torch.from_numpy(edges).unsqueeze(0).float() / 255.0

        transformations = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 0.05)),
        ])
        edges_tensor = transformations(edges_tensor)

        if self.flipped:
            edges_tensor = vision_F.hflip(edges_tensor)

        return image_tensor, edges_tensor, gt_or_name






