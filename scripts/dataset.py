from tkinter import image_types

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as vision_F
import os
import cv2 as cv
cv.setNumThreads(0)  # to prevent deadlocks when using opencv with pytorch dataloader
import numpy as np


# Subclass of Torch Dataset to load images and their corresponding ground truth depths
class CILDataset(Dataset):
    def __init__(self, path_to_data : str, test_dataset : bool = False):
        # path to the directory containing dataset images and ground truths
        self.path_to_data = path_to_data
        #this flag is used to specify if the dataset is a test dataset (i.e. it does not contain ground truth depth maps) or a training/validation dataset (i.e. it contains ground truth depth maps).
        self.test_dataset = test_dataset
        # retrieve sorted list of all PNG image files in the dataset folder
        self.image_paths = sorted([os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.png')])

        # load groundtruth paths if it is not a test dataset
        if not self.test_dataset:
            # retrieve sorted list of all ground truth numpy files
            self.np_gt_paths = sorted([os.path.join(path_to_data, f) for f in os.listdir(path_to_data) if f.endswith('.npy')])

            # ensure that the number of images matches the number of ground truth depth maps
            if len(self.image_paths) != len(self.np_gt_paths):
                raise Exception("Number of images and ground truths do not match.")

        # variable that will be set to true if the loaded image was randomly flipped
        self.flipped = False

    # define an empty constructor - needed to create "reference objects" of the dataset type in main.py, which are then used to create the actual datasets with the real constructor.
    @classmethod
    def empty_constructor(cls):
        obj = cls.__new__(cls)
        # initialize attributes with default empty values
        obj.path_to_data = ""
        obj.test_dataset = False
        obj.image_paths = []
        obj.np_gt_paths = []
        obj.flipped = False
        return obj

    # function that returns the size of the dataset
    def __len__(self) -> int:
        return len(self.image_paths)


    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, str]:
        # load image and convert to tensor
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # define standard transformations: gaussian blur to smooth high frequency noise, convert to tensor, and normalize to [-1, 1] range
        transformations = transforms.Compose([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        if self.test_dataset:

            # apply transformations to the test image
            image_tensor = transformations(image)

            # in case it is a test dataset, return the image tensor and the image name (without the path and extension) to be used for saving the predictions later.
            return image_tensor, img_path[-14:-8]
        else:

            # load gt and convert to tensor
            gt_path = self.np_gt_paths[idx]
            gt = np.load(gt_path)
            ground_truth = torch.from_numpy(gt).to(dtype=torch.float32)

            # apply standard transformations to the training/validation image
            image_tensor = transformations(image)

            # apply random color jittering with a 50% probability as data augmentation
            if torch.rand(1) < 0.5:
                color_transform = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
                image_tensor = color_transform(image_tensor)



            # apply random horizontal flip with a 50% probability to both image and ground truth
            if torch.rand(1) < 0.5:
                image_tensor = vision_F.hflip(image_tensor)
                ground_truth = vision_F.hflip(ground_truth)
                self.flipped = True

            else:
                self.flipped = False

            return image_tensor, ground_truth




# Subclass of CILDataset that computes and incorporates Canny edge maps alongside standard inputs
class CannyDataset(CILDataset):
    # fallback empty constructor that returns None
    def __init__(self) -> None:
        return None

    # constructor to initialize the dataset path and test flag using the parent class init
    def __init__(self, path_to_data : str, test_dataset : bool = False):
        super().__init__(path_to_data, test_dataset)
        self.path_to_data = path_to_data

    # retrieve image, Canny edges, and ground truth (or image name) for a given index
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, str]:

        # retrieve the transformed image tensor and ground truth (or name) from the parent dataset class
        res = super().__getitem__(idx)
        image_tensor : torch.Tensor = res[0]
        gt_or_name : str | torch.Tensor = res[1]



        # load image, convert to grayscale, run canny edge detection
        img_path = self.image_paths[idx]
        img = cv.imread(img_path)

        # convert the loaded image from BGR (OpenCV default) to grayscale
        img_grayscale= cv.cvtColor(img, cv.COLOR_RGB2GRAY)


        # run the Canny edge detector to extract contours
        edges = cv.Canny(img_grayscale, 100, 200)

        # convert the edges back to a tensor and concatenate it with the original image tensor
        edges_tensor = torch.from_numpy(edges).unsqueeze(0).float() / 255.0

        # apply Gaussian blur to smooth the binary edge map
        transformations = transforms.Compose([
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.05, 0.05)),
        ])
        edges_tensor = transformations(edges_tensor)

        # if the parent class randomly flipped the main image, we flip the edge map accordingly
        if self.flipped:
            edges_tensor = vision_F.hflip(edges_tensor)

        # return the image, Canny edges, and either ground truth or the image name
        return image_tensor, edges_tensor, gt_or_name






